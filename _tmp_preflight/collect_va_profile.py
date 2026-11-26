#!/usr/bin/env python3
"""collect_va_profile.py —— 从 vLLM/NPU profiler 产物里抽**小体积**证据包（<300 KB，可 git 推送）。

一个 rank 的 ASCEND_PROFILER_OUTPUT 动辄上百 MB；真正要看的只有几样：
按「算子桶」聚合、按「chunk/step」聚合、top kernel、列结构。

用法（容器内或宿主都行，纯 stdlib）：
  python3 _tmp_preflight/collect_va_profile.py /home/a00821909/va_prof --tag smoke1
  python3 _tmp_preflight/collect_va_profile.py /home/a00821909/va_prof --tag full128k --top 300

产物：_tmp_preflight/profile_bundle/<tag>/
  va_summary.md        人读摘要（桶表 + 步表 + top kernel + 看点）
  va_buckets.csv       按算子桶聚合（indexer/mla/moe/mhc/norm/quant/comm/gemm/other）
  va_steps.csv         按时间间隙切出的「步/ chunk」（每步耗时 + 占比 + top 算子）
  kernel_top.csv       耗时 top N kernel（带 pipe 比率列）
  op_statistic_top.csv 算子聚合 top N（原样截断）
  step_trace_time.csv  每步耗时（原样复制，若有）
  meta.txt             来源/大小/列名
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import re
import shutil
from collections import defaultdict

BUNDLE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_bundle")

# torch_npu 导出的 kernel_details.csv 用这些列名；不同 CANN 小版本略有差异，做别名兼容
ALIASES = {
    "name": ["Name", "Kernel Name", "kernel_name", "OP Name", "Op Name"],
    "type": ["Type", "Kernel Type", "OP Type"],
    "dur": ["Duration(us)", "Duration", "duration"],
    "start": ["Start Time(us)", "Start Time", "StartTime", "Timestamp"],
    "core": ["Accelerator Core", "Core Type", "accelerator_core"],
    "stream": ["Stream ID", "Stream Id", "stream_id"],
    "blocks": ["Block Dim", "Block Num", "block_dim"],
    "mac_ratio": ["aic_mac_ratio", "aicore_mac_ratio"],
    "mac_time": ["aic_mac_time(us)", "aic_mac_time"],
    "cube": ["cube_utilization(%)", "cube_utilization"],
}

KERNEL_FIELDS = ["Stream ID", "Name", "Type", "Accelerator Core", "Duration(us)",
                 "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
                 "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
                 "cube_utilization(%)", "Block Dim", "OP State"]

# 算子桶：按名字/类型里的关键字归类。顺序敏感（先匹配先归类）。
BUCKETS = [
    ("indexer", r"indexer|lightning|topk_ind|sparse_index|qli"),
    # 注意：kernel 名是驼峰（SparseFlashMla / HcPre / HcPost），下面已 lower 后再匹配
    ("mla_attn", r"sparse_?flash|flash_?attn|mla|attention|attn"),
    ("mhc", r"hc_?pre|hc_?post|mhc|hyper_?conn|sinkhorn"),
    ("moe", r"moe|grouped_matmul|gmm|dispatch|combine|gating|expert|swiglu"),
    ("comm", r"hccl|allreduce|all_reduce|allgather|all_gather|reduce_scatter|alltoall|all_to_all|hcom"),
    ("quant", r"quant|dequant|per_token|dynamic_quant"),
    ("norm", r"rms_norm|layernorm|layer_norm|norm_|reduce|cast|copy|transpose|contiguous"),
    ("gemm", r"matmul|gemm|bmm|mm_|linear"),
]


def find_csv(root: str) -> tuple[str, list[dict]]:
    """按优先级找一份「kernel 级」CSV，返回 (路径, 行列表)。

    优先级：torch_npu Text 导出的 kernel_details.csv > msprof 导出的 op_summary*.csv
            > 任何 kernel*.csv。同一优先级里取行数最多的那份。
    """
    for pat in ("**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
                "**/kernel_details.csv",
                "**/op_summary*.csv",
                "**/kernel*.csv"):
        best, best_n = None, 0
        for fp in glob.glob(os.path.join(root, pat), recursive=True):
            rows = read_rows(fp)
            if len(rows) > best_n:
                best, best_n = fp, len(rows)
        if best:
            print("[info] 用 CSV: %s  (%d 行, %.1f MB)"
                  % (os.path.relpath(best, root), best_n, os.path.getsize(best) / 1e6))
            return best, read_rows(best)
    return "", []


def read_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], key: str) -> str | None:
    if not rows:
        return None
    fields = list(rows[0].keys())
    for a in ALIASES.get(key, []):
        if a in fields:
            return a
    for f in fields:  # 宽松匹配
        for a in ALIASES.get(key, []):
            if a.lower().replace(" ", "") in f.lower().replace(" ", ""):
                return f
    return None


def fnum(v) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return 0.0


def bucket_of(name: str, typ: str) -> str:
    s = (name + " " + typ).lower()
    for b, pat in BUCKETS:
        if re.search(pat, s):
            return b
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="profiler 根目录（含 *_ascend_pt/ASCEND_PROFILER_OUTPUT）")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--max-kb", type=int, default=280, help="体积上限（KB），默认 280 留余量")
    ap.add_argument("--gap-us", type=float, default=2000.0,
                    help="切步的间隙阈值（us）：相邻 kernel 间隔超过它就算新的一步")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(os.path.normpath(args.root))
    dst = os.path.join(BUNDLE_ROOT, tag)
    os.makedirs(dst, exist_ok=True)

    source, rows = find_csv(args.root)
    if rows:
        rank_dir = os.path.dirname(source)
        kd_size = os.path.getsize(source)
        ap_dir = rank_dir
    else:
        print("[warn] 没有任何可用 CSV，回退解析 chrome trace")
        rows, source = load_trace_rows(args.root)
        rank_dir, kd_size = os.path.dirname(source), os.path.getsize(source)
        ap_dir = rank_dir

    c_name, c_type, c_dur = col(rows, "name"), col(rows, "type"), col(rows, "dur")
    c_start, c_core, c_stream = col(rows, "start"), col(rows, "core"), col(rows, "stream")

    # ---------- 1. 算子桶聚合
    bstat = defaultdict(lambda: {"n": 0, "dur": 0.0, "k": defaultdict(float)})
    total = 0.0
    for r in rows:
        n = (r.get(c_name) or "?") if c_name else "?"
        t = (r.get(c_type) or "") if c_type else ""
        d = fnum(r.get(c_dur)) if c_dur else 0.0
        b = bucket_of(n, t)
        e = bstat[b]
        e["n"] += 1
        e["dur"] += d
        e["k"][n] += d
        total += d
    total = total or 1.0
    with open(os.path.join(dst, "va_buckets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "kernels", "total_us", "pct", "avg_us", "top3_kernels(us)"])
        for b, e in sorted(bstat.items(), key=lambda kv: -kv[1]["dur"]):
            top3 = "; ".join("%s=%.0f" % (k, v)
                             for k, v in sorted(e["k"].items(), key=lambda kv: -kv[1])[:3])
            w.writerow([b, e["n"], "%.0f" % e["dur"], "%.2f%%" % (100 * e["dur"] / total),
                        "%.0f" % (e["dur"] / max(e["n"], 1)), top3])

    # ---------- 2. 切步（按时间间隙）
    steps = []
    if c_start and c_dur and rows:
        seq = []
        for r in rows:
            seq.append((fnum(r.get(c_start)), fnum(r.get(c_dur)),
                        (r.get(c_name) or "?") if c_name else "?"))
        seq.sort()
        cur = {"t0": seq[0][0], "dur": 0.0, "n": 0, "ops": defaultdict(float)}
        prev_end = seq[0][0]
        for st, du, nm in seq:
            if st - prev_end > args.gap_us and cur["n"] > 0:
                steps.append(cur)
                cur = {"t0": st, "dur": 0.0, "n": 0, "ops": defaultdict(float)}
            cur["dur"] = max(cur["dur"], st + du - cur["t0"])
            cur["n"] += 1
            cur["ops"][nm] += du
            prev_end = max(prev_end, st + du)
        if cur["n"]:
            steps.append(cur)
    with open(os.path.join(dst, "va_steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "start_us", "span_us", "kernels", "pct_of_total", "top3_ops(us)"])
        for i, s in enumerate(steps):
            top3 = "; ".join("%s=%.0f" % (k, v)
                             for k, v in sorted(s["ops"].items(), key=lambda kv: -kv[1])[:3])
            w.writerow([i, "%.0f" % s["t0"], "%.0f" % s["dur"], s["n"],
                        "%.2f%%" % (100 * s["dur"] / total), top3])

    # ---------- 3. top kernel
    fields_src = list(rows[0].keys()) if rows else []
    keep_cols = [c for c in KERNEL_FIELDS if c in fields_src]
    if not keep_cols:
        keep_cols = fields_src[:12]
    rows_sorted = sorted(rows, key=lambda r: -fnum(r.get(c_dur)) if c_dur else 0)
    top_rows = rows_sorted[:args.top]
    with open(os.path.join(dst, "kernel_top.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keep_cols)
        for r in top_rows:
            w.writerow([r.get(c, "") for c in keep_cols])

    # ---------- 4. 原样截断 op_statistic / 复制 step_trace_time
    for name, out, key in (("op_statistic.csv", "op_statistic_top.csv", "Total Time(us)"),
                           ("step_trace_time.csv", "step_trace_time.csv", None)):
        src = os.path.join(ap_dir, name)
        if not os.path.isfile(src):
            continue
        if key is None:
            shutil.copyfile(src, os.path.join(dst, out))
            continue
        orows = read_rows(src)
        orows.sort(key=lambda r: -fnum(r.get(key)))
        with open(os.path.join(dst, out), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(orows[0].keys()) if orows else [])
            w.writeheader()
            for r in orows[:args.top]:
                w.writerow(r)

    # ---------- 5. meta + 人读摘要
    with open(os.path.join(dst, "meta.txt"), "w") as f:
        f.write("source_root: %s\nsource_file: %s\nrank_dir: %s\nkernel_details_bytes: %d\n"
                "kernel_rows: %d\ntotal_kernel_us: %.0f\ncolumns: %s\n"
                % (args.root, source, rank_dir, kd_size, len(rows), total, ", ".join(fields_src)))

    lines = ["# vLLM-ascend profiler 摘要（%s）" % tag, "",
             "- 来源: `%s`" % rank_dir,
             "- kernel 数: **%d**，总耗时: **%.1f ms**" % (len(rows), total / 1000.0),
             "- 切步: **%d 步**（间隙阈值 %.0f us）" % (len(steps), args.gap_us), "",
             "## 算子桶", "", "| 桶 | kernel 数 | 总耗时(ms) | 占比 | 平均(us) |", "|---|---|---|---|---|"]
    for b, e in sorted(bstat.items(), key=lambda kv: -kv[1]["dur"]):
        lines.append("| %s | %d | %.1f | %.2f%% | %.0f |"
                     % (b, e["n"], e["dur"] / 1000.0, 100 * e["dur"] / total, e["dur"] / max(e["n"], 1)))
    if steps:
        lines += ["", "## 步（chunk）", "", "| # | span(ms) | kernel 数 | 占比 | top 算子 |", "|---|---|---|---|---|"]
        for i, s in enumerate(steps[:40]):
            top3 = "; ".join("%s=%.0f" % (k, v)
                             for k, v in sorted(s["ops"].items(), key=lambda kv: -kv[1])[:3])
            lines.append("| %d | %.1f | %d | %.2f%% | %s |"
                         % (i, s["dur"] / 1000.0, s["n"], 100 * s["dur"] / total, top3))
    lines += ["", "## Top 25 kernel（按单次耗时）", ""]
    for r in rows_sorted[:25]:
        nm = (r.get(c_name) or "?") if c_name else "?"
        du = fnum(r.get(c_dur)) if c_dur else 0.0
        lines.append("- `%s`  %.2f ms  (bucket=%s)" % (nm, du / 1000.0, bucket_of(nm, "")))
    with open(os.path.join(dst, "va_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------- 6. 体积检查与裁剪
    def size_kb() -> float:
        return sum(os.path.getsize(os.path.join(dst, x)) for x in os.listdir(dst)) / 1024.0

    while size_kb() > args.max_kb and len(top_rows) > 40:
        top_rows = top_rows[: int(len(top_rows) * 0.6)]
        with open(os.path.join(dst, "kernel_top.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keep_cols)
            for r in top_rows:
                w.writerow([r.get(c, "") for c in keep_cols])

    print("\n[ok] bundle: %s  (%.0f KB)" % (dst, size_kb()))
    for x in sorted(os.listdir(dst)):
        print("     %-24s %6.1f KB" % (x, os.path.getsize(os.path.join(dst, x)) / 1024.0))
    if size_kb() > args.max_kb:
        print("[warn] 仍超过 %d KB：删掉 op_statistic_top.csv 再试" % args.max_kb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
