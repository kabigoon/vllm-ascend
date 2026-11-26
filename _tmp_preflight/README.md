# `_tmp_preflight/`（分支 `dsv41-perf`，基于 tag `v0.30.0`）

用来**对照 vllm-ascend 和 sglang** 在 DSV4.1-Flash / A3 上的 prefill 表现。**不是 upstream 的一部分**，别往 main 合。

## 目录

| 文件 | 作用 |
|---|---|
| `collect_va_env.sh` | **宿主侧**收集脚本：环境/版本/树差异/权重/flags/自定义算子清单。只读，不占 NPU。 |
| `collect_in_container.sh` | 上面那个脚本在容器内执行的部分（不要单独调用）。 |
| `run_b.sh` + `serve_b.sh` | 宿主侧起 (b)：对齐 sglang 口径（4096 chunk / C=1 / 无投机 / no graph / w4a8）。 |
| `bench_client.sh` | 用 sglang 仓同一套 client 打 vLLM 的 `/v1/completions`，跑两遍取第二遍。 |
| `out/<tag>/` | `collect_va_env.sh` 的产物（纯文本，几 MB 内，**要 commit** 回传）。 |

## 用法

```bash
# 1) 收集环境（不需要 NPU，随时能跑）
bash _tmp_preflight/collect_va_env.sh                # tag 默认 <host>_<时间戳>

# 2) 回传
git add _tmp_preflight/out/<tag>
git commit -m "data(preflight): va env <tag>"
git push

# 3) 起服务（16:00 之后）
bash _tmp_preflight/run_b.sh                          # 裸镜像
PROBE=1 bash _tmp_preflight/run_b.sh                  # 额外单文件挂载 dsa_v41.py 探针

# 4) 打点（另一个终端）
bash _tmp_preflight/bench_client.sh 8900 /home/a00821909/bench
```

## 产物怎么看（给分析用）

- `00_host.txt`：`npu-smi` / driver `version.info` / 大页 / 磁盘 / docker runtime / 镜像 digest。
- `10_image_versions.txt`：`ROOT=`（editable 源码树根，挂载用这个）、CANN 目录、torch/torch_npu、py/so 计数。
- `11_flags_present.txt` / `12_flags_values.txt`：`vllm serve --help=all` 里关心的 flag 是否存在、取值。
- `20~24_*`：**镜像树 vs fork 树**的差异 —— `21_py_only_in_image.txt` 是关键（镜像不是 v0.30.0 tag 构建的，多出来的模块挂载会被遮掉）。
- `30/31_dsa_v41*` + `dsa_v41_image.py`：我们唯一要 patch 的文件，md5/diff + 镜像原文件（用来在"镜像版本"上重打补丁）。
- `40_custom_ops.txt`：镜像自带的 CANN 自定义算子清单（看有没有 `QuantSparseLightningIndexer` / `fused_*_manage`）。
- `50_weights.txt`：`/home/weights` 下有哪些 checkpoint、大小、quant_method。
- `99_summary.txt`：一页速览。

## 已知结论（截至 2026-11-26）

- 镜像 `v0.30.0-a3`：CANN 9.2.1 / torch 2.10.0+cpu / torch_npu 2.10.0.post7；editable 树根 `/vllm-workspace/vllm-ascend`。
- 宿主 HDK 25.5.1 能跑（torch_npu 走 32B padding 兼容路径），单卡 + 2 卡 HCCL 已冒烟通过。
- `--tokenizer-mode` 没有 `deepseek_v41`，用 **`deepseek_v4`**；`--swa-bounded-replay` 在 v0.30.0 不存在。
- 镜像树的 `.py` 数（768）≠ tag v0.30.0（714）⇒ **禁止整树挂载**，只允许单文件挂载。
- -71 上只有 **w4a8** 权重（w8a8 不在），而 sglang 基线是 w8a8 ⇒ 量化不对等，报告里要标。

## 补全镜像树（可选，之后可整树挂载）

```bash
bash _tmp_preflight/complete_image_tree.sh      # 补 54 个 .py + 全量内容 diff
PROBE=1 FULLTREE=1 bash _tmp_preflight/run_b.sh # diff 干净时用这个
```
