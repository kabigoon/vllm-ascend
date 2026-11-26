#!/usr/bin/env bash
# 在容器内跑：收集 ascend device/plog 日志里跟故障相关的部分，以及 worker 的真实 env。
# 只用 docker exec 调它；输出到 /out（宿主目录的 bind mount）。
set -u
OUT=${OUT:-/tmp/cf_out}
L=/root/ascend/log

{
  echo "### ascend log 目录"
  find "$L" -maxdepth 2 -type d 2>/dev/null | head -20
  echo
  echo "### 最近 2 小时内的 device 日志（按时间）"
  find "$L" -name "device-*.log" -newermt "-2 hours" 2>/dev/null | head -20
  echo
  echo "### 最近 2 小时内的 plog（按时间）"
  find "$L" -name "plog-*.log" -newermt "-2 hours" 2>/dev/null | head -20
} > "$OUT/60_ascend_log_tree.txt" 2>&1

# --- device 日志：只取错误上下文（每份最多 200 行），避免体积爆炸
: > "$OUT/61_device_errors.txt"
for f in $(find "$L" -name "device-*.log" -newermt "-2 hours" 2>/dev/null | head -6); do
  echo "===== $f" >> "$OUT/61_device_errors.txt"
  grep -a -iE "error|exception|aicore|vector core|kernel_name|op_name|507035|trap|pc" "$f" 2>/dev/null | tail -200 >> "$OUT/61_device_errors.txt"
done

# --- plog：HCCL / RUNTIME 的故障链
{
  for f in $(find "$L" -name "plog-*.log" -newermt "-2 hours" 2>/dev/null | head -8); do
    echo "===== $f"
    grep -a -iE "TaskExecStage|fault kernel_name|\[AIV\]|tag\[|AllGather|AllReduce|ReduceScatter|vector core exception|ErrCode=5|EE9999|EL[0-9]{4}" "$f" 2>/dev/null | tail -120
  done
} > "$OUT/62_plog_hccl_runtime.txt" 2>&1

# --- 故障算子的去重统计（最重要的一行：到底几个不同的 kernel 崩了）
{
  echo "### fault kernel_name 去重"
  grep -ah "fault kernel_name" "$L"/debug/plog/*.log "$L"/run/*/device-*.log 2>/dev/null \
    | grep -o "fault kernel_name=[^,]*" | sort | uniq -c | sort -rn | head -20
  echo
  echo "### 出现过的 aiv_* / hccl_* kernel"
  grep -ah -o "aiv_[a-zA-Z0-9_]*\|hccl_[a-zA-Z0-9_]*" "$L"/debug/plog/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -20
  echo
  echo "### HCCL tag / 算法"
  grep -ah -o "tag\[[^]]*\]\|AlgType([^)]*)\|rankSize:\[[0-9]*\]" "$L"/debug/plog/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -20
} > "$OUT/63_fault_kernels.txt" 2>&1

# --- 关键：worker 进程启动时到底有没有 HCCL_OP_EXPANSION_MODE
{
  echo "### 每个进程 environ 里的 HCCL_* / PROFILER / VA_PROBE"
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    if tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -q "vllm\|VLLM"; then
      echo "--- pid=$pid $(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-120)"
      tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -E "^(HCCL_|VLLM_|ASCEND_)" | sort
    fi
  done
} > "$OUT/64_worker_env.txt" 2>&1

echo "container-side done"
