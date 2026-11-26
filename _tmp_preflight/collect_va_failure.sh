#!/usr/bin/env bash
# 收集"vllm-ascend 起服务失败"现场，产物放 _tmp_preflight/out/<tag>/，可直接 commit 回传。
# 用法： bash _tmp_preflight/collect_va_failure.sh [tag]
# 注意：容器若已被删除，容器内的 ascend 日志就没了 —— 尽量在容器还活着时跑。
set -u
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SELF/.." && pwd)
TAG=${1:-fail_$(hostname -s)_$(date +%Y%m%d_%H%M%S)}
OUT="$SELF/out/$TAG"
mkdir -p "$OUT"
CNAME=${CNAME:-dsv41-va}
echo "==> out: $OUT"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------- 1. vllm 侧日志
LOGDIR=${LOGDIR:-/home/a00821909/logs}
{
  ls -lt "$LOGDIR" 2>/dev/null | head -10
  echo
  for f in $(ls -t "$LOGDIR"/serve_b*.log 2>/dev/null | head -2); do
    echo "===== $f  (size=$(du -h "$f" | cut -f1))"
    echo "--- head 5"; head -5 "$f"
    echo "--- tail 200"; tail -200 "$f"
  done
} > "$OUT/10_vllm_log_tail.txt" 2>&1

{
  L1=$(ls -t "$LOGDIR"/serve_b*.log 2>/dev/null | head -1)
  echo "### 所有 fault kernel_name（去重）"
  grep -ah "fault kernel_name" "$LOGDIR"/serve_b*.log 2>/dev/null \
    | grep -o "fault kernel_name=[^,]*" | sort | uniq -c | sort -rn
  echo
  echo "### 错误码族"
  grep -ah -o "EE[0-9]\{4\}\|EL[0-9]\{4\}\|EZ[0-9]\{4\}\|ERR[0-9]\{5\}\|ErrCode=[0-9]*" "$LOGDIR"/serve_b*.log 2>/dev/null \
    | sort | uniq -c | sort -rn | head -20
  echo
  echo "### 最后一个 serve 日志的最后 150 行"
  [ -n "${L1:-}" ] && tail -150 "$L1"
} > "$OUT/11_fault_summary.txt" 2>&1

# ---------- 2. 探针轨迹（不截断，去重）
L1=$(ls -t "$LOGDIR"/serve_b*.log 2>/dev/null | head -1)
if [ -n "${L1:-}" ]; then
  {
    echo "### VA_PROBE 总条数 = $(grep -c '\[VA_PROBE\]' "$L1")"
    echo "### tokens= 分布（看崩在 prefill 还是 decode）"
    grep -o "tokens=[0-9]*" "$L1" | sort | uniq -c | sort -rn | head -20
    echo
    echo "### 参数组合去重（每层一行）"
    grep -o "\[VA_PROBE\].*" "$L1" | sed 's/tokens=[0-9]*/tokens=N/' | sort | uniq -c | sort -rn | head -20
    echo
    echo "### 最后 20 条"
    grep "\[VA_PROBE\]" "$L1" | tail -20
  } > "$OUT/12_va_probe.txt" 2>&1
fi

# ---------- 3. 容器内 ascend 日志（容器还在才拿得到）
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CNAME"; then
  say "容器 $CNAME 还活着 → 收集 ascend device/plog"
  docker cp "$SELF/collect_fail_in_container.sh" "$CNAME:/tmp/cf.sh" >/dev/null 2>&1
  docker exec -e OUT=/tmp/cf_out "$CNAME" bash -lc 'rm -rf /tmp/cf_out; mkdir -p /tmp/cf_out; bash /tmp/cf.sh' \
    || echo "!! 容器内收集失败"
  docker cp "$CNAME:/tmp/cf_out/." "$OUT/" >/dev/null 2>&1 || echo "!! docker cp 失败"
  {
    echo "### docker inspect（含 exit code / OOM / 启动参数）"
    docker inspect "$CNAME" 2>/dev/null | head -80
    echo
    echo "### docker ps -a"
    docker ps -a 2>/dev/null | head -10
  } > "$OUT/65_container_state.txt" 2>&1
else
  say "⚠️ 容器 $CNAME 已经不在了 —— 容器内的 ascend 日志拿不到"
  {
    echo "容器 $CNAME 不存在；只能看 docker ps -a 与宿主侧的 dmesg。"
    docker ps -a 2>/dev/null | head -10
  } > "$OUT/65_container_state.txt" 2>&1
fi

# ---------- 4. 宿主侧
{
  echo "### npu-smi"; npu-smi info 2>&1 | head -40
  echo "### dmesg（最近）"; dmesg -T 2>/dev/null | tail -60
  echo "### 残留进程"; ps -ef | grep -E "vllm|EngineCore|Worker_DP" | grep -v grep | head -20
  echo "### 磁盘"; df -h /home /dev/shm 2>/dev/null
} > "$OUT/70_host_state.txt" 2>&1

# ---------- 5. 摘要
{
  echo "# failure summary  $(date -Is)"
  echo "tag: $TAG"
  echo "host: $(hostname)"
  echo
  echo "--- 崩掉的 kernel（去重）"; cat "$OUT/11_fault_summary.txt" 2>/dev/null | head -12
  echo
  echo "--- 探针"; head -12 "$OUT/12_va_probe.txt" 2>/dev/null
  echo
  echo "--- 文件清单"; ls -lh "$OUT" | awk '{print $9, $5}'
} > "$OUT/99_summary.txt" 2>&1

echo
cat "$OUT/99_summary.txt"
echo
echo "==> 回传："
echo "    cd $REPO && git add _tmp_preflight/out/$TAG && git commit -m 'data(preflight): va failure $TAG' && git push"
