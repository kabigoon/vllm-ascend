#!/usr/bin/env bash
# 收集 vllm-ascend 在这台机器上的"环境/版本/树差异/权重/flags"证据，供离线分析。
# 只读：不改镜像、不改 fork、不占 NPU（可以在没有 NPU 的时候跑）。
# 用法：  bash _tmp_preflight/collect_va_env.sh [tag]
# 产物：  _tmp_preflight/out/<tag>/...      （文本，几 MB 以内，可 commit）
set -u
IMG=${IMG:-quay.io/ascend/vllm-ascend:v0.30.0-a3}
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SELF/.." && pwd)
TAG=${1:-$(hostname -s)_$(date +%Y%m%d_%H%M%S)}
OUT="$SELF/out/$TAG"
mkdir -p "$OUT"
echo "==> image : $IMG"
echo "==> fork  : $REPO/vllm_ascend"
echo "==> out   : $OUT"

# ---------------- 00 宿主
{
  echo "### uname"; uname -a
  echo "### os"; (cat /etc/os-release 2>/dev/null || true) | head -4
  echo "### npu-smi"; npu-smi info 2>&1 | head -20
  echo "### driver"; cat /usr/local/Ascend/driver/version.info 2>&1
  echo "### hugepages"; grep -i huge /proc/meminfo 2>/dev/null | head -3
  echo "### disk"; df -h /home /dev/shm /var/lib/docker 2>&1
  echo "### docker runtime"; docker info --format 'default_runtime={{.DefaultRuntime}}' 2>&1
  echo "### related images"; docker images 2>/dev/null | grep -iE 'vllm-ascend|sglang' | head
  echo "### image digest/created"
  docker inspect -f '{{index .RepoDigests 0}}  created={{.Created}}  size={{.Size}}' "$IMG" 2>&1
} > "$OUT/00_host.txt" 2>&1

# ---------------- 10-40 容器内
docker run --rm \
  -v "$REPO:/fork:ro" \
  -v "$OUT:/out" \
  -v "$SELF/collect_in_container.sh:/collect.sh:ro" \
  "$IMG" bash /collect.sh

# ---------------- 50 权重
{
  echo "### /home/weights"; ls -d /home/weights/* 2>&1 | head -30
  for d in /home/weights/*/; do
    [ -d "$d" ] || continue
    echo "=== $d"; du -sh "$d" 2>/dev/null
    ls "$d" 2>/dev/null | head -8
    f="$d/config.json"
    if [ -f "$f" ]; then
      echo "--- quant/arch from $f"
      grep -o '"quant_method"[^,}]*' "$f" | head -2
      grep -o '"architectures"[^]]*]' "$f" | head -1
    fi
  done
} > "$OUT/50_weights.txt" 2>&1

# ---------------- 99 摘要
{
  echo "# summary  $(date -Is)"
  echo "image: $IMG"
  echo "host : $(hostname)"
  echo "npu-smi: $(npu-smi info 2>/dev/null | sed -n '2p' | tr -s ' ')"
  echo "driver : $(grep -E '^(Version|Innerversion)=' /usr/local/Ascend/driver/version.info 2>/dev/null | tr '\n' ' ')"
  echo
  grep -hE '^(ROOT|img_py|img_non_py|only_in_image|non_py_only_in_image)' \
       "$OUT"/10_image_versions.txt "$OUT"/20_tree_py_counts.txt "$OUT"/23_tree_artifact_counts.txt 2>/dev/null
  echo
  echo "dsa_v41:"; cat "$OUT/30_dsa_v41_md5.txt" 2>/dev/null
  echo
  echo "weights:"; grep -E '^/home/weights|^=== ' "$OUT/50_weights.txt" 2>/dev/null | head -20
  echo
  echo "out size: $(du -sh "$OUT" | cut -f1)   files: $(ls "$OUT" | wc -l)"
} > "$OUT/99_summary.txt" 2>&1

echo
cat "$OUT/99_summary.txt"
echo
echo "==> 回传："
echo "    cd $REPO && git add _tmp_preflight/out/$TAG && git commit -m 'data(preflight): va env $TAG' && git push"
