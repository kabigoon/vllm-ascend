#!/usr/bin/env bash
# 宿主侧：起 (b) 的 vllm-ascend 容器 + 服务。
#   PROBE=1 bash _tmp_preflight/run_b.sh    # 额外单文件挂载 dsa_v41.py（探针）
set -u
IMG=${IMG:-quay.io/ascend/vllm-ascend:v0.30.0-a3}
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SELF/.." && pwd)
EXTRA=()
if [ "${PROBE:-0}" = "1" ] && [ -f "$REPO/vllm_ascend/attention/dsa_v41.py" ]; then
  EXTRA+=(-v "$REPO/vllm_ascend/attention/dsa_v41.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py")
  EXTRA+=(-e VLLM_VA_PROBE=1)
  echo "!! 已挂载单文件 dsa_v41.py（VLLM_VA_PROBE=1）—— 先确认它和镜像 md5 一致"
fi
docker rm -f dsv41-va >/dev/null 2>&1 || true
exec docker run --rm -it --name dsv41-va \
  --net=host --ipc=host -e ASCEND_VISIBLE_DEVICES=0-15 "${EXTRA[@]}" \
  -v /home/weights:/home/weights \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v "$SELF/serve_b.sh:/serve_b.sh:ro" \
  "$IMG" bash /serve_b.sh
