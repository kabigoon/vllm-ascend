#!/usr/bin/env bash
# 宿主侧：用 sglang 仓的同一套 client 打 vllm 的 /v1/completions，跑两遍取第二遍。
#   bash _tmp_preflight/bench_client.sh [port] [outdir]
set -u
PORT=${1:-8900}; OUT=${2:-/home/a00821909/bench}
MODEL=${MODEL:-/home/weights/DeepSeek-V4.1-Flash-1017-w4a8}
IMG=${CLIENT_IMG:-quay.io/ascend/sglang:cann9.2.0-a3-20261109post3}
mkdir -p "$OUT"
docker run --rm --net=host -v /home/weights:/home/weights -v "$OUT:/out" "$IMG" bash -lc "
for i in 1 2; do
  python3 -m sglang.benchmark.serving --backend vllm \
    --host 127.0.0.1 --port $PORT \
    --model $MODEL --served-model-name dsv41 \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 1 --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 63623 --gsp-question-len 63623 --gsp-output-len 8 \
    --num-prompts 1 --max-concurrency 1 --warmup-requests 0 \
    --output-file /out/run\$i.jsonl || exit 1
done
echo '--- run2 TTFT ---'; grep -o '\"ttft[^,]*' /out/run2.jsonl | head
"
