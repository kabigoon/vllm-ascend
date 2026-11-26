#!/usr/bin/env bash
# (b) 对齐我们 sglang 口径的最小配置：4096 chunk / C=1 / 无投机 / no graph。
# 在容器内执行；由 run_b.sh 挂进来。
set -u
MODEL=${MODEL:-/home/weights/DeepSeek-V4.1-Flash-1017-w4a8}
PORT=${PORT:-8900}
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --served-model-name dsv41 \
  --max-model-len 150000 --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --data-parallel-size 4 --tensor-parallel-size 4 --enable-expert-parallel \
  --quantization ascend --block-size 128 \
  --safetensors-load-strategy lazy --tokenizer-mode deepseek_v4 \
  --enable-prefix-caching --enforce-eager \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}'
