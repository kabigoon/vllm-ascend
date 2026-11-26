#!/usr/bin/env bash
# 宿主侧一条命令：等 server 就绪 → run1 灌前缀 → run2 测 TTFT → (可选) run3 带 profiler。
# 用法： bash _tmp_preflight/bench_and_profile.sh [port] [outdir]
#        NO_PROFILE=1 bash ...        # 跳过 run3
set -u
PORT=${1:-8900}
OUT=${2:-/home/a00821909/bench}
MODEL=${MODEL:-/home/weights/DeepSeek-V4.1-Flash-1017-w4a8}
CLIENT_IMG=${CLIENT_IMG:-quay.io/ascend/sglang:cann9.2.0-a3-20261109post3}
PROF=${PROF_DIR:-/home/a00821909/va_prof}
mkdir -p "$OUT" "$PROF"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "等待 server 就绪 (http://127.0.0.1:$PORT/health)"
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ready after ~$((i*5))s"; break
  fi
  [ "$i" = 240 ] && { echo "!! 20 分钟还没就绪，放弃"; exit 1; }
  sleep 5
done

run_bench() {  # $1 = 序号
  docker run --rm --net=host \
    -v /home/weights:/home/weights -v "$OUT:/out" \
    "$CLIENT_IMG" python3 -m sglang.benchmark.serving \
      --backend vllm --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL" --served-model-name dsv41 \
      --dataset-name generated-shared-prefix \
      --gsp-num-groups 1 --gsp-prompts-per-group 1 \
      --gsp-system-prompt-len 63623 --gsp-question-len 63623 --gsp-output-len 8 \
      --num-prompts 1 --max-concurrency 1 --warmup-requests 0 \
      --output-file "/out/run$1.jsonl"
}

say "run1：灌 50% 前缀（结果丢弃）"
run_bench 1 | tee "$OUT/run1.log"

say "run2：**测 TTFT**（无 profiler）"
run_bench 2 | tee "$OUT/run2.log"

if [ "${NO_PROFILE:-0}" != "1" ]; then
  say "start_profile"
  curl -s -X POST "http://127.0.0.1:$PORT/start_profile" || echo "!! start_profile 失败（看 server log）"
  say "run3：带 profiler（这次 TTFT 会偏大，只用来取算子数据）"
  run_bench 3 | tee "$OUT/run3.log"
  say "stop_profile"
  curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" || true
  sleep 10
  say "profiler 产物"
  find "$PROF" -maxdepth 3 -name "kernel_details.csv" 2>/dev/null | head
  if [ -x "$(command -v python3)" ] && [ -f /home/a00821909/sglang/_tmp_preflight/collect_profile_bundle.py ]; then
    say "抽小包（collect_profile_bundle.py）"
    python3 /home/a00821909/sglang/_tmp_preflight/collect_profile_bundle.py "$PROF" --tag va_demo \
      || echo "!! bundle 抽取失败（可能目录层级不同，先看 find 的输出）"
  fi
fi

say "TTFT 汇总"
for f in "$OUT"/run2.jsonl "$OUT"/run3.jsonl; do
  [ -f "$f" ] || continue
  printf '%-28s ' "$(basename "$f")"
  grep -o '"ttft": *[0-9.]*' "$f" | head -1
  grep -o '"ttft": *[0-9.]*' "$f" | awk -F: '{v=$2+0; if(v>m)m=v} END{printf "  (max %s)\n", m}'
done

say "回传命令"
cat <<EOF
cd /home/a00821909/sglang && git add _tmp_preflight/profile_bundle/va_demo* 2>/dev/null
git commit -m "data(preflight): vllm-ascend demo bundle" && git push
# server 端日志：/home/a00821909/logs/serve_b.log
# bench 输出  ：$OUT/run{1,2,3}.jsonl / run{1,2,3}.log
EOF
