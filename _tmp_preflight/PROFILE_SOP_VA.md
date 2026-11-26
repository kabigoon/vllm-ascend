# vLLM-ascend prefill profiling —— 三步 SOP（与 sglang 同一套方法论）

> 目标：回答 **"vLLM 到底快在哪里"** —— 拿到 prefill 阶段每个 kernel 的耗时/占比/pipe 比率，
> 和我们 sglang 侧的 `KERNEL_CODE_MAP.md`（索引器 984 ms/次、4 次约 4 s/chunk）逐条对照。

## 0. 与 sglang 的差异（先记住三点）

| | sglang | vLLM-ascend |
|---|---|---|
| 触发 | `--profile-with-stack` 等启动开关 + `PyTorch` profiler | **`VLLM_TORCH_PROFILER_DIR` 必须在启动时给**（引擎初始化时创建 profiler 对象），之后用 `POST /start_profile`、`/stop_profile` 触发 |
| 产物格式 | `*_ascend_pt/ASCEND_PROFILER_OUTPUT/` | **完全一样**（`torch_npu.profiler` + `export_type=Text`） |
| Python 栈 | 有（`with_stack=True`） | **没有**（wrapper 写死 `with_stack=False`、`data_simplification=True`）。kernel 级分析不受影响，但拿不到"这个 kernel 从哪行代码来" |

## 1. 起服务（容器内，profiler 目录必须给）

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export VLLM_LOG_STATS_INTERVAL=2
export VLLM_TORCH_PROFILER_DIR=/home/a00821909/va_prof     # ← 关键，运行中改不了
mkdir -p "$VLLM_TORCH_PROFILER_DIR" /home/a00821909/logs

vllm serve /home/weights/DeepSeek-V4.1-Flash-1017-w4a8 \
  --host 0.0.0.0 --port 8900 --served-model-name dsv41 \
  --max-model-len 150000 --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 --tensor-parallel-size 4 --enable-expert-parallel \
  --quantization ascend --block-size 128 \
  --safetensors-load-strategy lazy --tokenizer-mode deepseek_v4 \
  --enable-prefix-caching --reasoning-parser deepseek_r1 \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  2>&1 | tee /home/a00821909/logs/serve.log
```

---

## Step 1 —— 小样本验证方法可用（5 分钟，先别上 128K）

```bash
# 容器内，另一个 shell
curl -s -X POST http://127.0.0.1:8900/start_profile

# 一个很小的请求：4K 输入、2 token 输出
curl -s http://127.0.0.1:8900/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","prompt":"hello world, this is a profiling smoke test","max_tokens":2,"temperature":0}' | head -c 200

curl -s -X POST http://127.0.0.1:8900/stop_profile
sleep 20

# 产物在哪
find /home/a00821909/va_prof -name kernel_details.csv | head
du -sh /home/a00821909/va_prof

# 抽小包
python3 /home/a00821909/vllm-ascend/_tmp_preflight/collect_va_profile.py \
  /home/a00821909/va_prof --tag smoke1
```

**通过判据（三条同时满足）**：
1. 打印 `[ok] bundle: ... (xxx KB)` 且 **< 300 KB**；
2. `va_steps.csv` 里有 1–2 步、`kernel_top.csv` 有几十行；
3. `va_summary.md` 里 `indexer` / `mla_attn` / `moe` 三个桶都非空。

不满足 → 先别进 Step 2，把 `meta.txt` 和 `va_summary.md` 发我（列名/切步可能要调）。

---

## Step 2 —— 完整一次 prefill（128K，50% 前缀命中）

```bash
curl -s -X POST http://127.0.0.1:8900/start_profile

# run1（冷）+ run2（命中），共 2 条 128K
HF_HUB_OFFLINE=1 vllm bench serve \
  --backend openai --base-url http://127.0.0.1:8900 --endpoint /v1/completions \
  --model dsv41 --tokenizer /home/weights/DeepSeek-V4.1-Flash-1017-w4a8 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 63623 --prefix-repetition-suffix-len 63623 \
  --prefix-repetition-num-prefixes 1 --prefix-repetition-output-len 8 \
  --num-prompts 2 --max-concurrency 1

curl -s -X POST http://127.0.0.1:8900/stop_profile
sleep 30

python3 /home/a00821909/vllm-ascend/_tmp_preflight/collect_va_profile.py \
  /home/a00821909/va_prof --tag full128k --top 300
```

⚠️ **profiled 的 TTFT 会明显偏大**（Level1 + PipeUtilization 采集开销）。TTFT 只认**不带 profiler** 的那次测量（31.0 s）；这里的数只用来看 kernel 结构。

---

## Step 3 —— 128K 的「后几轮」prefill（给 MindStudio Insight 看）

目的：避开首轮冷启动的编译/预热噪声，专门录**稳态**的几轮 prefill。

```bash
# ① 先跑 1 轮 warmup，不采
HF_HUB_OFFLINE=1 vllm bench serve ... --num-prompts 1 --max-concurrency 1   # 同上参数

# ② 开始采
curl -s -X POST http://127.0.0.1:8900/start_profile

# ③ 再打 1~2 轮（同参数）
HF_HUB_OFFLINE=1 vllm bench serve ... --num-prompts 1 --max-concurrency 1

# ④ 停
curl -s -X POST http://127.0.0.1:8900/stop_profile
sleep 30
du -sh /home/a00821909/va_prof
```

**给 Insight**：把整个 `<host>_<pid>_<ts>_ascend_pt` 目录拷到 MindStudio Insight（保持目录结构，Insight 认 `ASCEND_PROFILER_OUTPUT/`）。

**注意磁盘**：DP4×TP4 下 16 个 worker 可能各写一份 trace（每份几十~几百 MB）。采完先看 `du -sh`，只留"拿到请求的那个 rank"（`kernel_details.csv` 最大的），其余删掉再拷贝。

---

## 采集完怎么回传（单包 < 300 KB）

```bash
cd /home/a00821909/vllm-ascend
D=_tmp_preflight/profile_bundle/smoke1
du -sh $D
git add $D && git commit -m "data(profile): va $D" && git push origin dsv41-perf
```

> 推送经常 403（代理/体积），**一个 bundle 一个 commit**，超过 300 KB 就先 `rm` 掉 `op_statistic_top.csv` 再推。

## 我会怎么分析（拉下来之后）

1. `va_summary.md`：桶占比 → 一眼看出 vLLM 的时间花在 indexer / mla_attn / moe / comm 哪一块；
2. `va_steps.csv`：每个 4096-chunk 的 span 和内部 top 算子 → 和 sglang 的 **1.27 s/chunk** 逐条对比；
3. `kernel_top.csv`：找 `*indexer*` / `sparse_flash*` 的单次耗时与 `aic_mac_ratio`、`Block Dim` → **判断 vLLM 的索引器是否也走 legacy mask rescan**（sglang 那边我们测到 96% 时间在 mask 重扫、aic_mac_ratio 1.6%）；
4. 结论落成一句话：**vLLM 快，是因为索引器换成了 INT8 + 少扫 mask，还是因为 mHC/通信更便宜。**
