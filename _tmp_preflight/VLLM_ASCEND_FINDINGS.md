# vLLM-Ascend on A3 / HDK 25.5.x —— 2026-11-26 实机排障记录

> 目标：在 A3×16 上跑 **DeepSeek-V4.1-Flash(w4a8)**，拿 128K / 50% 前缀 / C=1 的 TTFT，与 sglang 基线（73.7 s）对比。
> 结果：**未跑通**（主动止损）。但把"为什么跑不通"钉到了 5 个具体点，其中 4 个已定位、3 个已有 workaround。
> 本文件不是 upstream 的一部分，随 `dsv41-perf` 分支保留。

## 1. 一句话结论

**镜像验证组合是 HDK 26.2.RC1，而两台可用机器是 25.5.1 / 25.5.0。** 差异不在初始化，而在两条**新执行路径**上：
HCCL 的 AIV 展开内核、以及 triton-ascend 生成的 Engram kernel —— 一碰就 `vector core exception`。
把这两条绕掉之后，又暴露"自定义算子没注册"（`aclnnHcPre` 不在 `libopapi.so`），此时模型初始化已完成、只差第一个 dummy forward。

## 2. 环境事实（实测，非文档）

| 项 | 值 |
|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0-a3`，id `77aea9096250`，created 2026-10-29，23.9 GB |
| 镜像内 CANN / torch | CANN **9.2.1**，torch **2.10.0+cpu**，torch_npu **2.10.0.post7**，python 3.12.16 |
| 代码安装方式 | **editable**，源码树在 `/vllm-workspace/vllm-ascend`（768 个 .py，与 tag v0.30.0 的 714 个差 54 个，多出来的全在 `_cann_ops_custom/.../tbe/` 下） |
| 发布矩阵 | vllm-ascend v0.30.0 ↔ vLLM v0.30.0 ↔ CANN 9.2.1 ↔ **HDK 26.2.RC1**（A2/A3） |
| 实机宿主 | `-71`：HDK **25.5.1**；另一台：HDK **25.5.0** + CANN 8.5.0（宿主 CANN 不影响容器） |
| 冒烟结果 | 单卡 bf16 matmul ✓；2 卡 HCCL `all_reduce` ✓；torch_npu 报 `32 padding size` 兼容告警（说明版本不匹配但已走兼容路径） |
| `--help=all` | `--tokenizer-mode` **没有** `deepseek_v41`（用 `deepseek_v4`）；`--swa-bounded-replay` **不存在**（那是 v0.31+ 的 flag） |

## 3. 五个拦路虎

| # | 症状（原始报错） | 根因 | 状态 |
|---|---|---|---|
| 1 | `fault kernel_name=aiv_all_gather_int32_t`；`ErrCode=507035 vector core exception`；`tag[AllGather_group_name_123]`，`count[384]`（= 384 个专家的计数），卡 43 s 后炸 | 官方 A3 示例把 `HCCL_OP_EXPANSION_MODE` 设成 **AIV**（把小集合通信展开到 vector core）。该 kernel 在 HDK 25.5.1 上不可用 | ✅ `unset HCCL_OP_EXPANSION_MODE` 后 `aiv_*` 归零 |
| 2 | `fault kernel_name=_gather_lookback_kernel`，同类 vector core exception | **Engram 的 Triton kernel**（`vllm_ascend/ops/triton/engram_lookback.py`，被 `models/deepseek_v41/engram/model_state.py:117` 调用）。triton-ascend 3.6.0.post2 生成的代码与 25.5.1 不兼容 | 🟡 已写 torch 等价实现（`VLLM_ASCEND_ENGRAM_LOOKBACK_TORCH=1`），未在 16 卡上验证 |
| 3 | `ImportError: libascend_hal.so` / `No module named torch_npu` / 容器内 `devices: 0` | 宿主**没有 Ascend Docker Runtime**（默认 runtime 不是 `ascend`），容器看不到设备 | ✅ 手动映射：`--device=/dev/davinci0..15` + `davinci_manager/devmm_svm/hisi_hdc` + 挂 `/usr/local/Ascend/driver`、`/usr/local/dcmi`、`/usr/local/bin/npu-smi`、`/etc/ascend_install.info`；不要用 `ASCEND_RT_VISIBLE_DEVICES=0-15`（区间写法不认，会只看到 1 卡） |
| 4 | 容器内 `import numpy` 失败（`numpy/ma/core.py` 缺失）；`pip` 也坏（`pip._vendor.idna.core` 缺失）；`find site-packages -name core.py \| wc -l` = **0** | **`-71` 那台的镜像副本被批量删除了所有 `core.py`**（疑似他人的清理脚本按 `core.py` 通配删除）。同一镜像在另一台机器上 `core.py=36`、`import numpy/torch/torch_npu` 全 OK ⇒ **镜像本身没坏，是 -71 的磁盘副本坏了** | ✅ 换机器 |
| 5 | `RuntimeError: Worker failed with error 'aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so, or libopapi.so not found.'`，发生在 `_initialize_kv_caches → determine_available_memory` 的 dummy forward | **自定义算子包没被 ACLNN 看到**：镜像把 custom ops 放在 `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so`，需要有 `ASCEND_CUSTOM_OPP_PATH`（或 `ASCEND_OPP_PATH` 的 vendors 段）指过去 | ❌ **未解决**（明天第一条） |

### 关键进展：第 5 条发生在哪一步

```
EngineCore.__init__ → _initialize_kv_caches → determine_available_memory → worker dummy forward
```
说明：**权重加载完成、模型初始化完成、KV cache 正在估算** —— 只差第一个真正的前向。这不是"模型跑不起来"，是"算子注册表没挂上"。

## 4. 明天怎么继续（按性价比）

1. **首选：换一台 HDK ≥ 26.2.RC1 的机器**（或等配 25.5 的镜像）。第 1、2 条在那台机器上大概率不存在，第 5 条也可能自动消失。
2. 若必须在 HDK 25.5.x 上跑，按顺序：
   a. **解第 5 条**：进容器 `env | grep -i -E "opp|custom"`；把 `ASCEND_CUSTOM_OPP_PATH` 指到
      `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors`（或 `_cann_ops_custom`），
      确认 `libcust_opapi.so` 与 `hc_pre/hc_post` 的 kernel json 可被找到，再起服务。
      旁证：`_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/hc_post/` 下有一堆
      `HcPost_*.json` —— 说明 **HcPre/HcPost 在 A3 上是编译好的**（这对 sglang 侧也是重要情报，见 §5）。
   b. 第 1、2 条 workaround 保留（`serve_b.sh` 内置，且带 fail-fast 自检：AIV 被设上就直接退出）。
   c. 兜底：**vllm-ascend v0.30.0rc2 + CANN 9.2.0** 的 venv（矩阵里 rc2 配 CANN 9.2.0 / torch_npu 2.10.0.post7，离 HDK 25.x 更近）。
3. **不要再做**：在这两台机器上反复调 `HCCL_BUFFSIZE`、numa、图参数 —— 失败点不在这里。

## 5. 对 sglang 侧有价值的三条情报

1. **镜像自带 `hc_pre` / `hc_post` 的 A3（ascend910_93）kernel**（`40_custom_ops.txt`、`kernel/ascend910_93/hc_post/*.json`）。
   而 sglang 在 A3 上把 mHC 退化成纯 torch（`MHC_T:14-49`），我们实测 mHC 约 **350–400 ms/forward**（其中 fp32 bmm ≈176 ms）。
   ⇒ **"sglang 能不能也调这两个算子"是一条不需要换框架的优化线**（和 INT8 索引器同属"门禁过严"类）。
2. **vllm-ascend 的索引器在 A3 上走 INT8（quant mode 2）**，三种 candidate 模式复用同一个 native op；
   而 sglang A3 走 BF16 的 `LightningIndexerV2` + `SparseLightningIndexer`（我们的 984 ms/次 就是这个）。
   其启动日志明确打了 `Patched AttentionConfig.indexer_kv_dtype to accept 'int8'`。
3. **`[VA_PROBE]` 实测**（我们在 `_select_sparse_indices` 加的探针，480 行）证实在 vllm-ascend 里
   `candidate_source=True`（layer 20）与 `uses_cfilter=True`（消费者层）**确实都在跑**，且 `packed=False`
   —— 即 A3 上没用那个 fused `*_manage` 打包算子。也就是说**两级候选结构两边都有**，差别只在精度与实现。

## 6. 本目录脚本（已推到 `dsv41-perf`）

| 脚本 | 用途 |
|---|---|
| `collect_va_env.sh` / `collect_fail_in_container.sh` | 环境/版本/树差异/权重/flags/custom-ops 清单；失败现场（vllm 日志 + ascend device/plog + worker env） |
| `collect_va_failure.sh` | 一键收现场到 `out/<tag>/`，可 commit 回传 |
| `complete_image_tree.sh` | 把镜像里 fork 缺的 54 个 .py（TBE 算子源码）补进 fork，并做全量内容 diff |
| `serve_b.sh` / `run_b.sh` | 起服务的（b）口径：4096 chunk / C=1 / 无投机 / enforce-eager / w4a8；内置两个 workaround + fail-fast |
| `bench_and_profile.sh` / `bench_client.sh` | 用 sglang 同一套 client 打 `/v1/completions`：run1 灌前缀 → run2 测 TTFT → run3 带 profiler |

## 7. 代码补丁清单（本分支相对 v0.30.0）

| 文件 | 改动 |
|---|---|
| `vllm_ascend/attention/dsa_v41.py` | `_select_sparse_indices` 入口加 `VLLM_VA_PROBE=1` 时的一行打印（index_source / candidate_source / uses_cfilter / packed / cand / tokens） |
| `vllm_ascend/models/deepseek_v41/engram/model_state.py` | `_gather_lookback_kernel` 的 torch 等价实现，`VLLM_ASCEND_ENGRAM_LOOKBACK_TORCH=1` 启用 |
