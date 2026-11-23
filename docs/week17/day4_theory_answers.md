## 第 1 题：InputBatch 与调度增量

`_update_states()`把 SchedulerOutput 的“增量”合并进持久化的 InputBatch，处理五类变化：

1. **B 已结束**（`finished_req_ids`）：`self.requests.pop()` 删 `CachedRequestState`，`input_batch.remove_request()` 删 persistent batch 里的槽位（token 行、block table 行、sampling 状态）；随后 `condense()` 压实。
2. **本轮未调度**（被抢占 / 预算用尽）：`cached_req_ids - scheduled_req_ids` 从 batch 移除，但保留 `self.requests`，下轮再 add。
3. **D 新加入**（`scheduled_new_reqs`）：建 `CachedRequestState`（prompt ids、block_ids、`num_computed_tokens`、sampling/pooling params、按 seed 建 generator、LoRA），存 `requests`，再 `add_request()` 填最小空槽。
4. **A decode / C chunked prefill**（`scheduled_cached_reqs`）：更新 `num_computed_tokens`（本轮起点）；追加新 `block_ids`（`resumed_from_preemption` 时整段替换）；PP 非 last rank 才用 `new_token_ids` 补历史；spec token 追加进 `token_ids_cpu` / `num_tokens`。A 和 C 的区别不在 `_update_states`，而在“本轮算几个 token”和“采样结果是否采纳”。
5. **收尾**：`condense()` → `_may_reorder_batch()`（FA3 按 query 长度把 decode/prefill 分段）→ `refresh_metadata()`（有 new/resumed/paused/finished 才把 SamplingMetadata 刷到 GPU）。

注意：A 的输入 token 是上一轮 sample 后由 `_bookkeeping_sync` 写进 `token_ids_cpu` 的；C 这轮的采样结果会被丢弃。

**为什么必须持久化 InputBatch**：HTTP request 只有 prompt + 参数，而一步执行需要的是“算到哪了”的运行状态：

- 完整 token 历史（prompt + 已生成），HTTP 里没有；
- KV 的物理位置（block table / slot）和 `num_computed_tokens`（chunked prefill 断点、prefix cache 命中偏移）；
- sampling 状态（generator/RNG offset、penalty 依赖的 output ids）。

每步重建 = 重新 tokenize + 重新分配 block + 丢掉已写 KV，等于把整段历史重新 prefill，prefix cache 全废；还要每步跨进程反序列化/校验请求，host 开销爆炸。SchedulerOutput 只描述“这一步”，InputBatch 才是可增量演进的“当前世界”。

## 第 2 题：Attention Metadata 与 Slot Mapping

| 项 | 值 |
| --- | --- |
| `query_start_loc` | `[0, 4, 5, 8]`（尾部 padding 填 8） |
| 绝对 positions | A: 20,21,22,23 / B: 7 / C: 15,16,17 |
| `seq_lens`（执行后） | `[24, 8, 18]` |
| `logits_indices` | `[3, 4, 7]` |

- `query_start_loc` = cumsum(`num_scheduled_tokens`) 前补 0（:930-936）。
- positions = `num_computed_tokens` + 请求内 arange（:899-902）。
- `seq_lens` = `num_computed_tokens` + `num_scheduled_tokens`（:938-940）。
- 普通 decode 下 `logits_indices = query_start_loc[1:] - 1`（:967），即拼平 token 维里每个请求的最后一个位置。

`slot_mapping`：`slot = block_table[req][pos // 16] * 16 + pos % 16`（:924）

- 请求 0：20→5×16+4=**84**，21→85，22→86，23→87
- 请求 1：7→12×16+7=**199**
- 请求 2：15→16×16+15=**271**，16→4×16+0=**64**，17→**65**

结果：`[84, 85, 86, 87, 199, 271, 64, 65]`。注意 271→64 是跨块跳变，说明 slot_mapping 是 scatter 映射，不是连续区间。

**为什么两者不能互相替代**：

- `query_start_loc`：token 粒度，描述“本步拼平后的 query 维如何切给各请求”，只含本步新 token；
- `block_table`：block（16 token）粒度，描述“逻辑 KV 位置 → 物理页”，覆盖整段历史（含 prefix cache 命中块）。

缺一不可：`query_start_loc` 给不出物理地址，`block_table` 给不出本步每请求算几个 token，也切不开拼平的 query 维。一个是“本步算多少”，一个是“历史存哪里”。

## 第 3 题：Qwen3 Attention 与 Paged KV

1. TP=1 下 Q：`[64, 16, 128]`（=2048），K：`[64, 8, 128]`（=1024），V 同 K；fused qkv 时输入是 `[64, 4096]`。
2. GQA：16 q heads / 8 kv heads = 每 2 个 q head 共享 1 个 kv head。K/V 本身与 q head 无关，按 q head 存只是复制同样内容 → 显存和带宽 ×2。attention 读一次 kv head 广播给组内 q heads。
3. 每层每个 block：`2 (K,V) × 16 token × 8 head × 128 dim × 2 B = 65,536 B = 64 KiB`（每 token 4 KiB）。
4. `reshape_and_cache_flash`：把本步新算的 K/V 按 `slot_mapping` scatter 写进 paged cache（可选 fp8 量化），纯写显存，不做 attention 数学。`flash_attn_varlen_func`：用 Q + paged K/V、`block_table`、`query_start_loc`(cu_seqlens_q)、`seq_lens`(seqused_k)、causal mask 算 softmax(QKᵀ/√d)V，读整段历史。分工 =“写一次 + 之后每步按页读”。
5. Q/K RMSNorm 是 head_dim 上的 reduce + 可学权重的 scale（per token/head）；RoPE 是按 position 的成对旋转（只作用 q/k，且必须在写 cache 前）；attention 是变长 paged 的 matmul+softmax。三者的并行维度、输入依赖、数据布局都不同；backend 可替换（FA / FlashInfer / Triton），把模型语义塞进 backend kernel 会把“模型”和“硬件”耦合，也妨碍分别 capture CUDA graph。可融合只发生在模型内部（residual+norm、rope 进 attention），不是“同一个 kernel”。

## 第 4 题：Sampler、Logits 与 Greedy

链路：last hidden states（只取 `logits_indices` 那些行）→ lm_head GEMM → logits `[M, vocab]` → 转 fp32 → allowed / bad_words / min_tokens / logit_bias → penalties → temperature → min_p → top_k/top_p → softmax → multinomial（或 argmax）→ sampled `[M,1]` → gather top-k logprobs → SamplerOutput → D2H 后写 `token_ids_cpu`。

- **不必对全部 prefill token 算 logits**：只有每个请求最后一个 token 的 hidden state 才预测下一个 token；中间 token 的 logits 只在 prompt_logprobs 时才要。`logits_indices = query_start_loc[1:]-1` 让 lm_head 只算 M 行而不是 N 行（vocab 151k × hidden 的大 GEMM）。
- **temperature=0 与 top_k=1**：都退化成 argmax。temp < 1e-5 时代码直接走 `greedy_sample`；top_k=1 只留最大 logit，分布退化为概率 1，采样结果仍是 argmax。差别只在实现：一个跳过 RNG/top-k kernel，一个仍跑 softmax+multinomial。
- **raw vs processed logprobs**：raw 是原始 logits 的 log_softmax（模型真实分布，默认返回给用户）；processed 是经 temperature/top-p 等变换后的采样分布（top-p 会把被过滤 token 打成 -inf）。采样必须用 processed，否则 temperature/top-p 无效；上报必须是 raw，否则 logprob 随用户参数变化、被过滤 token 变 -inf，失去“模型概率”的含义。所以两者都算，用 `logprobs_mode` 选上报哪个。
- **Partial prefill 不能写回输出**：此时 `seq_len = num_computed + scheduled < num_tokens`，最后一个 token 不是 prompt 末尾，它的 logits 预测的是 prompt 内部的下一个 token，不是回答。写进 `output_token_ids` 会把没算完的位置当成已生成：污染 output 历史 / penalty / stop 判定 / 流式输出，`num_tokens` 与 `num_computed_tokens` 不一致（下个 chunk 会重算并重写 KV）。代码仍然 sample（保持 batch 形状与图一致），但在 `_bookkeeping_sync` 里丢弃，并把 generator offset 回退 4 字节。

## 第 5 题：Host Overhead 与 GPU Execution

小模型单请求 decode：GPU kernel 每层只有几 µs，而每步要串行做一堆 host 工作，TPOT 被 host 主导。

- **CPU Scheduler**：`schedule()` 纯 Python/numpy，遍历 running/waiting、算 budget、分配 block，在 EngineCore 进程里紧挨着 forward，属于关键路径。
- **InputBatch / Metadata**：`_update_states` + `_prepare_inputs`：numpy cumsum、token 的 index_select、block table commit、slot_mapping、每个 kv_cache_group build metadata、`refresh_metadata`。
- **CPU→GPU**：input_ids、positions、seq_lens、query_start_loc、slot_mapping、block table、sampling metadata 多次小拷贝，每次都有 launch + PCIe 延迟（pin memory + non_blocking 只能部分掩盖）。
- **Kernel launch**：0.6B 一层也有 norm/rope/attention/GEMM/MLP 多个 kernel，每个约 3-10 µs；单 kernel 计算只有几 µs 时 launch gap 占主导 → GPU 空转。
- **CUDA Graph**：eager 下每个 kernel 都过 driver；piecewise/full graph 把整步 replay 成一次 launch，基本消掉 launch overhead，但要求静态形状（padding + 分桶），首步/超大 batch 会退回 eager。
- **Sampler + D2H 同步**：要拿 sampled id 更新下一步输入，`.cpu()/.item()` 会 sync 整条 stream，等 forward 全部结束；V1 用 pinned buffer + CUDA event + async output 把 sync 推到下一步之前，但仍有固定延迟。
- **Engine Core ↔ API Server**：ZMQ 传 EngineCoreOutputs、增量 detokenize、SSE/JSON；客户端慢还会反压。batch 变大时这些固定成本被摊薄，所以小 batch 才最明显。

**实验设计**：固定 batch=1、prompt 128、output 128、greedy，warmup 20 步，分 graph on/off 两组。

1. **GPU kernel 时间**：`torch.profiler(activities=[CPU, CUDA])` 或 `nsys profile`，取每步所有 kernel 的 self CUDA time 求和；或 forward 前后插 CUDA event（先 synchronize，再 `elapsed_time`）。
2. **Host dispatch**：profiler 中 CPU 侧 Preprocess / execute_model 的耗时，加上 nsys 时间轴上 kernel 之间的 gap（GPU idle）；也可用 CUDA event 夹住“上一步最后一个 kernel 结束 → 下一步第一个 kernel 开始”。graph on/off 的差 ≈ launch 开销。
3. **端到端 TPOT**：客户端流式记录每个 chunk 到达的 `time.perf_counter()`，相邻差分取 p50/p99；同时记 engine 侧 EngineCoreOutputs 时间戳，两者差 = 输出处理 + 网络 + detokenize。

分解：TPOT ≈ kernel_sum + launch_gap + H2D/sync + sampling/D2H + frontend。前三项用 profiler/event 拿，剩下的用端到端减出来。

**为什么只用 `time.perf_counter()` 不行**：CUDA 是异步的。包住 `execute_model` 只测到“入队/下发”耗时，GPU 可能还没开始；一旦加 `.cpu()`/`.item()`，测到的是“入队 + 等 GPU + 采样 + 反序列化”的混合，无法拆分，也分不清“GPU 忙”和“GPU 等 host”。perf_counter 没有 kernel 级归因，看不到 launch gap、stream 重叠、跨进程 ZMQ/调度延迟；客户端还混入网络与 detokenize。要分离三个指标，必须有 GPU 侧时间戳（CUDA event / CUPTI / nsys）。
