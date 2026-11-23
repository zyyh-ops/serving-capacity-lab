## 第 1 题：名义 token 需求与非线性

|             | Input                     | Output                   | 合计             |
| ----------- | ------------------------- | ------------------------ | ---------------- |
| A P128/O128 | 20×128 = **2,560 tok/s**  | 20×128 = **2,560 tok/s** | 5,120 tok/s      |
| B P512/O128 | 20×512 = **10,240 tok/s** | 2,560 tok/s              | **12,800 tok/s** |

B 的总量是 A 的 2.5×，但 prefill token 是 4×，decode token 相同，因此不能用总 token 数线性换算：

- **TTFT 由请求的关键路径决定**：排队时间 + `ceil(前缀token / 本步可用budget) × 步时`。B 每个请求要算 4× 前缀，512 token 在共享预算下常被拆成多块。
- **算力消耗不是线性关系**：prefill attention 的开销随上下文增长；decode 每 token 要读 KV，B 的上下文是 128+128 对 512+128，同等 output token 数的成本不同。
- **批次形态会变化**：是否与 decode 同批、块边界、kernel 效率（GEMM/GEMV）、KV 占用都会改变单步时间。
- token budget 是全局硬性限制，不同到达顺序下排队不同。

**区分 Prompt 变长与 Prefix Cache 命中**：比较实际计算的 token（`num_scheduled_tokens` 之和）与名义 `num_prompt_tokens`；命中时 `get_computed_blocks` 让请求从 `num_computed_tokens>0` 起步，可直接看 `num_cached_tokens` 和 `vllm:prefix_cache_hits/queries`。实验上做两组对照：同 prompt 关闭 prefix caching 重跑；“P512 命中 384”与 P128 对齐有效 prefill token——若 TTFT 接近但 TPOT 仍高，说明是上下文变长而非 prefill。

## 第 2 题：Token Budget 与 Partial Prefill

- 剩余预算 = 2048 − 180×1 = **1868**。
- 先让 A 用 `min(1200, 1868)=1200`，余 668；B 得 `min(1000, 668)=668`，剩 332 留待下一步。因此 **A 完成、B 只做 668 token（partial）**。若 chunked prefill 关闭，B 会因 1000 > 668 整条被跳过。
- 不能保证固定比例：调度是 FCFS 下逐请求 `min(剩余prompt, token_budget)` 贪心扣减，没有按比例分配器；谁先入队谁先占，`long_prefill_token_threshold` 还会再截断单请求块，加上 `max_num_seqs`、KV 分配失败/抢占都会改变结果。若 B 先到，则 B=1000、A=868。
- **证明 Partial Prefill**：该步 `num_scheduled_tokens[req] < num_tokens - num_computed_tokens`，且下一步 `num_computed_tokens` 继续累加；执行侧 `_bookkeeping_sync` 检测到 `seq_len < req_state.num_tokens` 会丢弃采样并回退 generator；Trace 中同一 req 连续多步出现、每次 `scheduled_tokens` 小于剩余 prompt。

## 第 3 题：TPOT 升、TTFT 平

假设与判别：

- **H1 长 Prefill 抬高步时**：V1 先排 180 个 decode 各 1 token，长 prefill 只吃剩余预算，decode 不会被“饿死”，但同一 step 里多了 512 token 的 chunk，整步变慢 → TPOT 升；短请求若同 step 被调度，TTFT 也随之略升，若块小则看不出。
- **H2 总负载/常驻序列增加**：+5 req/s×O128 让 running、KV、attention 长度上升，步时上升。
- **H3 CPU/SSE 前端开销**：tokenization、detokenization、SSE 序列化每 token 都要做，客户端 TPOT 上升而引擎侧不变。
- 附加：KV 压力/prefix 驱逐/抢占重算。

最小额外实验（每次只改一个变量）：

1. **token 对齐对照**：20 short + 20 short（同总 token 率）vs 20 short + 5 long。TPOT 不升 → 排除“总负载”，指向 H1。
2. **同请求率、降 prefill 计算**：把 long 换成同前缀（cache 命中）或 P128。TPOT 回落 → H1。
3. **前端隔离**：同负载离线跑 EngineCore，或对比引擎侧 `first_token_ts→last_token_ts` 与应用侧 TPOT。客户端升、引擎不升 → H3。
4. 每步 trace 记录步时与 batch 组成，直接看“含 long chunk 的步”是否变慢。

## 第 4 题：类别级 Goodput

| 指标                  | Short                     | Long                   | 合计            |
| --------------------- | ------------------------- | ---------------------- | --------------- |
| 窗口完成率            | 1180/1200 = **98.3%**     | 280/300 = **93.3%**    | 97.3%           |
| Goodput（达标/60s）   | 1100/60 = **18.33 req/s** | 210/60 = **3.5 req/s** | **21.83 req/s** |
| 完成请求联合 SLO      | 1100/1180 = **93.2%**     | 210/280 = **75.0%**    | **89.7%**       |
| 相对计划数的 SLO 达成 | 1100/1200 = **91.7%**     | 210/300 = **70.0%**    | **87.3%**       |

整体 Goodput 21.83 req/s（计划 25 的 87.3%）会被 Short 的体量主导（Short 占 80%）。Long 完成率还有 93%，但完成请求里只有 75% 达标，即 70 条“完成但违约”；只报整体会把它稀释成 87%，看起来和 Short 差不多，实际上 Long 的 SLO 达成率低了 20+ 个百分点。必须分类型报告，并给出最差类和按计划 cohort 的口径。

## 第 5 题：不能断言 Chunked Prefill 导致 HoL

- **Waiting=0 不能作为反证或证据**：V1 每步把 waiting 队首请求按 chunk 搬进 running，长 prefill 被打散后算 running；waiting 是瞬时采样，会漏掉亚秒级/步内等待。TTFT +30% 也可能只是“短请求与长 chunk 同 step，步时变长”。
- **需要的 Scheduler 证据**：每步 trace 的 `num_scheduled_tokens`、`num_computed_tokens`、waiting 长度（每步采样，不是 0.5s）、budget 余量、被 skip 的请求、抢占。要证明：短请求在 waiting 中确实被跳过 ≥1 步、其排队时间≈TTFT 增量，且同期 budget 被 long chunk 占用；配合 OTel 的 `queued_time`/`prefill_time` 区分排队与计算。
- **替代解释**：①总负载增加（+2560 input tok/s，约 +50%），batch/步时上升；②KV 压力、prefix 驱逐、抢占重算；③CPU/SSE 前端；④非均匀 batch 触发 CUDA graph 回退/eager。
- **2048→512 变好 ≠ 越小越好**：TTFT 改善是因为单步长 prefill 有上限、短请求最多等一步；但块变小 → 步数、调度与 kernel launch 开销上升，prefill 吞吐下降，long 的 TTFT 变差，TPOT/goodput 可能恶化；还要受 capture size、attention 分块效率约束。必须同时看 short/long 的 TTFT、TPOT、goodput，并用 trace 确认收益来自“排队减少”，而不是 batch 组成变化。
