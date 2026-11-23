## 第 1 题

由时间戳可推出：共生成 4 个 token，因此只有 3 个 ITL。

| 指标               | 计算                        | 结果                                                         |
| ------------------ | --------------------------- | ------------------------------------------------------------ |
| TTFT               | 100 − 0                     | **100 ms**                                                   |
| ITL₁ / ITL₂ / ITL₃ | 120−100 / 155−120 / 175−155 | **20 / 35 / 20 ms**                                          |
| 平均 TPOT          | mean(ITL) = 75/3            | **25 ms**（若用 (E2E−TTFT)/(N−1) = 80/3 ≈ **26.7 ms**，多出的 5 ms 是收尾开销） |
| E2E                | 180 − 0                     | **180 ms**                                                   |

**为什么 SSE chunk 数量不能始终当作 token 数：**

1. 一个 chunk 可能包含多个 token，例如多步调度、投机解码一次输出 k 个 token，或增量 detokenize 合并了若干 token。
2. 一个 token 的文本可能被拆进多个 chunk，例如 UTF-8 多字节字符、转义序列、detokenizer 缓冲 flush 导致它被拆开。
3. 有些 chunk 不含 token：role 首包、`finish_reason`/usage 尾包、keep-alive/ping、空 delta。
4. 网络或缓冲会把多个 SSE 事件合并成一次读取，反之也可能被拆开。
5. 工具调用等场景的 delta 是结构化片段，不是 token。

结论：token 数应基于 `usage.completion_tokens` 或自行分词统计，不要数 chunk。

## 第 2 题

- **`--request-rate inf --max-concurrency 8`**：闭环、有背压。8 个 worker 发完一个才发下一个，在途请求恒 ≤ 8，实际到达率 = 8 / 时延，会自动降速。
- **`--request-rate 8`**：开环、恒定速率，按 8 req/s 注入，不管服务端是否完成。

服务器只能稳定做 **5 req/s**，第二组会发生：注入速率 > 服务能力，队列无界增长（Little's law: L = λW，W 线性上涨），TTFT/TPOT/E2E 持续恶化、超时和报错增多，KV cache 与内存压力上升，但吞吐仍被压在上限约 5 req/s（甚至因抖动更低），并可能触发 preemption。简单说，这是过载崩溃，而非等价负载。

**为什么第一组会掩盖内部排队**：客户端自身限流形成背压，服务变慢后 worker 阻塞，到达率被压到与服务器处理能力一致，在途请求数始终只有 8，服务端内部队列很短，因此看不到积压；同时这条“软负载”无法测出饱和点和 SLO 边界。

## 第 3 题

**瓶颈判断**：KV cache 使用率仅 35%，且 preemption 始终不变，说明瓶颈不在显存。`num_requests_waiting 明显增加` 表明请求卡在准入/调度环节，主要可能是：

- **Scheduler token budget**：`max_num_batched_tokens` 被长 prefill 吃满，chunked prefill 排队，造成队头阻塞；
- **Active sequence limit**：`max_num_seqs` 已满，running 达到上限，新请求只能进 waiting；
- 次要可能：每步 CPU/调度开销、detokenize 或前端开销成为固定瓶颈。

**为什么不能看 TTFT 恶化就判断 KV 不足**：TTFT = 排队时间 + prefill 计算。KV 不足的典型信号是 KV 使用率接近上限，preemption 激增，并伴随 swap/recompute；本题 KV 仅 35%，preemption 为零，说明是“有内存但不让进”，即调度/并发上限问题；长 prompt 的 prefill 计算同样会推高 TTFT。

**建议补充的指标**：

- 区分 token budget：每步 `num_batched_tokens` 及其占 `max_num_batched_tokens` 的比例、prefill 与 decode 的 token 配比、prefill 排队时长、每步 batch 组成。
- 区分 active seq limit：`num_running_reqs` 是否恒等于 `max_num_seqs`、running/waiting/swapped 三条曲线、队列等待时间直方图。
- 区分 GPU 计算瓶颈：DCGM GPU 利用率/SM occupancy、每步 `iteration_time` 分解（调度 vs forward vs detokenize）、MFU/FLOPs 利用率、Nsight Systems 中 kernel 间隔与 tensor pipe 吞吐；再对比“加并发后 token/s 是否还涨”。

## 第 4 题

- **Throughput = 100 / 10 s = 10 req/s**
- **Goodput = 70 / 10 s = 7 req/s**（即 70% 请求满足双 SLO）

**为何到达率提高后 Throughput 升、Goodput 反降**：Throughput 只看完成数，负载高时 batch 更满、单位时间完成更多请求；但排队时延把 TTFT/TPOT 推过 SLO，满足双 SLO 的比例下降。过了饱和点后 Throughput 趋于平台，Goodput 继续掉，因为“做完了但都不达标”。

**改善 Goodput 的方向**：

- Admission Control：SLO-aware 准入、过载时主动拒绝/降级，避免无效排队。
- Scheduler：按 SLO 优先级调度、chunked prefill 降 TTFT、prefill/decode 分离部署、避免队头阻塞。
- Batching：调 `max_num_seqs` / `max_num_batched_tokens`，continuous batching + 多步调度、投机解码降 ITL、CUDA Graph 降固定开销。
- KV Cache：PagedAttention 块大小、prefix caching 复用系统提示、KV 量化或扩显存余量以减少 preemption。

## 第 5 题

- **更受 Prefill 计算影响：B**（512 token prompt，prefill 占主导，TTFT 明显更高）；A 的 64 token prefill 可忽略。
- **更易暴露 Decode Host Launch Overhead：A**（256 次 decode step，每步计算量小、固定开销占比高；B 只有 32 步，且被长 prefill 摊薄）。
- **TTFT / TPOT / 输出吞吐变化**：A：TTFT 低，TPOT 受每步固定开销影响相对偏高，但因输出长，输出 token 吞吐较高；B：TTFT 高，TPOT 受长 KV 读取略升但总步数少，输出吞吐（32 token / E2E）明显偏低，E2E 主要由 prefill 决定。
- **为什么不能直接比 E2E 判断哪条执行路径更高效**：两者工作量与 token 配比不同（A 320 token、B 544 token，且 prefill/decode 比例相反），E2E 把“做了多少活”和“做得多高效”混在一起；短 E2E 可能只是活少。应比归一化指标：TTFT、TPOT/ITL、tokens/s、每 token 成本、MFU。
- **实验设计（控变量 + 观测）**：固定模型/GPU/dtype/并发/到达率与 CUDA Graph 配置，只改 input/output 长度（如 64/512 × 32/256 的四组矩阵），预热后多次取 p50/p95；用 `vllm bench serve` 的 `--random-input-len/--random-output-len` 控制负载；Prometheus 看 `time_to_first_token_seconds`、`time_per_output_token_seconds`、`e2e_request_latency_seconds`、`num_requests_running/waiting`、`gpu_cache_usage_perc`、prompt/generation tokens 速率；DCGM 看 GPU 利用率；Nsight Systems 抓 timeline，看 kernel 间空隙、`cudaLaunchKernel` 占比，并用 `--enforce-eager` vs CUDA Graph 对比隔离 launch overhead，用 NVTX 标记 prefill/decode 区间做归因。