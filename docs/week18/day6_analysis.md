# Week 18 Day 6 — SLO 容量 / 准入 / 过载恢复

> 三个 mode 分开报：**open**（全量注入）、**rps gate**（客户端令牌桶，78/s + burst 20）、**cost gate**（按 `P+2O` 计费，24,000/s + burst 6,000）。`local_reject` 是**客户端**拒绝、请求从未进 vLLM，不是 HTTP 429；但它留在 offered 分母里。

## 0. 结论先行

1. **本配置实测到的最高"全达标"点 = 90 req/s**（`sweep_90`：3,588/3,588 全 200、0 reject、cohort SLO **100%**、TTFT p99 **81.8 ms**、TPOT p99 **11.52 ms**）。这是**单 seed / 40 s 的探索性 pilot**，不是容量证书；90 → 105 之间没有逐点测，拐点位置未知。
2. **过载（105 req/s）不是"排队崩"，而是"并发顶满 → TPOT 通胀"**：running 钉在 **253**（`max_num_seqs=256`）、KV 峰 **0.799**、**preemptions 0**、sampled waiting 峰只有 **19**（且只在最后 ~3.5 s）；代价是 TPOT p50 **16.93** / p99 **18.89 ms**（基线 5.53/6.88）⇒ 25 s 内 2,601 个请求只有 **476 个**达 SLO（goodput **19.04 req/s**，17.0%）。
3. **fail-fast 令牌桶把"全部违约"换成"24.4% 拒绝 + 全达标"**：`recovery_gate` 在 overload 只放行 **78.68 req/s**，accepted-only SLO **100%**、goodput **77.52 req/s**（open 19.04，**4.1×**）；代价是 offered 口径只兑现 **75.6%**。
4. **队列清零 ≠ 延迟恢复**：open 臂的 `recovery_clear_s = 0.106` 是采样假象（waiting 全程只有 5 个非零样本，且全在 overload 末段）；真正的恢复是 **p50 ~5 s、p99 ~10–15 s**——60–65 s 桶 TTFT p50 仍 **49.2 ms** / TPOT p50 **8.05 ms**，到 65–70 s 桶才回到 **24.5 / 5.62 ms**。
5. **混合负载的 `P+2O` 配额是"配出来的"，不是量出来的**：同一 60 req/s 计划下 `mix_open` **100%** 达标（goodput 59.08），`mix_cost` 却主动拒掉 **472 个（13.2%）**，goodput 反而降到 **51.22（-13.3%）**。名义需求 28,800 成本/s > 配额 24,000/s ⇒ 拒绝是配置算术的结果；且 long 拒 **27.8%** vs short **8.5%（3.3×）**，不是 2× 权重比。
6. **饱和吞吐 ≈ 12.2k tok/s**：90 req/s → 11,274 tok/s；105 过载 → **12,170 tok/s**（与 Day5 同配置 rate=100 的 12,168.5 一致）。需求再加 10% 只多 8% 吞吐，却把 TPOT p99 从 11.5 拉到 18.9 ⇒ 边际收益为负。

## 1. 实验配置

| Variant | 计划/seed | Policy | Phases (s:req/s) | 关键参数 |
|---|---|---|---|---|
| `sweep_70/80/90` | 40 s 每点，seed 2031 | open | `reference:40:70/80/90` | long_frac 0，无拒绝 |
| `recovery_open` | seed 2031 | open | `baseline:35:65, overload:25:105, recovery:60:65` | long_frac 0 |
| `recovery_gate` | seed 2031 | rps | 同上 | `--admit-rate 78 --burst 20` |
| `mix_open` | seed 2031 | open | `mixed:60:60` | long_frac **0.25** |
| `mix_cost` | seed 2031 | cost | `mixed:60:60` | `cost-rate 24000`、`burst 6000`、`w1=1,w2=2` |

**服务端（五个臂共用同一个进程，未重启）**：vLLM **0.10.2 V1**、`Qwen/Qwen3-0.6B` bf16、RTX 4090 24 GB（570.124.06）、`max_model_len 2048`、`gpu_memory_utilization 0.7`、`max_num_seqs 256`、`max_num_batched_tokens 2048`、`--enable-chunked-prefill`、`--no-enable-prefix-caching`、`num_gpu_blocks 8254 → override 4096`（block 16 ⇒ **65,536 tok = 32.00x**）、piecewise cudagraph（capture ≤512）。

- 运行顺序固定：sweep(06:22–06:24) → recovery_open(06:25) → recovery_gate(06:27) → mix_open(06:37) → mix_cost(06:38)；**无 AB/BA 交叉**。
- 所有臂同一 Poisson 计划生成器 + 同 seed ⇒ baseline 两臂 offered 完全一致（2,270 / 2,601 / 3,935），这是两臂可比的前提。
- **4096 blocks 与 `max_num_seqs=256` 在 128+128 负载下恰好等价**：256 req × 256 tok = 65,536 tok。实测 running 253 时 KV 0.799（余 824 blocks ≈ 100 次 prefill）⇒ 按线性外推 KV 要到 ~316 并发才满，**本负载下先绑的是 `max_num_seqs`，KV 只落后约 23%**；两个限制在这个实验里无法分离。

## 2. SLO 容量候选

| 点 (req/s) | Injected | Completed | Goodput | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | E2E p50 | out tok/s | running 峰 | KV 峰 | Good/Offered |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 69.95 | 68.95 | 68.95 | 26.8 / **55.8** | 5.78 / **7.25** | 763 | 8,826 | 84 | 0.262 | 100% |
| 80 | 80.10 | 78.65 | 78.65 | 32.6 / **69.1** | 6.80 / **8.67** | 900 | 10,067 | 107 | 0.337 | 100% |
| **90** | 89.70 | 88.08 | 88.08 | 41.1 / **81.8** | 8.32 / **11.52** | 1,098 | 11,274 | 147 | 0.465 | **100%** |
| 105（overload 臂） | 104.04 | 95.08 | **19.04** | 82.0 / **204.6** | 16.93 / **18.89** | 2,245 | 12,170 | 253 | 0.799 | 17.0% |

- **最高候选 = 90 req/s**，判据是 TTFT ≤200 ms 与 TPOT ≤15 ms 的**联合**达标（cohort 3,588/3,588）；90 点 TPOT p99 已用掉 SLO 的 **77%**，余量只有 23%。
- **为什么只能算 pilot**：单 seed（2031）、每点 40 s、每点只跑一次、无 100 点、无重复、无 AB/BA；且 40 s 窗口的完成率被"起始空管道"系统性压低 ≈ in-flight/T（in-flight 40–50、E2E 0.9–1.1 s ⇒ **-1.0 ~ -1.2 req/s**），所以 90 的真稳态完成率 ≥ 88.08（≈ 89.7 注入）。
- **不要复用 Day2 的 80**：Day2 的 80 是另一套 server 配置下的确认点；本配置实测 90 达标、80 只用掉 58% TPOT 预算。也不要把 90 当认证：Day5 同配置的 rate=100 窗口 TPOT p99 为 16.81 ms（>15），与本表 105 点的 18.89 同向，说明拐点落在 **90–100** 而不是 90 以上很远处。
- **缺的证据**：100 点；≥3 个 seed；≥300 s 窗口；门限敏感性（TTFT 100/200、TPOT 10/15）；重跑顺序交叉；单点重复以给出 p99 的置信区间。

## 3. Admission versus availability

| Arm | Phase | Offered | Injected | Local reject | HTTP 失败 | Window goodput | Good/Offered | Accepted-only SLO |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| open | baseline 65 | 2,270 | 2,270 | 0 | 0 | 63.74 | 100% | 100% |
| open | overload 105 | 2,601 | 2,601 | 0 | 0 | **19.04** | **17.0%** | **17.0%** |
| open | recovery 65 | 3,935 | 3,935 | 0 | 0 | 64.40 | 99.7% | 99.7% |
| rps gate | baseline 65 | 2,270 | 2,270 | 0 | 0 | 63.74 | 100% | 100% |
| rps gate | overload 105 | 2,601 | **1,967** | **634** | 0 | **77.52** | 75.6% | **100%** |
| rps gate | recovery 65 | 3,935 | 3,926 | 9 | 0 | 65.68 | 99.8% | 100% |
| **(总计, gate vs open)** | 120 s | 8,806 | 8,163 vs 8,806 | 643 vs 0 | 0 | 68.0 vs 55.3 均速 | **92.7% vs 75.3%** | 100% vs 75.3% |

- **baseline 两臂完全相同**（0 reject、0 差异）⇒ 78/s 的门在 65/s 下不触发，这一行是"计划一致"的对照组。
- **门口只买到 accepted 的 SLO 吗？不是。** 在 overload，open 的 2,601 个成功里只有 441 个达标（17.0%），gate 拒掉 24.4% 后剩下 1,967 个 **全部**达标；**按 offered 口径**的 goodput 从 19.04 升到 77.52（**4.1×**）。所以这不是"用可用性换指标"，而是把注定违约的那部分请求提前挡在系统外。
- **代价是真实存在的**：offered 口径只兑现 75.6%（vs open 100% "送达"），整个 120 s 累计 good 8,163/8,806 = 92.7%（open 6,633 = 75.3%），均值 good 速率 68.0 vs 55.3 req/s（**+23%**）。
- **生产里 429 的含义不同**：真实 429 是**服务端**拒绝（有 `Retry-After`、计入可用性/错误率 SLO、可见于网关与客户端重试），且系统确实收到了负载；这里的 `local_reject` 发生在 `session.post` 之前（`dispatch_t=None`），**服务端完全不知道**，所以"服务端 goodput 漂亮"与"端到端 24% 需求被丢弃"可以同时成立。gate 还是 **fail-fast 而非排队**：拒绝哪个请求取决于桶的瞬时余额，不是 FIFO，因此拒绝在时间上可预测（与负载形状绑定），但对个别请求不可预测。

## 4. Overload → recovery

| Arm | Phase | 注入 req/s | 窗口完成 | 窗口 goodput | waiting 峰 | in-flight 期末 |
|---|---|---:|---:|---:|---:|---:|
| open | baseline | 64.86 | 63.74 | 63.74 | 0 | 39 |
| open | overload | 104.04 | 95.08 | 19.04 | **19** | **263** |
| open | recovery | 65.58 | **68.93** | 64.40 | 0 | 62 |
| gate | overload | 78.68 | 77.52 | 77.52 | 0 | 68 |
| gate | recovery | 65.43 | 65.68 | 65.68 | 0 | 53 |

**机制**：注入 104/s > 稳态服务能力，多出来的需求不是被"排队"，而是被**吸收成每请求更慢**——9/s 的缺口全部变成 in-flight 增长（39 → 263，E2E 0.73 → 2.25 s），running 一路顶到 253/256、KV 0.799。2,601 注入中 2,377 在窗口内完成，剩下 **224 个在 recovery 开始后的几秒内完成**；这解释了 recovery 相 `completed 68.93 > 注入 65.58`，**不是容量变高**。gate 臂因为没有超额注入，running 只到 81、KV 0.25，全程没有队列，也就没有"恢复期"。

**三级恢复必须分开**：

| 级别 | 观测 | 说明 |
|---|---|---|
| queue clear | `0.106 s` | 采样假象：waiting 只有 56.3–59.8 s 的 5 个非零样本，recovery 开始时本就是 0 |
| TTFT/TPOT 恢复 | p50 **~5 s**、p99 **~10–15 s** | 5 s 桶：60–65 s 仍 49.2/8.05 ms（p99 62.7），65–70 s 的 p50 回到 24.5/5.62 ms（基线 24.9/5.53），p99 到 75–80 s 才回到 50.8 ms |
| 完全排空 | ~120 s 窗口末 | in-flight 从 263 回到 62（基线稳态 39–56） |

- recovery 相的 cohort p99 仍被污染：TTFT p99 **98.0 ms**、TPOT p99 **13.84 ms**（基线 52.6/6.88）——因为 cohort 里含 overload 期间到达的请求；这是"cohort vs window"两套口径必须分开报的原因。
- gate 臂的 recovery 与 baseline 无差（53.1/7.22 vs 51.6/6.89），且只残余 9 个 reject（overload 把桶打空后的回填）。

## 5. Mixed request fairness（`mix_open` vs `mix_cost`）

| Arm | 类别 | Offered | Good | Good/Offered | 被拒 | TTFT p99 | TPOT p99 |
|---|---|---:|---:|---:|---:|---:|---:|
| open | short (P128/O128) | 2,717 | 2,717 | **100%** | 0 | 61.7 | 9.90 |
| open | long (P512/O128) | 871 | 871 | **100%** | 0 | 66.5 | 9.98 |
| cost | short | 2,717 | 2,487 | **91.5%** | 230（8.5%） | 49.6 | 6.93 |
| cost | long | 871 | 629 | **72.2%** | **242（27.8%）** | 53.9 | 6.89 |

- **总量判决**：60 req/s 混合负载在 open 下 **本来就全达标**（goodput 59.08），cost 门却拒了 13.2% 只换来 goodput 51.22 ⇒ 在这个负载上**准入是净损失**。它把尾延迟压下来了（TPOT p99 9.92 → 6.93、TTFT p99 63.8 → 51.4），但那是"并发从 92 降到 52"的结果，不是"准入让服务更快"。
- **拒绝是桶的算术，不是容量判决**：60/s × (0.75×384 + 0.25×768) = **28,800 成本/s**，配额只有 24,000/s；实测接受成本 1,438,080 vs 可用 1,446,000（**99.4% 用满**）⇒ 472 个拒绝是确定性结果。
- **`P+2O` 是否预测真实服务成本？** 名义 384 vs 768（2×），但实测同一 batch 内 long 与 short 的 p99 几乎相同：TPOT 6.89 vs 6.93（open 9.98 vs 9.90），TTFT 只差 4–5 ms。输出长度相同（128）时，512 的 prefill 摊进 128 步解码后增量很小 ⇒ **2× 权重把 long 高估了**。注意这是混合 batch 下的观测，不是隔离测量。
- **公平性**：按请求看 long 被拒概率是 short 的 **3.3×**；按成本份额看"公平"应是 2×。3.3× 的机制是 fail-fast 桶在余额 ∈[384,768) 时只能放 short ⇒ **单一令牌桶 + 重成本项 = 对重请求系统性惩罚**。要按权重公平分配，应做分层/加权桶（WFQ），而不是调大 burst。

## 6. Alternative explanations and cautions

- **没有服务端准入插件**：所有拒绝都在客户端；服务端从未返回 429，`http_error_or_timeout` 全 0，因此"可用性"在这里是**客户端定义**的。
- **client policy 改变了实际负载**：gate 臂注入 78.68/s 而 open 是 104.04/s，两臂的到达曲线不同 ⇒ 不能把"gate 的 TPOT 更好"读成"gate 让服务更快"。
- **顺序/状态**：同一 server 进程、无重启、顺序固定（sweep → open → gate → mix_open → cost），图缓存与页面/显存状态可能随顺序漂移，未做 AB/BA。
- **单 seed + 短窗口**：40–60 s、每点一次，p99 对 Poisson 抖动敏感；同一 65/s 的 baseline 两臂 TTFT p99 差 1.0 ms、overload 的 gate 与 baseline 差 4 ms，可视为噪声尺度。
- **采样口径**：waiting/KV 是 0.25 s 采样，峰是"采样到的峰"；且 open 臂的过载不在 scheduler 队列里（见 §4）⇒ 用 waiting 做告警/准入信号会漏报。
- **`local_reject` 不进服务端分母**：生产里被拒/超时会计入故障率与重试放大，这里没有。
- **混合两臂的 long 采样**：`long_frac=0.25`、871 个 long，p99 由约 9 个样本支撑，尾部估计不稳。

## 7. Release decision

**Observed**：本配置 4096 blocks/65,536 tok、`max_num_seqs=256`、chunked prefill、prefix cache 关；sweep 70/80/90 全部 100% 达标（TPOT p99 7.25/8.67/11.52，TTFT p99 55.8/69.1/81.8）；105 过载下 running 253、KV 0.799、preemptions 0、waiting 峰 19、goodput 19.04、TPOT p99 18.89；gate 78/s 下 overload accepted SLO 100%、goodput 77.52、拒 634；混合 60/s 下 open 100% 达标而 cost 拒 472（13.2%）且 goodput 低 13.3%，long 拒 27.8% vs short 8.5%；recovery_open 的 queue-clear 0.106 s、恢复 p50 ~5 s / p99 ~10–15 s、in-flight 263 → 62。

**Inferred**：拐点落在 90–105（Day5 rate=100 的 16.81 佐证 90–100 之间）；过载的主失效是**并发饱和导致的 TPOT 通胀**，不是 scheduler 排队（本负载下 `max_num_seqs` 先绑、KV 落后约 23%）；gate 的收益来自"把违约定额挡在系统外"，其成本是 offered 口径的兑现率；cost 门的 472 个拒绝可由桶算术完全解释，且对重请求不公平。

**Unknown**：100 点的实际位置；多 seed / 长时间窗下的稳定性；服务端自身准入（队列上限/429）会怎样改变失效模式；`P+2O` 权重与真实 GPU 时间的换算；open 臂过载期间的请求究竟停在 API frontend / 引擎输入队列 / 客户端哪一层（waiting=0 但 in-flight=263）。

**安全运行点（有依据的）**：

- **全量注入 ≤ 80 req/s**：相对实测 90 有 ≥11% 余量，TPOT p99 8.67 ms 只用掉 SLO 的 58%，TTFT p99 69 ms 用掉 35%。
- **过载保护用 78 req/s 全局限流 + burst 20**：实测在 105 req/s 冲击下 accepted SLO 100%、拒绝 24.4%、恢复期无残留尾巴。
- **不要承诺 90**：单 seed、单窗口、无 100 点；90 点 TPOT p99 已用掉 77% 预算，p99 的抽样误差足以翻过 15 ms。

**准入策略与缺点**：优先 **rps 门**（可解释、行为与负载形状弱相关、无需成本模型）；**cost 门在权重被实测校准前不要用**——它会把重请求拒到 3.3× 并使总 goodput 下降。两种门都只是**客户端 fail-fast**，正确形态应是网关侧限流 + 真实 429（带 `Retry-After`）+ 分级 SLO + 重试预算。

**上生产前必须补的实验**：100 点与 95 点；≥3 seed × ≥300 s；AB/BA 顺序交叉；long_frac 0.1/0.5 下的再标定；用实测 token-time 校准 cost 权重并改分层桶；对照服务端准入（`max_num_seqs` 收紧 / 队列上限）与客户端限流的失效模式差异。
