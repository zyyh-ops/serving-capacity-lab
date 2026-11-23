## 第 1 题：SLO Goodput 与拒绝率

- Offered Rate = 6000 / 60 = **100 req/s**
- Accepted Rate = (6000 − 1200) / 60 = **80 req/s**
- Completed Throughput = 4600 / 60 ≈ **76.7 req/s**
- SLO Goodput = 4200 / 60 = **70 req/s**
- 拒绝率 = 1200 / 6000 = **20%**
- 所有 Offered 的 SLO 达成率 = 4200 / 6000 = **70%**
- Accepted-only SLO Rate = 4200 / 4800 = **87.5%**；若只算成功请求，则 4200 / 4600 ≈ **91.3%**

**误导原因**：Accepted-only 的分母把 1200 个拒绝和 200 个失败移出去了，准入控制只要“拒掉难请求”，就能把这个比例刷高；它衡量的是幸存者质量，而不是所有用户拿到好体验的比例，也不能跨准入策略比较。业务应同时看 offered-level goodput、拒绝率和失败率。

## 第 2 题：Token Bucket 与 Open-loop

- 上限 = 初始 burst 20 + 80 × 60 = **4820**（若空桶起步则 4800）。6000 > 4820，桶会长期见底，实际通常接近上限。
- 实际可能更少：burst 只能吸收 20 个瞬时超额，Poisson 成簇到达会成批被拒；桶满时新补充的 token 被容量截断，窗口末尾的残余 token 不能结转；fail-fast 不排队、不重试。客户端实现见 [capacity_probe.py:120](scripts/capacity_probe.py#L120)（按 `planned_t` 计费、满桶起步）。
- `max_concurrency=80` 是**在途并发**上限（闭环背压）：吞吐 ≈ 80 / 平均时延；满了让请求排队，表现为时延膨胀而不是拒绝。80 并发在 1 s 时延下是 80/s，2 s 时只有 40/s。Token bucket 是**准入速率**上限（开环）：与请求多久完成无关，超速立即拒绝，保护尾延迟。vLLM 内部相近的是 `max_num_seqs`（running 上限），不是速率。

## 第 3 题：Workload-aware Admission

- 短：128 + 2×128 = **384**；长：512 + 2×128 = **768**
- 名义总需求 = 40×384 + 10×768 = **23,040 cost/s**
- 15,000 只覆盖 **65.1%**，缺口 8,040（35%）；连 40 short/s 都已 15,360 > 15,000。
- 长期全收时，积压会线性增长；在 vLLM 里就是 waiting 变长、KV 满后 preempt 重算，SLO 崩。
- 固定 RPS 不行：同样 50 req/s，全短为 19,200、题中混合为 23,040；长请求 P 大且 attention 随 P² 增长，RPS 无法表达“每个请求有多贵”。
- `P+2O` 不是 GPU 时间：prefill 算力受限（含 O(P²) attention），decode 带宽受限；batching 摊薄权重、prefix cache、chunked prefill、KV 占用都会改变单请求成本，成本也不可按请求简单相加。
- 校准：固定 `max_num_seqs` / `max_num_batched_tokens` 做短/长微基准，测每类 GPU 时间，回归 `cost = aP + bO + cP²`（可加 batch 项），用 TTFT/TPOT/queue time 交叉验证，再用独立 seed 验证预测的可持续 λ。

## 第 4 题：Overload Recovery 与 Queue Conservation

- 净排空 = 90 − 60 = 30/s，因此 180 / 30 = **6 s**。
- 但 TTFT P99 ≠ 清空时刻：TTFT = 排队 + prefill；最后一批请求排队近 6 s，它们的 TTFT 仍 ≥ 6 s，P99 要等这批慢请求离开统计窗口才回落。清空只说明 waiting≈0，不代表 running 批和 KV 压力恢复。
- 真实 vLLM 失效点：按 token 预算调度且 running 优先（[scheduler.py:196](vllm/v1/core/sched/scheduler.py#L196)），waiting 还受 `max_num_seqs`/KV 限制（[scheduler.py:333](vllm/v1/core/sched/scheduler.py#L333)）；KV 不足会抢占并 `num_computed_tokens=0` 重算（[scheduler.py:269](vllm/v1/core/sched/scheduler.py#L269)），等效服务率下降；chunked prefill、prefix cache、变长输出、新到达、采样窗口都会让“60/90 恒定”失效。队列按“请求数”算，实际工作量按 token 算，180 个长短混合请求不是同质工作。
- 恢复判据：cohort 的 joint SLO attainment 连续 N 个窗口达标 + p99 TTFT/TPOT 达标 + waiting=0 + KV 利用率低于阈值 + 无 preemption，而不是“队列清空”瞬间。

## 第 5 题：Admission Policy 的因果评估

|              |                Open |                  Gate |
| ------------ | ------------------: | --------------------: |
| Offered SLO  | 3500/5000 = **70%** |   3900/5000 = **78%** |
| 成功条件 SLO |                 70% | 3900/4000 = **97.5%** |
| 拒绝率       |                   0 |               **20%** |
| p99 TTFT     |              450 ms |             **90 ms** |

- 净效果：good +400（+11.4%），success −1000（−20%）。Gate 让 1400 个原本的 SLO 违约消失：其中 400 转为达标，1000 转为拒绝；代价是成功数 −1000。
- 四角度：Goodput 小增；20% 拒绝是硬成本；被接纳者体验 5× 改善，但 1/5 用户直接失败；业务可用性取决于“慢成功”和“快速失败”哪个更贵——违约赔偿/流失高且拒绝可重试、可降级时 Gate 有价值，反之未必。示意：一次违约代价 > 约 0.71 次成功收入，Gate 才净赚。
- 单轮不能定论：seed、顺序、cache/GPU 状态、workload 都会混淆；450→90 ms 也可能只是少接了 20% 流量。
- 验证设计：3–5 seeds，每 seed 两臂用同一条 offered trace；AB/BA 交叉（Open→Gate / Gate→Open），换臂重启 server 或清 prefix cache/KV，固定 warmup；预注册主指标 offered-level joint SLO goodput，护栏为拒绝率、success、p99 TTFT/TPOT、preemption；覆盖 short-only、25% long、burst/overload-recovery、λ=70/90/105；按 seed 配对差 + bootstrap CI，只有下界 >0 且拒绝率在业务可接受范围才采用。

**源码锚点**：调度器先跑 running、扣 `token_budget`，再用剩余预算接 waiting；KV 不足则 preempt 并重置重算（[scheduler.py:177](vllm/v1/core/sched/scheduler.py#L177)）。对应指标是 `num_requests_waiting/running`、`kv_cache_usage_perc`、`num_preemptions` 和 TTFT/TPOT 直方图（[loggers.py:188](vllm/v1/metrics/loggers.py#L188)）。
