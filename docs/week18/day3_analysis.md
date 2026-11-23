# Week 18 Day 3 — Workload Shape Analysis

## 0. 结论先行

- 7 个 run（seed 2027 ×5 + seed 2028 ×2），合计 **53,407** 条请求，全部 HTTP 200、0 失败、0 `client_guard_drop`、preemptions 差分 0。
- **形状轴**（10 req/s）：P128/O128 TTFT P50/P99 = **16.21/30.18 ms**、TPOT **4.53/5.07 ms**；P512/O128 = **27.25/43.94**、**5.05/5.72**；P128/O512 = **20.85/38.46**、**5.46/6.36**。三点 waiting 全 0，goodput ≈ 10 req/s，无容量拐点。
- **干扰轴**（短流完全相同，另加 5 req/s 长流）：短类 TTFT P50 **16.77 → 18.23 ms（+8.7%）**、P99 **31.82 → 38.29（+20.4%）**；TPOT P50 **4.79 → 5.33（+11.2%）**、P99 **5.27 → 6.15（+16.7%）**。短类 goodput 19.87 → 19.90，整体 **24.84**，joint SLO **1.0**；2028 复现为 +11.3% / +8.9%（P50）。
- **这不是队列阻塞**：9,697 个 `/metrics` 样本 waiting 全 0，running ≤52，KV ≤0.158；偏移来自 prefill/decode 共享 step。长 prefill 重叠时短 TTFT 中位 +4~9 ms；TPOT 随 in-flight 单调上升。
- **4 条 SLO 违约**（long_prefill 3 条、control_2028 1 条，TTFT 201–264 ms）都与 ~200 ms 客户端事件循环停顿重合（`dispatch_lag` 197/204 ms；0.495 s metrics 间隔），当时服务端 waiting=0，最可能是客户端测量伪影。
- **证据边界**：Day 3 无 `vllm serve` 日志；budget 2048 / chunked prefill / prefix cache 沿用 Day 1（C 级）；只能确认 `max_num_seqs ≥ 52`；主实验 single seed；duration 60/133/280/575/1172 s 不等。

## 1. Experiment contract

| 项 | 取值 |
|---|---|
| 模型 / 版本 | `Qwen/Qwen3-0.6B`，vLLM **0.10.2**，RTX 4090 24 GB，CUDA 12.8（Day 1 记录） |
| 请求 / SLO | `temperature=0`、`ignore_eos=true`、stream + usage；TTFT ≤ **200 ms** 且 TPOT ≤ **15 ms** |
| 到达 | 每类独立 Poisson；short 用 `seed`，long 用 `seed+100003`；按 `planned_t` 排序 |
| 长度 | short 128/128 或 512/128；long 512/128；`range_ratio=0`（实际 prompt 有低尾，见 §4） |
| Warmup / cache | `t0` 前每类 1 条非流式 warmup；prefix cache 开启，未清空，跨 run 累积（推断） |
| 客户端 | 单进程 asyncio + aiohttp；guard 512；timeout 90 s；`/metrics` 0.25 s；server 同机 loopback |
| Server CLI | **未记录**；沿用 Day 1：`vllm serve ... --dtype auto --max-model-len 2048 --gpu-memory-utilization 0.7` |

| run | T (s) | seed | target short/long req/s | P/O | n |
|---|---:|---:|---:|---:|---:|
| shape_baseline | 60 | 2027 | 10 / 0 | 128/128 | 615 |
| shape_long_prefill | 133 | 2027 | 10 / 0 | 512/128 | 1,332 |
| shape_long_decode | 280 | 2027 | 10 / 0 | 128/512 | 2,807 |
| short_control | 575 | 2027 | 20 / 0 | 128/128 | 11,442 |
| short_plus_long | 1,172 | 2027 | 20 / 5 | 128/128 + 512/128 | 29,125 |
| control_2028 | 180 | 2028 | 20 / 0 | 128/128 | 3,565 |
| mixed_2028 | 180 | 2028 | 20 / 5 | 128/128 + 512/128 | 4,521 |

- **固定 req/s**：三个 shape run 目标都是 10 req/s（实际 10.015–10.25，Poisson 抖动），只改 P 或 O。
- **固定短流**：`short_control` vs `short_plus_long`（及 2028 对）；short 的 `planned_t` 和 prompt token 数逐条一致（§3）。
- **最干净的单变量对照**：baseline vs long_decode，前 615 条同到达计划、同 prompt（token 数逐条一致），只改 output 128 → 512。
- **口径**：`T=duration`；`injected=count(dispatch_t<T)/T`；`completed/goodput` 只算 `end_t<T`；尾延迟对最终成功统计，**含 Drain**；`joint_slo_cohort` 只除以成功数（不含失败/drop）。
- 数据：`comparison.csv` 只含 2027 的 5 个 run；2028 两轮直接读各自 `summary.json`。

## 2. Shape-only comparison

| Case | 注入 req/s | 平均 P | O | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | Goodput | Waiting 峰 | Running 峰 | KV 峰 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P128/O128 | 10.25 | 135.4 | 128 | 16.21 / 30.18 | 4.53 / 5.07 | 10.05 | 0 | 14 | 0.0222 |
| P512/O128 | 10.015 | 518.4 | 128 | 27.25 / 43.94 | 5.05 / 5.72 | 9.9323 | 0 | 15 | 0.0698 |
| P128/O512 | 10.025 | 135.7 | 512 | 20.85 / 38.46 | 5.46 / 6.36 | 9.95 | 0 | 52 | 0.1585 |

- **P 从 135 → 518（×3.83）**：TTFT P50 +11.04 ms（+68.1%）、P99 +13.77（+45.6%）；TPOT P50 只 +0.52 ms（+11.5%）。prefill token 只进入“首 token 之前”的 step，之后转普通 decode，所以主要打在 TTFT。
- **O 从 128 → 512（×4）**：TPOT P50 +0.93 ms（+20.5%）、P99 +1.29（+25.4%）；TTFT 也 +4.65（+28.7%），因为 in-flight 从 6.1 抬到 28.2，后来者的 prefill 也进入更忙的 batch。
- **配对**（baseline vs O512，615 条同 prompt/到达）：ΔTPOT 中位 **+0.97 ms**，**615/615 全变慢**；ΔTTFT 中位 +4.96 ms。TPOT 是系统性的，不是少数长尾。
- **TPOT 与 in-flight 单调**（P50）：control 4.71→4.79→4.93→5.12（5–10/10–20/20–30/30–40），mixed 5.07→5.26→5.43→5.48，decode 5.03→5.07→5.33→5.55→5.85。V1 每个 decode 每步 1 token，TPOT 近似 step 时间。
- 三点 waiting 全 0、preemptions Δ0、KV ≤16%；goodput 与 injected 的 0.5–2% 差异是窗口结束时在途请求（baseline 60 s 有 12/615 落在窗口后）。

## 3. Short-class interference comparison

| seed | Case | Short 注入 | Long 注入 | Short TTFT P50/P99 | Short TPOT P50/P99 | Short goodput | Overall goodput | Running 峰值 | KV 峰值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2027 | Short-only | 19.8991 | 0 | 16.77 / 31.82 | 4.79 / 5.27 | 19.8713 | 19.8713 | 29 | 0.0464 |
| 2027 | Short + Long | 19.9036 | 4.9471 | **18.23 / 38.29** | **5.33 / 6.15** | **19.8951** | **24.8362** | 35 | 0.0897 |
| 2028 | Short-only | 19.8056 | 0 | 16.46 / 31.75 | 4.97 / 5.69 | 19.7389 | 19.7389 | 28 | 0.0433 |
| 2028 | Short + Long | 19.8056 | 5.3111 | **18.32 / 37.10** | **5.41 / 6.08** | **19.7444** | **25.0333** | 36 | 0.0715 |

- **同流验证**：2027 的 11,442 对、2028 的 3,565 对，`max abs(Δplanned_t)=0`，`prompt_tokens_est` mismatch 0；2027 mixed 的 short 总数 23,327，前 11,442 条与 control 对齐。内容一致性由确定性 `RandomDataset` + 同 seed 推断（records 未存 prompt）。
- **配对差**（mixed − control）：2027 ΔTTFT 中位 **+1.62 ms**（p95 +13.98，65.3% 变慢）、ΔTPOT **+0.417 ms**（p95 +0.91，**96.0%** 变慢）；2028 +1.85 / +0.41，65.8% / 90.6%。ΔE2E 中位 ≈128×ΔTPOT，说明变慢主要来自 decode。
- **长 prefill 重叠**：13.48% 的 short 落在 long 首 token 前的窗口内，TTFT 中位 22.92 vs 18.39；配对 ΔTTFT +5.92 vs 全体 +1.62；≥2 窗口（1.05%）时 +9.11 ms。说明 prefill 确实会推迟 short 首 token，但量级只有几 ms。
- **同 in-flight 残差**：10–20 档 control 16.82/4.79 vs mixed 18.77/5.26；20–30 档 18.24/4.93 vs 20.60/5.43，即 +2 ms TTFT、+0.5 ms TPOT。in-flight 是内生的，不能当随机化控制。
- **TTFT 先坏还是 TPOT 先坏**：没有时间上的先后；P50 上 TPOT +11.2% 略大于 TTFT +8.7%，P99 上 TTFT +20.4% 大于 TPOT +16.7%；配对层面 TPOT 更一致。形状轴的分工更清楚：P512 主要打 TTFT（+68%），O512 主要打 TPOT（+20.5%）。
- **waiting 时序**：9,697 个样本 waiting>0 次数为 0，但 short 已变慢 → 本轮 waiting 不是前兆指标；变慢发生在“无 admission 排队”区间，属于 step-level 效应。只有 running 触到 `max_num_seqs`、budget 不足或 KV 触发抢占时，waiting 才会成为主信号（Day 2 的 100 req/s 是那个区间）。
- **4 条越线请求**：long_prefill id 335/336/337 TTFT 226.6/201.6/232.5 ms，紧随其后的 id 338–340 `dispatch_lag` 197.7/149.6/67.0 ms；control_2028 id 11 TTFT 264.3 ms，id 12/13 lag 204.0/118.7 ms，且 metrics 间隔 0.516→1.012 s = **0.495 s**。这些请求 E2E 正常、TPOT 甚至偏低（3.5–4.8 ms），形态符合“客户端事件循环停顿约 0.2–0.25 s，首事件被延迟处理”。不影响 P99，但把 joint cohort 拉到 0.99775/0.99972。
- **SLO 余量**：mixed TTFT P99 41.96 ≈ **4.8×**，TPOT P99 6.15 ≈ **2.4×**；所有点 P99 余量 ≥2.3×。这是“远离饱和的机制实验”，不是逼近 SLO 的容量实验。

## 4. Counter-hypotheses

- **总负载 vs prefill 特异性**：mixed 同时 +25% 请求、+25% output token、**+95% prefill token**，单看 mixed 无法分离。形状轴给出分解：P512 的 ΔTTFT:ΔTPOT ≈ **21:1**，O512 ≈ **5:1**，说明 prefill 主要影响 TTFT、decode 量主要影响 TPOT；按 in-flight 分层后仍有 +2 ms/+0.5 ms 残差，与“长 prefill 进入同一步”相容，但不能定量归因。
- **客户端 CPU/事件循环**：`dispatch_lag` P99 1.29→2.38 ms，正常；两次 ~200 ms 停顿足以解释全部 4 条 max 违约，但不足以解释 96% 请求上的 +0.42 ms TPOT 中位偏移。`sse_choice_event_gap_p95` +56.9% 远大于 TPOT +11.2%，说明客户端事件处理确实更忙，但该指标不是 token ITL。
- **Cache / Graph warmup**：每个 run 只做 1 条 shape-specific warmup；`server.log` 无重启记录，推断 compile/graph 已热；首条测量请求 TTFT 11–31 ms，无 Day 2 的 2–3 s 冷启动。运行顺序固定（baseline→…→mixed），后跑的 mixed 若受热缓存影响应更快，实际更慢，混杂方向保守。prefix cache 未清空，long prompt 可能挤掉 short 前缀块，方向不确定。
- **长度 / token / SSE 映射**：P_est 与服务端 `prompt_tokens_total` 差分相差 <0.1%；output token 精确为 128/512，gen counter 差 ≤30 token。TPOT = `(e2e−ttft)/(usage_tokens−1)`，e2e 含 usage 尾包；SSE 事件不保证 1 token，`sse_event_gap_p95` 是“每请求 P95 的跨请求 P95”，不是 ITL。名义 P128/P512 实际 P50 = **136/520**（chat template +8），且 <2% 低尾（min 37–41 / 345–353）。
- **KV / `max_num_seqs`**：KV 峰值 0.158、preempt Δ0、running 峰值 52；没有观察到 waiting，故没有证据表明 `max_num_seqs` 是瓶颈，但也不能排除它恰好等于 52。
- **统计功效**：主实验 single seed；duration 60–1172 s 不等，baseline 615 条 P99 很弱（最差 ~7 条），29k 条的 mixed 稳定得多；固定 128/512、T=0，不可外推。Poisson 有限窗使 baseline 实际 10.25 vs 10（+2.5%），是抽样波动。

## 5. Optional Chunked Prefill A/B

**未执行**：没有 512-budget 产物，`server.log` 也没有第二次 `vllm serve`。

- 预算核算：最坏一步 = 52 decode + 520 prompt = **572**；2048 budget 下只占 **27%**，520-token prompt 可一次装下；512 budget 下 **520 本身已 >512**，任何 520-token prompt 都必然拆成 ≥2 块（baseline 的 151 token 仍可一次放下）。
- 证明 chunking 需要 scheduler 证据：`num_scheduled_tokens[req]` 小于剩余 prompt；`num_computed_tokens` 两步只前进一部分；首个 chunk 后无 sampled token；`query_start_loc`/`max_query_len` 显示 1-token decode 与 >1-token prefill 共存。
- 若 budget 512 后“变好”，不能直接归因于 chunking 更好：更小 budget 单步更快但步数更多；chunk 占用更多 step 边界，其他 prefill 可能等待；TTFT 是各步总和。需要 AB/BA + 多 seed + step trace。

## 6. Conclusion

**Observed**：7 run/53,407 请求全 200、0 失败/drop、preempt Δ0；shape 三点 TTFT/TPOT、goodput、running/KV 如上；615/615 条 O512 的 TPOT 全部变慢（中位 +0.97 ms）；干扰轴 short TTFT/TPOT +8.7%/+11.2%（P50），goodput 不降，joint SLO 1.0；2028 复现；4 条 TTFT>200 ms 与客户端停顿重合；9,697 个 metrics 样本 waiting 全 0。

**Inferred**：长 prefill 的干扰是“步内共享”而非队列阻塞（waiting=0、KV≤16%）；TPOT 主要由 running 数决定（输出 tok/s ÷ in-flight ≈ 1/TPOT）；2048 budget 下没有 chunking 的直接证据；4 条违约最可能是客户端伪影；short 的 +1.6 ms/+0.42 ms 是真实但很小的系统偏移。

**Unknown / follow-up**：server CLI、`max_num_seqs` 实际值、prefix cache 是否清空、graph 状态；per-step `num_scheduled_tokens`/`num_computed_tokens`；GPU 利用率/MFU 与 step 时间分解（Day 5）；客户端停顿根因；形状轴第二 seed、更长窗口、更多长度；>50 req/s 的 admission knee 与 512-budget A/B。
