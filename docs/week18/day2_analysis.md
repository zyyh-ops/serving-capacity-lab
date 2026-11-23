# Week 18 Day 2 — Capacity Sweep Analysis

## 0. 结论先行

- **候选容量区间：80–90 req/s**。
  - **80 req/s 是已证实的 SLO-safe 点**：240 s、19,082 条请求全部完成，窗口内 18,998 条，goodput = completed = **79.16 req/s**，P99 TTFT **73.2 ms**、P99 TPOT **10.24 ms**，waiting 全程 0。
  - **90 req/s 是边界点**：240 s、21,545 条，完成 89.17 req/s，goodput **86.38 req/s**；但 P99 TPOT **15.39 ms** 已越过 15 ms SLO，663 条（**3.1%**）TPOT>15 ms，因此只能算“吞吐边界/候选”，不能算严格全达标。
  - **100 req/s 已过载**：30 s 内 waiting 从 17.3 s 起持续增长到峰值 110，running 封顶 256，goodput 崩塌到 **18.03 req/s**，P99 TPOT 20.9 ms，P99 TTFT 1135 ms（窗口内 747 ms）。
- **120 req/s 不是有效的 open-loop 点**：403 条请求被 `max_inflight_guard=512` 拦截，实际只 dispatch **107.97 req/s**；它只能说明“系统在约 91 req/s 完成率附近封顶 + 客户端保护触发”，不能当作 120 req/s 的注入结果。
- 观测到的最高**完成吞吐**约 **91 req/s**、最高**输出 token 吞吐**约 **11.65k tok/s**；最高 **goodput** 出现在 90 req/s（86.38 req/s）。三者不是同一个“容量”。

## 1. Measurement Contract

- Arrival window：`[t0, t0+T]`；Pilot `T=30 s`，Confirm `T=240 s`。Drain：到达窗口结束后继续等请求完成，最长 90 s。
- `offered_realized = scheduled / T`；`injected_proxy = count(dispatch_t < T) / T`。前者是计划，后者才是客户端实际发出的代理指标。
- `achieved_window = count(success and end_t < T) / T`；`goodput_window = count(success and TTFT<=200 ms and TPOT<=15 ms and end_t < T) / T`。
- `completed_after_drain`（探针字段 `success_total_after_drain`）是**最终成功总数**，不是“额外吞吐”；`summary.md` 的 completed/goodput 只算窗口内，二者不能混用。
- 尾延迟有两套：
  - `ttft_p99_ms` / `tpot_p99_ms`：对**所有最终成功请求**统计，可能包含 Drain 阶段；
  - `ttft_p99_in_window_ms` / `tpot_p99_in_window_ms`：只对 `end_t < T` 的成功请求统计。
  - 100 req/s 点两者相差很大（1135 ms vs 747 ms），**不能把含 Drain 的 P99 标成稳态 P99**。
- `client_guard_drop` 是**未发送**的请求，不是服务端拒绝；120 点有 403 条，故该点的 offered 不等于 injected。
- `dispatch_t` 记录在 `session.post()` 之前，不是 TCP 发包时刻；包含事件循环调度与 aiohttp 请求对象创建开销，不含内核发送缓冲和 frontend 排队。
- Pilot 是**发现性实验**，只用于定位拐点；Confirm 只覆盖 80/90 两个点、单个 seed，仍不足以宣称“正式最大容量”。

## 2. Raw Data

### 2.1 负载与吞吐

| 点 | T (s) | 目标 req/s | 实际注入 req/s | 窗口完成 req/s | Goodput req/s | 窗口完成/注入 | Goodput/完成 | 总成功（含 Drain） | 窗口后完成 | Client drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rate_20 | 30 | 20 | 19.67 | 19.37 | 17.20 | 98.5% | 88.8% | 590 | 9 | 0 |
| rate_40 | 30 | 40 | 39.93 | 39.40 | 39.27 | 98.7% | 99.7% | 1,198 | 16 | 0 |
| rate_60 | 30 | 60 | 60.53 | 58.77 | 58.70 | 97.1% | 99.9% | 1,816 | 53 | 0 |
| rate_80 | 30 | 80 | 80.73 | 77.43 | 77.30 | 95.9% | 99.8% | 2,422 | 99 | 0 |
| rate_100 | 30 | 100 | 101.97 | 89.83 | 18.03 | 88.1% | 20.1% | 3,059 | 364 | 0 |
| rate_120 | 30 | 120 | 107.97 | 91.00 | 5.27 | 84.3% | 5.8% | 3,239 | 509 | **403** |
| confirm_80 | 240 | 80 | 79.51 | 79.16 | 79.16 | 99.6% | 100% | 19,082 | 84 | 0 |
| confirm_90 | 240 | 90 | 89.77 | 89.17 | 86.38 | 99.3% | 96.9% | 21,545 | 145 | 0 |

> 数据源：[summary.csv](results/week18_day2/summary.csv)、[summary.md](results/week18_day2/summary.md)、[confirm_80](results/week18_day2/confirm_80_seed2027/summary.json)、[confirm_90](results/week18_day2/confirm_90_seed2027/summary.json)。`summary.md` 不包含 confirm 行，上表由 confirm 的 `summary.json` 补入。

### 2.2 延迟、队列与 KV

| 点 | P99 TTFT 全部 / 窗口内 (ms) | P50 / P95 / P99 TPOT (ms) | Waiting 首季度→末季度（峰值） | Running 峰值 | KV 峰值 | 输出 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| rate_20 | 3145 / 3149 | 4.69 / 6.73 / 6.88 | 0.13 → 0（2） | 78 | 0.146 | 2,478.9 |
| rate_40 | 46.8 / 46.8 | 4.96 / 5.15 / 5.23 | 0 → 0（0） | 39 | 0.062 | 5,043.2 |
| rate_60 | 67.7 / 68.2 | 5.52 / 6.23 / 6.43 | 0 → 0（0） | 73 | 0.116 | 7,522.1 |
| rate_80 | 83.9 / 84.5 | 7.65 / 9.35 / 9.68 | 0 → 0（0） | 119 | 0.184 | 9,911.5 |
| rate_100 | 1135 / 747 | 18.87 / 20.76 / 20.87 | 0 → 66.5（110） | **256** | 0.409 | 11,498.7 |
| rate_120 | 2675 / 2676 | 20.26 / 20.57 / 20.68 | 20.3 → 245.4（256） | **256** | 0.419 | 11,648.0 |
| confirm_80 | 73.3 / 73.2 | 7.77 / 9.59 / 10.24 | 0 → 0（0） | 125 | 0.198 | 10,132.3 |
| confirm_90 | 102.6 / 102.7 | 10.78 / 14.72 / 15.39 | 0 → 0（0） | 202 | 0.316 | 11,413.3 |

> 所有点 `preemptions` 差分均为 0；`rate_20` 有 1 次 metrics scrape error，其余为 0。P50 TPOT 由逐请求 `request_records.json` 复算，与 `summary.csv` 的 P95/P99 一致。

## 3. Queue and Throughput

### 3.1 注入率是否可信

- 20–100 的 30 s 点，实际注入与目标偏差最大约 2%（100 点 101.97），可以作为 open-loop 近似。
- 120 点计划 121.4、实际只发出 107.97；403 条从未进入 HTTP。原因不是服务端拒绝（所有 attempted 都是 200），而是客户端 `pending >= 512` 的保护逻辑。**该点不能用“120 req/s 下系统完成 91 req/s”来表述**，只能说“在 512 在途保护下，实际注入约 108 req/s 时完成 91 req/s”。
- 120 点的 waiting 峰值 256、running 峰值 256，合计约 512，与 guard 上限吻合；说明 guard 与服务端队列在同一个在途池里相互挤压，不能把 drop 单独归因于客户端 CPU 或服务端。
- `outstanding_at_arrival_end`（表中“窗口后完成”列）= 9/16/53/99/364/509/84/145；因为所有已发请求最终都成功，它恰好等于窗口后完成数。120 点的 509 已贴着 guard 512，说明该点测到的是“被客户端保护截断的系统”，不是完整 offered load。

### 3.2 完成率与吞吐平台

- 80 → 90（240 s）：完成率 79.16 → 89.17（+12.6%），输出吞吐 10,132 → 11,413 tok/s（+12.6%）；goodput 79.16 → 86.38（+9.1%）。这一段仍近似线性，但 TPOT 已从 P99 10.24 ms 升到 15.39 ms，**延迟余量基本用完**。
- 90 → 100：注入 89.77 → 101.97（+13.6%），完成仅 89.17 → 89.83（+0.7%），输出 11,413 → 11,499（+0.8%）；goodput 反而 86.38 → 18.03。**这是典型的“吞吐平台 + 延迟崩塌”**。
- 100 → 120：实际注入 101.97 → 107.97（+5.9%），完成 89.83 → 91.00（+1.3%），输出 11,499 → 11,648（+1.3%）；goodput 18.03 → 5.27。最高完成吞吐约 **91 req/s**，最高输出吞吐约 **11.65k tok/s**，但都伴随队列和/或 guard drop。
- 因此“最大吞吐”和“最大 goodput”不是同一个点：
  - 最大输出吞吐：120 点（11,648 tok/s，但 403 drops，goodput 5.27）；
  - 最大完成吞吐：120 点（91.0 req/s，但输入被截断）；
  - 最大 goodput：90 点（86.38 req/s，仍有 3.1% TPOT 违约）；
  - 严格 SLO-safe 最大点：80 点（79.16 req/s，零违约）。

### 3.3 队列增长

- 80/90 的 240 s：waiting 全程 0（0.25 s 采样），running 峰值 125/202，KV 峰值 0.198/0.316；没有观察到 scheduler 排队。
- 100：waiting 在 **17.3 s** 首次 >0，**21.7 s** running 首次到 256，峰值 110（29.5 s），末季度均值 66.5，最后 10 s 斜率约 **+9.05/s**。这不是一次尖峰，而是持续增长。
- 120：waiting 在 **4.4 s** 首次 >0，**5.4 s** running 首次到 256，峰值 256（21.3 s），末季度均值 245.4；队列早已顶满，guard 又在客户端截断注入。
- 100/120 的 `running` 恰好封顶 **256**，且 waiting 在 running 接近/达到 256 后开始增长；这与 [scheduler.py:334–337](vllm/v1/core/sched/scheduler.py#L334-L337) 的准入条件一致：`len(self.running) == max_num_running_reqs` 时停止从 waiting 取新请求。由于没有 server 启动日志，只能**推断**本 server 的 `max_num_seqs=256`（或等价上限），不能当作已确认配置。
- KV 峰值只有 0.409/0.419，且 preemptions=0；**没有证据支持“KV 容量不足”**。队列的直接原因是 running 达到并发上限 + 吞吐平台，而不是 KV 池耗尽。

### 3.4 尾延迟与 SLO 违约分解

对“最终成功”的请求逐条判 SLO，得到：

| 点 | TTFT>200 ms | TPOT>15 ms | 两者同时 | 违约并集 | 达标数/总数 |
|---|---:|---:|---:|---:|---:|
| rate_20 | 65 | 0 | 0 | 65 | 525/590 |
| rate_40 | 4 | 0 | 0 | 4 | 1,194/1,198 |
| rate_60 | 2 | 0 | 0 | 2 | 1,814/1,816 |
| rate_80 | 4 | 0 | 0 | 4 | 2,418/2,422 |
| rate_100 | 1,327 | 2,420 | 1,229 | 2,518 | 541/3,059 |
| rate_120 | 2,720 | 2,981 | 2,620 | 3,081 | 158/3,239 |
| confirm_80 | 0 | 0 | 0 | 0 | 19,082/19,082 |
| confirm_90 | 5 | 663 | 0 | 668 | 20,877/21,545 |

关键读数：

- **90 点不是 TTFT 问题，而是 TPOT 问题**：TTFT P99 只有 102.6 ms，但 TPOT P99 15.39 ms；P95 TPOT 已到 14.72 ms，分布整体贴着 SLO 边缘。`goodput/completed = 96.9%`，说明约 3.1% 请求因 decode 速度不达标而被 goodput 剔除。
- **100 点 TPOT 中位数已超 SLO**：P50 18.87 ms > 15 ms，P99 20.87 ms；同时 TTFT P95 约 1060 ms。goodput 只剩 18.03，不是“少量长尾”，而是大多数请求都不达标。
- **120 点 92% 请求 TPOT>15 ms**，P50 TTFT 2.35 s；系统已进入“高并发 decode + 长队列”状态，且输入被 guard 截断。
- 20 点的 65 个 TTFT 违约几乎全部来自前 5 s 冷启动（见 §3.5），不是稳定负载下的容量信号。

### 3.5 窗口/Drain 偏差与 20 点冷启动

- 30 s 80 点：窗口内完成 2,323/2,422（95.9%），99 条在窗口后完成；240 s 80 点：窗口内完成 18,998/19,082（99.6%），84 条在窗口后完成。**30 s 的 77.43 被窗口边界系统性压低**；240 s 的 79.16 更接近持续完成率。
- 100 点窗口后完成 364 条、120 点 509 条；总墙钟只比 30 s 多 2.37 s / 3.67 s。这说明这些请求大多已接近完成，Drain 很快结束；**不能把 364/509 当成额外稳定吞吐**。其余 30 s 点的 Drain 为 0.41/0.59/0.62/0.75 s（20/40/60/80），240 s confirm 为 0.73/0.91 s（80/90）。
- 20 点是 sweep 首轮：前 1 s 内到达的 21 条请求 TTFT 约 2.9–3.35 s，1–5 s 的 P50 约 421 ms，5 s 之后 P50 回到 16–17 ms；65/590 条 TTFT>200 ms。与此同时 waiting 峰值只有 2、KV 峰值 0.146、running 峰值 78——**不是服务端队列**，更像是首次 graph capture/compile 或 engine warmup 造成的引擎级停顿。由于没有 server log，具体原因仍是推断；但可以确定：**20 点的 17.2 goodput 不能用于容量曲线**，应视为 warmup 未排除的失败点。

## 4. Capacity Knee Hypothesis

### 4.1 拐点位置

按三种“容量”定义分别报告：

1. **严格 SLO-safe 容量**（所有成功请求都满足 TTFT≤200 ms 且 TPOT≤15 ms）：已测最大点是 **80 req/s**；90 点有 3.1% TPOT 违约，不能算全达标。
2. **Goodput 最大点**：**90 req/s**，86.38 req/s；但它是“用 3.1% 违约换来的更高 goodput”，不是严格容量。
3. **服务吞吐平台**：约 **90–91 req/s 完成 / 11.4–11.65k output tok/s**；120 点在 guard 截断下也只到 91.0，说明再往上加注入不会显著提高完成率。

因此，**候选容量区间取 [80, 90] req/s**：

- 80 是**已验证下界**（240 s、零违约、零队列）；
- 90 是**边界候选**（240 s、零队列、TPOT P99 刚越过 SLO、3.1% 违约）；
- 100 是**已越过的过载点**（持续队列、TPOT 中位数超 SLO、goodput 崩塌）。

如果只能报告一个保守数字，应报 **80 req/s**；如果报告区间，应写 **80–90 req/s，90 为边界点，不能把 90 当作零违约容量**。

### 4.2 机制解释（推断）

- 80 → 90：running 中位数从约 78 升到约 123（末季度均值 84 → 138），batch 更大，输出吞吐提高 12.6%，但单步 decode 变慢，TPOT P99 从 10.24 升到 15.39。**这是 batching 的吞吐/延迟权衡到达 SLO 边界**，不是 KV 不足。
- 90 → 100：running 达到 256 的准入上限，waiting 开始持续增长；batch 不能再扩大，吞吐只提高 0.8%，多出来的负载全部变成排队和 TTFT。**这是并发上限 + 吞吐平台的硬饱和**。
- KV 峰值 0.42、preemptions=0，说明 KV 池不是本轮限制；`max_num_seqs=256` 是更直接的 running 上限（待 server log 确认）。
- 因此“拐点”其实是两段：**80–90 的 SLO/decode 拐点** + **100 的并发/队列拐点**。把两者混成一个“GPU 最大容量”会过度解释。

### 4.3 仍存在的替代解释

- **Client bottleneck**：120 点确有 403 drops、dispatch lag P99 13.1 ms；但 100 点没有 drop，waiting 仍持续增长，说明 100 的队列至少在服务端 scheduler。90 点实际注入 89.77、waiting 0，也不支持“client 发不出去”。要完全排除 client，需要多 client 分片、提高 guard、记录 client CPU/事件循环延迟。
- **`max_num_seqs` 结构上限**：100/120 的 running 恰好 256，很像配置上限而非 GPU 算力极限；若把 `max-num-seqs` 调到 512，100 点的队列形态可能改变。需要用不同 `max-num-seqs` 复测，才能把“配置上限”和“硬件吞吐”分开。
- **Prefix cache / warmup**：Day 2 没有关闭 prefix cache 的证据；80 confirm 在 90 之前跑，90 可能受益于已热的公共前缀。prefix cache 主要影响 TTFT，不太可能解释 90 的 TPOT 边界，但仍是一个未控变量。
- **窗口长度与尾态**：100 只有 30 s，队列从 17.3 s 才开始增长；30 s 可能还没到真正的稳态排队。需要用 180–300 s 的 100 点确认队列斜率是否持续。
- **采样混叠**：waiting peak 是 0.25 s 离散采样，真实峰值可能更高；`waiting=0` 也不能证明亚秒级从未排队。
- **单 seed / 固定长度**：90 只有 seed 2027 一轮；80 有 seed 2026（30 s）和 2027（240 s），一致性较好。输入/输出固定 128/128，换长度后 token 吞吐与 TPOT 都会变，容量不可直接外推。
- **P99 估计不确定性**：30 s 点只有 590–3,642 请求（P99 由最差 6–36 条决定）；240 s 点 19–21k 请求更稳，但仍为单轮。90 的 TPOT P99 15.39 与 SLO 15 只差 0.39 ms，处在测量噪声可能影响的边缘，**这也是不能把 90 直接判成“稳定达标”或“稳定不达标”的原因**。

### 4.4 需要重复的实验

- **80/85/90/95/100** 各 3 seeds × **300 s**（或 ≥5 min），保持 Poisson、无 `max-concurrency`；把 client guard 提高到 1024–2048，确保 120 点不再 drop（或明确标记为 capped）。
- 每轮前显式 warmup 并丢弃前 10–15 s；报告“稳态窗口”与“含 Drain cohort”两套数字，初始/终端瞬态分开。
- 重启 server 或至少清 prefix cache；记录完整启动 CLI、`max-num-seqs`、`max-model-len`、graph/compile 状态。
- 采 `/metrics` 0.2 s 或 Prometheus；记录 client CPU、dispatch lag、实际 inflight、TCP 发送时间（若可得）；补 GPU 利用率/MFU。
- 做 `max-num-seqs=128/256/512` 的对照，验证 100/120 的 256 封顶是否是配置上限；做 100 的 240 s 重复，确认 waiting 斜率是否持续。
- 对 90 的 TPOT P99 给 bootstrap 置信区间；如果 15.39 ms 的区间跨过 15 ms，就不能把 90 判为“稳定超 SLO”或“稳定达标”。

## 5. Evidence Strength

### 5.1 Observed（直接观测）

- 8 个负载点的窗口统计、逐请求记录与 0.25 s metrics 采样均已落盘：[summary.csv](results/week18_day2/summary.csv)、各点 `request_records.json` / `metrics.jsonl`。
- 80 confirm（240 s）：19,082 条全部成功，窗口内 18,998 条，goodput **79.16 req/s**，P99 TTFT **73.2 ms**、P99 TPOT **10.24 ms**，waiting 0，KV 峰值 0.198。
- 90 confirm（240 s）：21,545 条全部成功，窗口内 21,400 条，goodput **86.38 req/s**，P99 TTFT **102.6 ms**、P99 TPOT **15.39 ms**；663 条 TPOT>15 ms（3.1%），waiting 0，KV 峰值 0.316。
- 100（30 s）：窗口完成 89.83 req/s、goodput **18.03**；waiting 峰值 110、running 256、KV 0.409；P99 TPOT 20.87 ms（P50 已 18.87 ms）；P99 TTFT 1135 ms（窗口内 747 ms）。
- 120（30 s）：403 次 client guard drop；实际注入 107.97、完成 91.0、goodput **5.27**；waiting 峰值 256、running 256、KV 0.419。
- 所有点 HTTP 失败/超时为 0，preemptions 差分为 0；`rate_20` 有 1 次 metrics scrape error。
- 20 点前 5 s 是冷启动长尾：65 条 TTFT>200 ms 中约 63 条集中在前 5 s；5 s 后 P50 TTFT 回到 16–17 ms。

### 5.2 Inferred（有证据但未独立证实）

- **80–90 req/s 是 SLO 拐点区间**；80 是已验证的安全点，90 是边界候选，100 已过载。
- **服务吞吐平台约 90–91 req/s / 11.4–11.65k output tok/s**；再增加注入主要增加队列和延迟，而不是完成率。
- **100/120 的硬队列与 `max_num_seqs=256` 的准入上限一致**；KV 仅约 42% 且无抢占，**不是 KV 容量瓶颈**。源码 [scheduler.py:334–337](vllm/v1/core/sched/scheduler.py#L334-L337) 支持“running 满后 waiting 才增长”的机制，但 server 的 `max_num_seqs` 实际值没有日志确认。
- **90 的 TPOT 边界来自更大 batch 的 decode 变慢**，而不是 TTFT 队列；90 的 TTFT P99 只有 102.6 ms，waiting 全程 0。
- **20 点的低 goodput 是 warmup/冷启动伪影**，不能作为 20 req/s 的容量证据。
- **120 点不是有效 open-loop**；其完成率 91.0 只能作为“capped client 下的吞吐上限”参考。

### 5.3 Unknown（不能下结论）

- 90 req/s 能否在 >240 s、多 seed 下持续；85/95 等中间点完全未测。
- GPU 利用率/MFU；瓶颈是 compute、batch 调度、client 还是 SSE 处理；没有 profile，不能归因。
- 120 req/s 的真实 open-loop 行为；当前被 guard 截断。
- P99 的置信区间；90 的 15.39 ms 与 SLO 15 ms 只差 0.39 ms，单轮数据不足以判定长期稳定性。
- 30 s 100 点是否已进入真正稳态排队；需要 180–300 s 重复确认。

## 6. Day 3 Handoff

- **Chosen non-saturated reference workload**：**80 req/s，240 s，seed 2027**。理由：19,082 条全部成功，窗口内 18,998 条，goodput = completed = 79.16 req/s，P99 TTFT 73.2 ms、P99 TPOT 10.24 ms，waiting 全程 0，KV 0.198；这是本轮唯一“零 SLO 违约 + 零队列 + 长窗口”的点。
- **Chosen near-knee workload**：**90 req/s，240 s，seed 2027**。理由：完成 89.17、goodput 86.38，P99 TTFT 102.6 ms、P99 TPOT 15.39 ms，waiting 0，KV 0.316；它给出最高 goodput，但 3.1% 请求 TPOT>15 ms，适合作为“边界/压力”对照，**不适合作为“零违约基线”**。
- **Known limitations**：
  - 90 只有单 seed、单轮；80 有 30 s 与 240 s 两个窗口，但仍非多 seed；
  - 100 只有 30 s，队列从 17.3 s 才开始增长，稳态性未确认；
  - 120 被 guard 截断，不能作为 120 req/s 的 open-loop 证据；
  - 固定 128/128、温度 0、`ignore_eos`，容量不可直接外推到其他长度；
  - 没有 GPU 利用率/MFU 与 client CPU 数据，无法把“配置上限”和“硬件吞吐”彻底分开。
- **Day 3 建议**：以 80 作为安全基线、90 作为近拐点压力点；
