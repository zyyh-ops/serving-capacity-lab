# Week 18 Day 2 — Open-loop Source Trace

## 0. 结论先行

- 探针的 `planned_t` 是 Poisson 到达计划，`dispatch_t` 是协程真正开始执行、即将调用 `session.post()` 的时刻，**不是 TCP 发包时刻**；`end_t` 是收到 `[DONE]`/流结束后的客户端时刻。
- `summary.md` 的 completed/goodput 只统计 `end_t < T` 的窗口内完成；`ttft_p99_ms` 统计“最终成功”的全部请求（可能含 Drain），`ttft_p99_in_window_ms` 才是窗口内完成请求的 P99。两者在 100 req/s 点相差 1135 ms vs 747 ms，不能混用。
- 120 req/s 点触发 403 次 `client_guard_drop`，实际只 dispatch 107.97 req/s，因此它不是有效的 120 req/s open-loop 点。

## 1. Environment & Evidence

- vLLM：**0.10.2**（V1）；Python **3.12.16**、torch **2.8.0+cu128**、CUDA **12.8**、driver **570.124.06**、**RTX 4090 24 GB**。

- Model / tokenizer：`Qwen/Qwen3-0.6B` 与同模型 tokenizer；`temperature=0.0`、`ignore_eos=true`、`max_completion_tokens=128`、`stream=true`、`stream_options.include_usage=true`；输入/输出长度固定 **128/128**（`RandomDataset`，`range_ratio=0`）。

- Server / client：server 监听 `127.0.0.1:8000`（探针 `base_url`），客户端同机 loopback；aiohttp `TCPConnector(limit=0, limit_per_host=0)`；单请求 timeout **90 s**，Drain timeout **90 s**，`max_inflight_guard=512`，metrics 采样 **0.25 s**。client 的 fd limit 未记录。

- Server startup CLI：
  ```bash
  # Day 1 记录；Day 2 没有 server log，不能当作已确认配置
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve Qwen/Qwen3-0.6B \
    --dtype auto --max-model-len 2048 --gpu-memory-utilization 0.7
  ```
  
- CUDA Graph / Compilation / KV capacity：记录 KV 可用约 **14.11 GiB / 132,064 tokens / 8,254 blocks**；Day 2 观测到的 KV 峰值为 **0.42**（100/120 req/s 点），即约 42%，**没有接近 KV 容量**。

- 负载点与重复：
  - Pilot sweep：20/40/60/80/100/120 req/s，`duration=30 s`，`seed=2026`；
  - Confirm：80/90 req/s，`duration=240 s`，`seed=2027`；
  - 80 同时有 30 s 和 240 s 两个点，可用于检查窗口/Drain 偏差；90 只有 240 s 点。
  
- 原始产物：[summary.csv](results/week18_day2/summary.csv)、[summary.md](results/week18_day2/summary.md)；confirm 点：[confirm_80 summary](results/week18_day2/confirm_80_seed2027/summary.json)、[confirm_90 summary](results/week18_day2/confirm_90_seed2027/summary.json)。

## 2. vLLM official load generator

### 2.1 `get_request()`：计划到达如何生成

[vllm/benchmarks/serve.py:117–205](vllm/benchmarks/serve.py#L117-L205)：

1. 对每个请求预计算 `current_request_rate`，再取 `theta = 1 / (rate * burstiness)`，从 `np.random.gamma(shape=burstiness, scale=theta)` 采样间隔。
   - `burstiness=1` 时 gamma 退化为 **指数分布**，即 Poisson 到达；
   - `burstiness<1` 更突发，`>1` 更均匀。
2. 对间隔做累加得到 `delay_ts`。当没有 ramp-up 时，用 `target_total_delay_s = total_requests / request_rate` 对整串间隔做**归一化**，把采样总和拉回目标总时长，避免 1–2% 的随机偏差累积到吞吐统计里。
3. `start_ts = time.time()`，逐个请求 `await asyncio.sleep(start_ts + delay_ts[i] - now)` 后 `yield`。
   - 这里只保证“计划 yield 时刻”；`yield` 之后还要经过 `benchmark()` 建 task、事件循环调度、`aiohttp` 连接池，才真正发出 HTTP。
   - 因此官方 bench 的 `request_rate` 是**计划到达率**，不是已证实的 socket 发送率；源码没有为每个请求记录 dispatch 时间戳。

### 2.2 `benchmark()`：计划发送如何变成 HTTP task

[vllm/benchmarks/serve.py:506–578](vllm/benchmarks/serve.py#L506-L578)：

1. `semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None`。
   - `max_concurrency=None` 时**没有客户端信号量**，每个 yield 都直接建 task；
   - 显式设置 `max_concurrency` 时，信号量在**客户端**，请求会在 `session.post()` 之前排队，实际发起率被压到 `C / E2E` 附近，而不是目标 `request_rate`。这正是 Day 1 cap1 的背压位置。
2. `async for request, rate in get_request(...)` 中，每个请求 `asyncio.create_task(limited_request_func(...))`；task 真正开始执行时才会进入 `async_request_openai_chat_completions()`，最终在 `session.post()` 处发出 HTTP。
3. `benchmark_duration` 在 `asyncio.gather(*tasks)` **之后**才取（[:578–583](vllm/benchmarks/serve.py#L578-L583)）。所以官方 `Request throughput` 的分母包含**最后一个请求完成前的 Drain 时间**，不是固定 arrival window；有限请求下它会低于到达窗口内的完成率，不能直接当作稳态吞吐。
4. `endpoint_type`/`backend` 决定 request function：`openai-chat` 对应 `async_request_openai_chat_completions`，URL 必须以 `chat/completions` 结尾；本日探针也走 `/v1/chat/completions`，语义一致。

### 2.3 客户端流式执行：TTFT / ITL / E2E 的边界

[vllm/benchmarks/lib/endpoint_request_func.py:212–327](vllm/benchmarks/lib/endpoint_request_func.py#L212-L327)：

- POST body 用 `messages`、`max_completion_tokens`、`stream=True`、`stream_options.include_usage=True`；`ignore_eos` 可选。
- `StreamedResponseHandler`（[:20–58](vllm/benchmarks/lib/endpoint_request_func.py#L20-L58)）按 `\n\n` 切 SSE，去掉 `data: `，跳过 `:` 注释/ping，把不完整 JSON 留在 buffer；这比“按行读”更稳。
- `st = perf_counter()` 在 `session.post()` 之前；每个非 `[DONE]` data 事件先打 `timestamp`，再 `json.loads`：
  - 第一个带 `choices` 的事件 → **TTFT**，即使 `delta.content` 为空（role/空 delta 也会触发）；
  - 后续带 `choices` 的事件 → 追加 `timestamp - most_recent_timestamp` 到 ITL；
  - `usage` 事件更新 `output_tokens`，同时更新 `most_recent_timestamp`；
  - `latency = most_recent_timestamp - st`，包含 usage 尾包，因此 E2E 可能略大于“最后一个 token”的时刻。
- `calculate_metrics()`（[serve.py:253–346](vllm/benchmarks/serve.py#L253-L346)）只统计 `success`：
  - `TPOT = (latency - ttft) / (output_len - 1)`；`output_len <= 1` 时 TPOT 记 0；
  - ITL 是所有请求的间隔**池化**，不是每请求等权；
  - goodput 逐请求判 `all(SLO >= actual)`（源码 `all(s >= r ...)`），分母同样是含 Drain 的 `dur_s`；
  - 失败请求不进入 completed/goodput 分子，但仍占 `num_prompts`/offered。
- **口径差异**：官方 bench 的 `request_throughput = completed / benchmark_duration`（含 Drain），自定义探针的 `completed_rate_in_window` 用固定 `T`，另给 `completed_after_drain`。因此不能把两套数字直接对齐。

## 3. Custom probe（本轮实际数据来源）

### 3.1 请求生成：Poisson 计划

[scripts/openloop_probe.py:215–229](scripts/openloop_probe.py#L215-L229)：

```python
t = 0.0
while len(schedule) < args.max_requests:
    t += rng.expovariate(args.rate)
    if t >= args.duration:
        break
    schedule.append(t)
```

- 间隔服从指数分布，`planned_t` 为累加值，起点 `t=0`；`t >= duration` 即停止，不补齐到整窗口。
- **没有**官方 bench 的“归一化到 `N/rate`”步骤，所以有限窗口内实际计划数会围绕 `rate × duration` 波动：20 点实际 19.67、100 点 101.97、120 点计划 121.4，都是正常抽样波动。
- 随后用同一 seed 生成 `RandomDataset`，固定 128/128、`range_ratio=0`；请求数与 schedule 严格一致。

### 3.2 客户端执行：从 `planned_t` 到 `dispatch_t`

[scripts/openloop_probe.py:249–267](scripts/openloop_probe.py#L249-L267)：

1. `t0 = loop.time()`，启动 metrics 采集 task，然后按 `schedule` 逐个 `await asyncio.sleep(max(0, t0 + when - loop.time()))`。
2. 到点后先检查 `len(pending) >= max_inflight`：
   - 若触发，记录 `status='client_guard_drop'`、`dispatch_t=None`，**不发送**；
   - 否则 `asyncio.create_task(send_one(...))`，把 task 放进 `pending`，完成回调负责移除。
3. `send_one()` 的第一行 `started = time.monotonic()` 才是 `dispatch_t`；随后才进入 `session.post()`。

因此：

- `dispatch_t - planned_t` 包含**事件循环调度延迟**和 `aiohttp` 连接/请求对象创建开销，但**不包含** TCP 三次握手、TLS、内核发送缓冲、服务端 frontend 排队；
- loopback 下这部分通常很小（本轮 P99 1.7–13.1 ms），但 120 点 P99 13.1 ms 与 guard drop 同时出现，不能把“客户端没瓶颈”说死；
- `max_inflight=512` 是**保护阈值**，不是吞吐参数。120 点有 403 条请求根本没发出，`offered_realized=121.4` 只是计划率，`injected_rate_dispatch_proxy=107.97` 才是实际发出率；此时不能声称完成了全部 offered load。

### 3.3 SSE 解析与每请求指标

[scripts/openloop_probe.py:81–160](scripts/openloop_probe.py#L81-L160)：

- payload：`messages`、`temperature=0.0`、`max_completion_tokens=128`、`ignore_eos=true`、`stream=true`、`include_usage=true`；
- `first_choice`：第一个解析成功且含 `choices` 的 SSE data 事件，`ttft_ms = first_choice - started`；与官方一致，**不要求 content 非空**；
- `tokens`：从 `usage.completion_tokens` 取；`e2e_ms = last_data - started`，`last_data` 是最后一个解析成功的 data 事件（含 usage 尾包）；
- `tpot_ms = (e2e_ms - ttft_ms) / (tokens - 1)`；
- `success` 要求：HTTP 200、至少一个 choices、有 usage completion_tokens、至少一个可解析 data、收到 `[DONE]`；缺一即记 error；
- `end_t` 在 `finally` 中记录，即使异常/取消也有值；`request_records.json` 保留逐请求明细，便于复算。

### 3.4 窗口、Drain 与 guard 的统计口径

[scripts/openloop_probe.py:163–207](scripts/openloop_probe.py#L163-L207)：

- `T = arrival_duration_s`；
- `sent`：非 guard drop 且 `dispatch_t < T`；
- `completed_in`：`success and end_t < T`；
- `completed_after_drain`：`success` 总数（探针 summary 的 `success_total_after_drain`）；
- `goodput_in_window`：窗口内完成且 `TTFT <= 200 ms and TPOT <= 15 ms` 的请求数 / `T`；
- `outstanding_at_arrival_end = len(sent) - len(sent 中 end_t < T)`；
- waiting 的 first/last quarter 分别取 `t < 0.25T` 与 `t >= 0.75T` 的采样均值；peak 是 0.25 s 采样到的最大值。

这解释了为什么 `summary.md` 的 30 s 80 点完成率只有 77.43 req/s，而 240 s confirm 80 点是 79.16 req/s：30 s 窗口结束时还有 99 条在途，它们被算进 `completed_after_drain`，但不进入窗口完成率。**短窗口会系统性低估完成率，不能拿 30 s 的 77.43 当容量。**

### 3.5 `dispatch_t` 与 TCP 真实发包的差异

- `dispatch_t` 记录在 `session.post()` 之前，**不是** socket write 完成时刻；`aiohttp` 的连接复用、DNS（这里没有）、请求头序列化都可能让真实发包晚于 `dispatch_t`。
- 服务端 `/metrics` 的 `arrival_time` 更晚，包含 HTTP/frontend/tokenize 排队；client 的 TTFT 从 `started` 起算，包含这些前置开销，所以 client TTFT 通常大于 server 直方图 TTFT。
- 本实验是 loopback，网络项很小；但“dispatch 率 = 真实到达率”仍只是**代理指标**，不是 TCP 级证据。

## 4. Server metrics（`/metrics` 采集）

### 4.1 采集器

[scripts/openloop_probe.py:63–78](scripts/openloop_probe.py#L63-L78)：

- 每 `metrics_interval`（默认 0.25 s）GET 一次 `/metrics`，单次 timeout 3 s；
- 以 `stop` Event 控制结束；异常写入 `metrics_error`，不中断主负载；
- 采集与负载共用同一个事件循环和 aiohttp session；25 ms 级别的 scrape 开销很小，但高频采样时仍是客户端 CPU 的一部分。

[scripts/openloop_probe.py:34–47](scripts/openloop_probe.py#L34-L47)：

- 用正则匹配 `name{labels} value`，忽略 labels；
- `waiting/running/preemptions/generation_tokens` 对同名样本**求和**，`kv_usage` 取**最大值**；
- 单 engine 时这样等价于全局值；多 engine 时 `waiting/running` 应求和、`kv_usage` 应加权/平均，当前实现的 `max` 会偏高，这是代码的适用边界。

### 4.2 指标语义

[vllm/v1/metrics/loggers.py:188–249](vllm/v1/metrics/loggers.py#L188-L249)：

| 指标 | 类型 / 语义 | 关键限制 |
|---|---|---|
| `vllm:num_requests_running` | Gauge，“Number of requests in model execution batches” | 包含 prefill/decode 的 running 请求；不是 GPU 利用率 |
| `vllm:num_requests_waiting` | Gauge，“Number of requests waiting to be processed” | 只是 scheduler 的 waiting 队列；不含 client 信号量、HTTP/frontend 排队 |
| `vllm:kv_cache_usage_perc` | Gauge，1=100% | 是 KV block 占用比例，不等于 GPU 算力利用率 |
| `vllm:num_preemptions_total` | Counter（源码名 `vllm:num_preemptions`，Prometheus 暴露加 `_total`） | 累计值，要做前后差分；0 只说明本窗口没发生 |
| `vllm:generation_tokens_total` | Counter | 累计生成 token；可做差分，但 scrape 间隔会平滑掉短时波动 |

`scheduler.py:334–337` 的准入条件：

```python
while self.waiting and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break
```

其中 `self.max_num_running_reqs = self.scheduler_config.max_num_seqs`（[:71](vllm/v1/core/sched/scheduler.py#L71)）。因此当 `running` 达到 `max_num_seqs` 时，新请求只能留在 `waiting`，**这是结构性排队，不一定是 KV 不足**。

### 4.3 本轮的采样观察

- 30 s 点每轮约 102–113 个样本（中位间隔 0.27–0.29 s），240 s 点约 869–878 个；`rate_20` 有 **1 次** `metrics_error`，其余为 0。
- 所有点 `preemptions` 差分均为 **0**；KV 峰值从 20 点的 0.146 升到 100/120 点的 0.409/0.419，**远未到 1.0**。
- 80/90 的 240 s 点：`waiting` 全程为 0（采样到 0.25 s 粒度），`running` 峰值分别 125/202；100 点在 **17.3 s** 首次采样到 waiting>0，21.7 s running 首次到 **256**；120 点在 **4.4 s** 首次 waiting>0，5.4 s running 首次到 256。
- `waiting` first-quarter → last-quarter：100 点 **0 → 66.5**，120 点 **20.3 → 245.4**；80/90 点均为 **0 → 0**。这是“持续排队”与“无队列”的最直接区分。
- `waiting=0` 不能证明没有排队：0.25 s 采样会漏掉亚秒级峰值，且不含 client/HTTP 队列；只能说明“采样时刻 scheduler 队列为空”。

## 5. Hypotheses / alternatives

| Hypothesis | Supporting evidence | Alternative explanation | How to rule out |
|---|---|---|---|
| Server saturates at ~90–91 req/s | 100/120 点完成率 89.8/91.0，输出吞吐 11.50/11.65k tok/s 基本平台；TPOT P99 20.9/20.7 ms；100 点 waiting 从 17.3 s 起持续增长；running 封顶 256 | 可能是 batch/调度上限（running 封顶 256）与更大 batch 的 TPOT 代价，而非纯 GPU FLOPs 饱和；KV 仅 0.42、抢占为 0 | 扫 `max-num-seqs=128/256/512`；采 GPU 利用率/MFU；做 240 s 100 req/s 重复，看队列斜率是否持续 |
| Load generator bottleneck at 120 | 403 次 guard drop；dispatch lag P99 13.1 ms；实际只发 107.97 req/s | 服务端排队导致在途数超过 512，guard 是被动保护；不能区分“client 发不出”与“server 处理不完” | 提高/取消 guard（如 2048）、多 client 分片、记录 client CPU/事件循环延迟、对比实际 socket 发送率 |
| Transient queue / burst | 20 点前 5 s TTFT 长尾；40/60/80 点各有 2–4 条 TTFT>200 ms 的孤立尖峰 | 首轮 engine warmup/compile；Poisson 短窗口抖动；单次 GC/调度抖动 | 正式测量前跑 warmup 并丢弃；每点 240–300 s、3 seeds；看队列是否持续而非单点尖峰 |
| KV capacity is the limit | 无；KV 峰值 0.42，`preemptions=0` | 如果真是 KV 不足，应看到 KV 接近 1.0、抢占增长、running 被迫下降；本轮都没有 | 继续加压到 KV 接近 1.0，或显式减小 KV 池；观察 preemption 与 waiting 的关系 |
| Cold start / prefix cache confounds | `rate_20` 是首轮，0–1 s 内 21 条请求 TTFT 约 2.9–3.35 s，1–5 s 的 P50 约 421 ms；之后 P50 回到 16–17 ms | 首次 CUDA Graph capture/compile、engine warmup，或 prefix cache 未热；没有 server log 无法区分 | 重启 server 后先跑显式 warmup；关闭 prefix cache 各跑一遍；记录首请求与 graph capture 日志 |
| Window / drain artifact | 30 s 80 点 completed 77.43，240 s 80 点 79.16；100 点 364 条在窗口后完成 | 短窗口把在途请求算进 `after_drain`，官方 bench 的 duration 也含 Drain | 固定 `T` 统计 + 单独报告 drain；用 240 s 重复；不要用 batch throughput 代替窗口 throughput |

## 6. Conclusions (Observed / Inferred / Unknown)

**Observed**：

- 20–100 req/s 的 30 s 点，实际 dispatch 率与目标偏差 ≤2%；120 点因 403 次 guard drop，实际仅 107.97 req/s。
- 80/90 的 240 s confirm：完成率 79.16/89.17 req/s，goodput 79.16/86.38 req/s；90 点 P99 TTFT 102.6 ms、P99 TPOT **15.39 ms**，663/21,545 条 TPOT>15 ms（3.1%），waiting 全程 0。
- 100 点：完成 89.83、goodput **18.03** req/s；waiting 峰值 110，running 封顶 256，KV 0.409，P99 TTFT 1135 ms（窗口内 747 ms），P99 TPOT 20.9 ms；2420/3059 条 TPOT>15 ms。
- 120 点：完成 91.0、goodput **5.27** req/s；waiting 峰值 256，running 256，KV 0.419；403 条未发送。
- 所有点 HTTP 失败/超时为 0，preemptions 差分为 0；`rate_20` 有 1 次 metrics scrape error。

**Inferred（有数据支持但未独立证实）**：

- SLO 容量拐点在 **80–90 req/s**：80 是已证实的全达标点；90 是 TPOT SLO 边界点（goodput 更高，但 3.1% 请求不达标）；100 已进入持续排队/过载。
- 服务吞吐平台约 **90–91 req/s / 11.4–11.65k output tok/s**；再提高注入不会显著提高完成率，只会增加队列与延迟。
- 100/120 的 hard queue 与 `running=256` 封顶一致，推断本 server 的 `max_num_seqs=256`（或等价准入上限）；KV 仅 0.42 且无抢占，所以**不是 KV 容量瓶颈**。
- `rate_20` 的前 5 s 长尾是**冷启动/预热瞬态**，不是 20 req/s 的稳态容量证据；若把它当容量会严重低估。

**Unknown / cannot conclude**：

- Day 2 server 的完整启动参数、`max-num-seqs` 实际值、prefix cache 是否关闭、CUDA Graph/compile 状态（无 server log）。
- 90 req/s 能否在更长时间（>240 s）、多 seed 下持续满足 SLO；85/95 等中间点未测。
- GPU 利用率/MFU；瓶颈到底是 compute、batch 调度还是 client；需要 profile。
- 120 req/s 的真实 open-loop 行为（当前被 guard 截断）。
- P99 的统计置信区间（30 s 点仅 2.4–3.6k 请求；240 s 点虽 19–21k，但仍为单 seed/单轮）。
- 换输入/输出长度、换 `max-num-seqs`、换 prefix cache 后的容量是否可迁移。
