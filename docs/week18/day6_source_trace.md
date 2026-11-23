# Week 18 Day 6 — Source Trace / Admission Contract

## 0. 结论先行

- **全部拒绝都发生在 socket 之前**：`Bucket.allow(planned_t, cost)` 失败即记 `local_reject` 且 `dispatch_t=None`，请求**从不进 vLLM**（[capacity_probe.py:351-363](scripts/capacity_probe.py#L351-L363)）⇒ 本实验没有一次 HTTP 429，所谓"可用性"是客户端定义的。
- **服务端只有两个硬闸门**：`max_num_seqs`（[scheduler.py:71](vllm/v1/core/sched/scheduler.py#L71)、waiting 循环里的 [scheduler.py:335](vllm/v1/core/sched/scheduler.py#L335)）与 KV 块（[kv_cache_manager.py:270-272](vllm/v1/core/kv_cache_manager.py#L270-L272) → 抢占 [scheduler.py:252-288](vllm/v1/core/sched/scheduler.py#L252-L288)）。本实验 4096 blocks × 16 tok = 65,536 tok，而 128+128 的请求正好 256 tok ⇒ **KV 容量 256 请求 ≡ `max_num_seqs=256`**，两个限制在本次负载下无法分离。
- **过载时两个闸门都没"报警"**：105 req/s 下 running 253、KV 0.799、preemption 0、waiting 峰 19；需求缺口被吸收成 TPOT（5.53 → 16.93 ms p50）。**`vllm:num_requests_waiting` 只数 `len(self.waiting)`**（[scheduler.py:1182-1184](vllm/v1/core/sched/scheduler.py#L1182-L1184)），不数引擎输入队列、API frontend 或客户端 ⇒ `Waiting=0` 不是"没过载"的证据（引擎自己在 06:25:58–06:26:18 同样报 `Running: 205→234, Waiting: 0`）。
- **度量是两套口径**：arrival-cohort（按到达相位取分母，含拒绝）与 completion-window（按完成时刻落在相位内取分母），分别在 [capacity_probe.py:234-264](scripts/capacity_probe.py#L234-L264) 计算，混用会直接把 recovery 的排空读成"容量变高"。

## 1. 环境与证据链

| 项 | 值 | 证据文件 |
|---|---|---|
| vLLM / 引擎 | **0.10.2**、`VLLM_USE_V1=1`、单 engine（`EngineCore_DP0`）、无 Scheduler patch、无 profiler | `servers/admission_256.log:10` 的 `Initializing a V1 LLM engine (v0.10.2)` |
| 硬件 / 精度 | RTX 4090 24 GB、driver 570.124.06、bf16、TP=PP=DP=1 | `contract.txt`、`models.json` |
| 模型 | `Qwen/Qwen3-0.6B`、`max_model_len=2048` | `models.json` |
| 启动 CLI | `--dtype auto --max-model-len 2048 --gpu-memory-utilization 0.7 --max-num-seqs 256 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching --num-gpu-blocks-override 4096` | `servers/admission_256.log.cmd` |
| KV | block 16；默认 **8254 blocks → override 4096**（65,536 tok、32.00x） | 日志前缀 `[kv_cache_utils.py:802]`、`metrics_pre.txt` 的 `cache_config_info` |
| chunked prefill | 开；`max_num_batched_tokens=2048`，`max_num_partial_prefills=1` 默认 | 日志前缀 `[scheduler.py:222]`、[config/scheduler.py:54](vllm/config/scheduler.py#L54) |
| 图 / 编译 | level 3、`cudagraph_mode=1`（PIECEWISE）、capture size 8…512、`use_async_output_proc=True` | `core.py:76` 的 config dump |
| 客户端 | `aiohttp`、`request_timeout 150 s`、`drain_timeout 120 s`、`metrics 0.25 s` 采样、SLO = TTFT 200 ms + TPOT 15 ms | `summary.json.meta`、[capacity_probe.py:386-418](scripts/capacity_probe.py#L386-L418) |
| 证据 | 每臂 `summary.json` / `request_records.json` / `metrics.jsonl` / `metrics_pre|post.txt` / `contract.txt` / `probe.log` | `results/week18_day6/*/` |

- 五个臂**共用一个 server 进程**（`APIServer pid=4341`，06:21 启动），无重启 ⇒ 臂间差异不含冷启动。
- `probe.log` 里的 `out` 路径是 `/data/vllm-learning/results/...`（运行机），本仓库 `results/` 是拷回的副本；时间戳统一是 **UTC**，与 doc 里 +08:00 的本地时间相差 8 h。
- `contract.txt` 是每臂的自动指纹：时间戳 + `vllm_version 0.10.2` + GPU/驱动 + `phases=... policy=... seed=... long_frac=...`。

## 2. 请求路径（客户端侧）

1. **相位解析**：`parse_phases("name:sec:rate,...")` 逐段累加 `start/end`，拒绝重复名、非正速率/时长（[capacity_probe.py:46-59](scripts/capacity_probe.py#L46-L59)）⇒ 相位边界是**计划时间**，不是实际观测。
2. **到达计划**：`schedule_requests` 用 `rng.expovariate(rate)` 在同一 seed 下生成固定 `planned_t`；类别用**独立** `Random(seed + 100003)`（[capacity_probe.py:62-76](scripts/capacity_probe.py#L62-L76)）⇒ 同 seed 的 open/gate 两臂拿到**逐请求相同**的到达时刻与类别，这是 baseline 两臂 offered 完全相等（2,270 / 2,601 / 3,935）的原因。
3. **样本构造**：短/长各用一个 `RandomDataset`（`seed+17` / `seed+29`），`range_ratio=0` ⇒ 长度恒定（128/128、512/128），无长度噪声（[capacity_probe.py:294-302](scripts/capacity_probe.py#L294-L302)）。
4. **准入决策**：主循环 `await asyncio.sleep(planned_t)` 后，用**计划时间** `bucket.allow(r['planned_t'], cost)`，不是事件循环实际 dispatch 时间（[capacity_probe.py:349-355](scripts/capacity_probe.py#L349-L355)）⇒ 决策在跨重复可复现；`rps` 的 cost=1，`cost` 的 cost=`1*input_len + 2*output_len`（短 384 / 长 768）。
5. **本地拒绝**：`reason='local_reject'` 或 `client_guard_drop`（`max_inflight_guard=2048`，本次 0 次）直接写入记录并 `continue`，**不创建 `send_one` task**（[capacity_probe.py:356-363](scripts/capacity_probe.py#L356-L363)）⇒ 无 `dispatch_t`、无 HTTP、服务端零感知。
6. **时间戳语义**：`dispatch_t = time.monotonic() - t0` 取自 `session.post` **之前**（[capacity_probe.py:137-142](scripts/capacity_probe.py#L137-L142)）⇒ 含事件循环调度与 aiohttp 建连，**不等于 TCP 发送时刻**，更不等于服务端到达；`dispatch_lag = dispatch_t - planned_t`（实测 p99 ≤ 4.1 ms，说明客户端没有被自己的循环拖住）。
7. **SSE 与指标**：`ttft` = 首个 `choices` 事件 − dispatch；`e2e` = 最后一条 data − dispatch；`tpot = (e2e − ttft)/(usage.completion_tokens − 1)`（[capacity_probe.py:186-194](scripts/capacity_probe.py#L186-L194)）。**必须**同时看到 `choices`、`usage` 和 `[DONE]` 才算成功，否则记 `incomplete SSE` 失败。
8. **失败记录**：非 200 记 `HTTP <code>` + 前 400 字符；异常记 `repr(err)`；`drain_timeout` 到点则 cancel 并记 `client_cancelled_at_drain_deadline`（[capacity_probe.py:195-202](scripts/capacity_probe.py#L195-L202)）。

## 3. vLLM 服务端容量路径

**每个 step 的 `Scheduler.schedule()`**（[scheduler.py:177-600](vllm/v1/core/sched/scheduler.py#L177-L600)）：

1. `token_budget = self.max_num_scheduled_tokens`（= `max_num_batched_tokens` = 2048，[scheduler.py:73](vllm/v1/core/sched/scheduler.py#L73)、[scheduler.py:196](vllm/v1/core/sched/scheduler.py#L196)）。
2. **先跑 running**（[scheduler.py:206-298](vllm/v1/core/sched/scheduler.py#L206-L298)）：每条请求 `num_new_tokens = num_tokens_with_spec − num_computed_tokens`（解码时 =1），`min(token_budget)` 后尝试 `allocate_slots`。
3. **分配失败 = 唯一抢占入口**（[scheduler.py:252-288](vllm/v1/core/sched/scheduler.py#L252-L288)）：FCFS 下 `self.running.pop()`（队尾）→ `free()` → **`num_computed_tokens = 0`** → `waiting.prepend_request()`。V1 没有 swap，抢占 = 整段重算；若弹出的正是当前请求则 `can_schedule=False` 并 break。
4. **再跑 waiting**（[scheduler.py:332-540](vllm/v1/core/sched/scheduler.py#L332-L540)）：入口就是 `if len(self.running) == self.max_num_running_reqs: break`（[scheduler.py:335](vllm/v1/core/sched/scheduler.py#L335)）⇒ `max_num_seqs` 是**准入闸门**；chunked prefill 开启时 `num_new_tokens` 被 `min(token_budget)` 切块（[scheduler.py:421-435](vllm/v1/core/sched/scheduler.py#L421-L435)）；`allocate_slots` 返回 `None` 时直接 `break`，请求**留在 waiting**（[scheduler.py:474-486](vllm/v1/core/sched/scheduler.py#L474-L486)）。
5. 收尾把 skipped 请求插回队首并断言 `len(self.running) <= max_num_running_reqs`（[scheduler.py:542-555](vllm/v1/core/sched/scheduler.py#L542-L555)）。

**三层"队列"必须分开**（本实验的核心坑）：

| 层 | 计数来源 | 本次证据 |
|---|---|---|
| 客户端/HTTP 在途 | `dispatch_t` 后、响应前（in-flight = 注入累计 − 完成累计） | overload 期末 **263** |
| 引擎/API frontend 输入队列 | 无 `/metrics` 指标 | **不可见**（未证伪也未证实） |
| vLLM scheduler `waiting` | `len(self.waiting)`（[scheduler.py:1183](vllm/v1/core/sched/scheduler.py#L1183)） | 峰 **19**，只在 56.3–59.8 s |

- **为什么 105 req/s 时 waiting 几乎不涨**：running 到 253 时 KV 用量 0.799（≈52.4k tok = 253 × ~207 tok），还剩 **824 blocks ≈ 100 个 prefill**，新请求要 8 blocks 就能上 ⇒ `allocate_slots` 成功、闸门不触发；多出来的需求只让**batch 变长**，于是 TPOT 从 5.53 涨到 16.93 ms，E2E 从 0.73 涨到 2.25 s。对照核算：253 running ÷ 95 req/s ≈ 2.66 s 的在途时间，与实测 E2E 2.245 s（p50）+ 尾部一致。
- **Day2 的 100/120 点出现 waiting 110/256 并不矛盾**：那是"注入 ≫ `max_num_seqs`/KV 能容纳的并发"的形态；本配置下 256 并发刚好被 65,536 tok 覆盖，所以失效前移到 TPOT。**两种失效模式取决于需求/并发比，不能只看 waiting。**
- `--num-gpu-blocks-override` 只改 **block pool 大小**（8254 → 4096，[kv_cache_utils.py:799-805](vllm/v1/core/kv_cache_utils.py#L799-L805)），**不**改 `max_num_seqs`、**不**改 `max_num_batched_tokens`、**不**改 `max_model_len`，也**不**改每条请求的 block 用量（由 `num_computed_tokens + num_new_tokens` 与 `max_model_len` 决定，[kv_cache_manager.py:259-268](vllm/v1/core/kv_cache_manager.py#L259-L268)）。
- **指标口径**：`vllm:num_requests_running` / `num_requests_waiting` 是 **Gauge**（[loggers.py:188-205](vllm/v1/metrics/loggers.py#L188-L205)），每 step 由 `SchedulerStats` 刷新（[loggers.py:511-519](vllm/v1/metrics/loggers.py#L511-L519)）；`vllm:num_preemptions` 是 **Counter**（[loggers.py:271-275](vllm/v1/metrics/loggers.py#L271-L275)），必须差分；`vllm:kv_cache_usage_perc` 是 0–1 比例（[loggers.py:244-248](vllm/v1/metrics/loggers.py#L244-L248)），不是百分比。

## 4. 时间边界 / SLO 口径

| Metric | 时间窗口 | 分母 | 代码 | 本次数值示例 |
|---|---|---|---|---|
| Offered arrival rate | 相位计划窗 `[start,end)` | 相位秒数 | [capacity_probe.py:252](scripts/capacity_probe.py#L252) | overload 104.04（含拒绝） |
| Injected proxy | `dispatch_t < end` 的到达者 | 相位秒数 | [capacity_probe.py:253](scripts/capacity_probe.py#L253) | gate overload 78.68 |
| Completed throughput | `success and start ≤ end_t < end` | 相位秒数 | [capacity_probe.py:254](scripts/capacity_probe.py#L254) | recovery_open 68.93 |
| SLO Goodput | 同上 + `good()` | 相位秒数 | [capacity_probe.py:255](scripts/capacity_probe.py#L255) | recovery_open overload 19.04 |
| Cohort SLO attainment | 按 `arrival_phase` 归组 | 该相位**全部 offered** | [capacity_probe.py:249](scripts/capacity_probe.py#L249) | cost 臂 86.8% |
| Accepted-only SLO | 同上 | 该相位**最终成功者** | [capacity_probe.py:250](scripts/capacity_probe.py#L250) | gate overload 100% |

- **为什么 recovery 的完成率能超过到达率**：recovery 窗口 `[60,120)` 的完成数含 224 个 **overload 期间到达**的请求（它们 60 s 时还在途），所以 `68.93 > 65.58`。窗口完成率是"单位时间内完成多少"，不是"这一刻的稳态服务率"；短窗口还会反向偏差：40 s 窗口、E2E ~1 s ⇒ 起始空管道让完成率比注入低 **≈1.0–1.2 req/s**（`sweep_90` 的 88.08 vs 89.70 就是这个量级）。
- `recovery_clear_s` 由 `analyze_day6.py` 用"overload 的 waiting 峰 >0 且 recovery 内 waiting=0 连续 ≥1 s"估算（[analyze_day6.py:9-31](scripts/analyze_day6.py#L9-L31)）⇒ 它只描述 **scheduler 队列**，本 run 给出 0.106 s 而实际恢复是 p50 ~5 s / p99 ~10–15 s，正是"queue clear ≠ 延迟恢复 ≠ 全排空"的实证。

## 5. Evidence hierarchy

**Observed**：`cache_config_info` 的 4096 blocks / block 16 / prefix cache off；CLI 的 `max_num_seqs 256`、`max_num_batched_tokens 2048`、chunked prefill on；`kv_cache_utils.py:802` 的 8254 → 4096 override；sweep 70/80/90 的 completed 68.95 / 78.65 / 88.075 与 TTFT p99 55.8 / 69.1 / 81.8、TPOT p99 7.25 / 8.67 / 11.52、全 100% cohort SLO；overload 的 running 253、KV 0.799、preemptions 0、waiting 峰 19（仅 56.3–59.8 s）、TPOT p50 16.93 / p99 18.89、goodput 19.04；gate overload 注入 78.68、拒 634、accepted SLO 100%；in-flight 39 → 263 → 62；mix_cost 拒 472（短 230 / 长 242）；全部臂 `metrics_errors=0`、`http_error_or_timeout=0`、`guard_drop=0`、`record_accounting_ok=true`。

**Inferred**：过载被吸收为"batch 变长 / TPOT 通胀"而非 scheduler 排队（waiting 峰 19、KV 0.799、由 253 running 与 95 req/s 反推 2.66 s 在途时间）；`max_num_seqs` 先绑、KV 落后 ~23%（由 0.799 与每请求 block 数外推）；mix_cost 的 472 个拒绝由桶算术决定（需求 28,800/s > 24,000/s，接受成本 1,438,080/1,446,000 = 99.4%）；吞吐天花板 ≈12.2k tok/s（90 点 11,274、105 点 12,170、Day5 rate=100 的 12,168.5）。

**Unknown / cannot conclude**：in-flight 的 263 个请求具体停在客户端、API frontend 还是引擎输入队列（无相应指标）；4096 blocks 与 `max_num_seqs=256` 谁是第一约束（两者在本负载下等价，未做 512-seq 或更大 block 的对照臂）；单 seed + 40–60 s 窗口下 p99 的置信区间；`--num-gpu-blocks-override` 与 chunked prefill 各自对 TPOT 拐点的贡献；`P+2O` 权重与真实 GPU 时间的换算；服务端准入（429/队列上限）下的失效模式。
