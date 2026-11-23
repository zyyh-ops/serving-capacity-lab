# Week 18 Day 3 — Source Trace

## 0. 结论先行

- vLLM V1 的 `Scheduler.schedule()` **没有独立的 prefill/decode 阶段**：每步以 `max_num_batched_tokens` 为 `token_budget`，先给 running 各分 token，再把剩余预算给 waiting 的新 prefill；`num_scheduled_tokens` 是本步每个请求实际计算的 token 数，`num_computed_tokens` 是已算前缀长度。chunked prefill 只是把 `num_new_tokens` 截成 `min(num_new_tokens, token_budget)`。
- 本轮 waiting 采样恒 0、running ≤52、KV ≤0.158；**若 budget=2048**，最坏需求 `52+520=572` 只占 27%，没有 chunking 的结构性理由；TTFT/TPOT 偏移只能来自“prefill 与 decode 共享一次 forward”的 step-level 效应。
- 客户端 short 计划只由 `seed` 决定（long 用 `seed+100003`），本轮 11,442 对 / 3,565 对 `planned_t` 完全相等、prompt token 数 0 mismatch；`dispatch_t` 在 `session.post()` 之前，`guard drop` 是未发送，不是 HTTP 拒绝。
- 4 条 TTFT>200 ms 与 ~200 ms 客户端事件循环停顿重合（`dispatch_lag` 197/204 ms，metrics 间隔 0.495 s），当时 waiting=0；不能当服务端容量证据。
- **配置是 C 级证据**：`server.log` 无 `vllm serve` 行；budget 2048 / chunked prefill / prefix cache 沿用 Day 1。

### 0.1 Environment / reproducibility

| 项 | 值 |
|---|---|
| vLLM / 硬件 | **0.10.2**（Day 1）；Python 3.12.16、torch 2.8.0+cu128、CUDA 12.8、driver 570.124.06、RTX 4090 24 GB；本仓库 `vllm/` 为节选源码 |
| 模型 | `Qwen/Qwen3-0.6B` + tokenizer；revision 未 pin；`Qwen3ForCausalLM` |
| Server CLI | **Day 3 未记录**；沿用 Day 1：`CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve Qwen/Qwen3-0.6B --dtype auto --max-model-len 2048 --gpu-memory-utilization 0.7` |
| Engine | V1；bf16；TP=PP=DP=1；`max_model_len=2048`；`max_num_batched_tokens=2048`；chunked prefill 开启；prefix cache 开启；`level=3` / PIECEWISE graph / max capture 512（均 Day 1） |
| `max_num_seqs` | Day 2 推断 256；Day 3 只能确认 **≥52** |
| KV | 14.11 GiB / 132,064 tokens / 8,254 blocks（Day 1）；Day 3 峰值 0.158 ≈ 1,304 blocks |
| Warmup / cache | `t0` 前每类 1 条非流式 warmup；`server.log` 无重启记录，prefix cache 未清空，跨 run 累积（推断） |
| 客户端 | 单 asyncio 进程 + aiohttp；guard 512；timeout 90 s；`/metrics` 0.25 s；同机 loopback |
| 顺序 / 时长 | baseline(60) → long_prefill(133) → long_decode(280) → control(575) → mixed(1172)，seed 2027；control_2028(180) → mixed_2028(180)，seed 2028 |

> `run_day3.sh` 的 `SECONDS_OVERRIDE` 默认 60，但落盘的主实验是 60/133/280/575/1172；以 `summary.json` 为准。`comparison.csv` 只含 2027 的 5 个 run，2028 两轮直接读各自 `summary.json`。

## 1. vLLM 0.10.2 V1 scheduling trace

### 1.1 状态归属

| 状态 | 位置 | 作用 |
|---|---|---|
| `waiting`/`running`、`requests`、`num_computed_tokens` | `Scheduler`（[scheduler.py:112–124](vllm/v1/core/sched/scheduler.py#L112-L124)） | 准入、排队、prefill 进度 |
| `num_scheduled_tokens` / `total_num_scheduled_tokens` | `SchedulerOutput`（[output.py:129–134](vllm/v1/core/sched/output.py#L129-L134)） | 本步每请求/整批要算的 token 数 |
| KV 分配 / 抢占 | `Scheduler` + `KVCacheManager`（[scheduler.py:252–291](vllm/v1/core/sched/scheduler.py#L252-L291)） | `allocate_slots` 失败才抢占 |
| `CachedRequestState`、`InputBatch`、positions/seq_lens | GPU worker（[gpu_model_runner.py:555–568](vllm/v1/worker/gpu_model_runner.py#L555-L568)、[gpu_model_runner.py:873–944](vllm/v1/worker/gpu_model_runner.py#L873-L944)） | 持久化请求状态、组 batch、算张量布局 |

scheduler 不执行 forward，只输出 `SchedulerOutput`；worker 用 `num_computed_tokens + arange` 生成 positions（[gpu_model_runner.py:898–902](vllm/v1/worker/gpu_model_runner.py#L898-L902)），用 `num_computed_tokens + num_scheduled_tokens` 生成 `seq_lens`（[gpu_model_runner.py:938–944](vllm/v1/worker/gpu_model_runner.py#L938-L944)）。

### 1.2 token budget / 计数

- `max_num_running_reqs = max_num_seqs`，`max_num_scheduled_tokens = max_num_batched_tokens`（[scheduler.py:71–73](vllm/v1/core/sched/scheduler.py#L71-L73)）；每步 `token_budget = max_num_scheduled_tokens`（[scheduler.py:196](vllm/v1/core/sched/scheduler.py#L196)）。
- running：`num_new_tokens = num_tokens_with_spec + placeholders − num_computed_tokens`，再受 `long_prefill_token_threshold`、budget、`max_model_len` 限制（[scheduler.py:211–224](vllm/v1/core/sched/scheduler.py#L211-L224)）；正常 decode 为 1。
- waiting：先查 prefix cache 命中（[scheduler.py:378–382](vllm/v1/core/sched/scheduler.py#L378-L382)），`num_new_tokens = num_tokens − num_computed_tokens`，再截到剩余 budget（[scheduler.py:399–435](vllm/v1/core/sched/scheduler.py#L399-L435)）。
- 每调度一个请求，写 `num_scheduled_tokens[req]`、扣 `token_budget`（[scheduler.py:293–298](vllm/v1/core/sched/scheduler.py#L293-L298)、[scheduler.py:524–527](vllm/v1/core/sched/scheduler.py#L524-L527)）；`_update_after_schedule` 在步末把 `num_computed_tokens += num_scheduled_tokens`（[scheduler.py:630–646](vllm/v1/core/sched/scheduler.py#L630-L646)）——这是分块 prefill 的进度条。running 已占 KV、可被调度；waiting 未准入、不占 KV。

### 1.3 `schedule()` 走读

1. running 优先（[scheduler.py:206–250](vllm/v1/core/sched/scheduler.py#L206-L250)）：每个请求先拿 token；`num_new_tokens==0` 时 `continue`（不 break），后面的请求仍可能被调度。
2. KV 分配失败 → FCFS 下 `running.pop()`、释放、`num_computed_tokens=0`、放回 waiting 头部（[scheduler.py:252–291](vllm/v1/core/sched/scheduler.py#L252-L291)）；本轮 preempt Δ0，未触发。
3. waiting 准入（[scheduler.py:332–336](vllm/v1/core/sched/scheduler.py#L332-L336)）：`len(running)==max_num_running_reqs` 就 break —— `max_num_seqs` 限制的是 **running 列表长度**，不是 token 数。
4. chunked prefill（[scheduler.py:429–435](vllm/v1/core/sched/scheduler.py#L429-L435)）：未开启且 `num_new_tokens>budget` 时放回 waiting 头部；开启时 `num_new_tokens=min(num_new_tokens,budget)`。
5. 接纳后 `running.append`、`status=RUNNING`、写 `num_computed_tokens`（[scheduler.py:509–529](vllm/v1/core/sched/scheduler.py#L509-L529)）；`SchedulerOutput` 汇总（[scheduler.py:567–600](vllm/v1/core/sched/scheduler.py#L567-L600)）。收尾允许 running 里有请求本步未被调度（[scheduler.py:542–555](vllm/v1/core/sched/scheduler.py#L542-L555)）。

### 1.4 一个混合 step（用本轮数字）

设 running=35、budget=2048、waiting 有 1 条 520-token long：

1. 35 个 decode 各 1 token → `total=35`、`budget=2013`。
2. long 无缓存命中 → `num_new_tokens=520 ≤ 2013`；`allocate_slots` 成功；running=36、`budget=1493`。
3. worker：35 个 decode 的 positions 是各自 `num_computed_tokens`，long 是 0–519；`query_start_loc` 类似 `[0,1,…,35,555]`，`max_query_len=520`；一次 forward 处理 **555** 个 token（1-token decode query 与 520-token prefill query 共存）。
4. 该 prefill 一步完成，最后位置 sample 出首 token（TTFT 在本步结束）；下一步 long 变成普通 decode。

若 budget=512：running 先拿 35，剩 477，long 只能拿 477；第 1 个 chunk 不产生 token，`num_computed_tokens=477`；第 2 步再算 43 token 才出首 token，TTFT 多跨一步。

### 1.5 何时拆、怎么证明

- 条件：`chunked_prefill_enabled` 且 `num_new_tokens > budget`；KV 分配失败；`long_prefill_token_threshold > 0` 且 prompt 超阈值。注意 `max_num_partial_prefills` 只在 `SchedulerConfig`（[config/scheduler.py:54–66](vllm/config/scheduler.py#L54-L66)），本快照 V1 `schedule()` **没有引用它**。
- 证据：`SchedulerOutput.num_scheduled_tokens[req]` 小于剩余 prompt；`num_computed_tokens` 两步只前进一部分；首个 chunk 后无 sampled token（[scheduler.py:976–977](vllm/v1/core/sched/scheduler.py#L976-L977)、[scheduler.py:1026–1028](vllm/v1/core/sched/scheduler.py#L1026-L1028)、[gpu_model_runner.py:962–967](vllm/v1/worker/gpu_model_runner.py#L962-L967)）；`query_start_loc`/`max_query_len` 显示混合 query 长度（[gpu_model_runner.py:930–936](vllm/v1/worker/gpu_model_runner.py#L930-L936)、[gpu_model_runner.py:1041–1050](vllm/v1/worker/gpu_model_runner.py#L1041-L1050)）。
- 本轮：waiting=0、running≤52、最坏 572<2048 → **2048 下不需要拆**；若实际 budget=512，则 520-token prompt 必然拆，但没有 per-step trace，不能证明。

### 1.6 `max_num_seqs` 限制什么

限制 running 序列数（准入并发）；不限制每步 token（budget 管）、不决定 GPU batch、不代表 GPU 饱和；budget 用尽时 running 请求也会被跳过。本轮只能推断 `max_num_seqs ≥52`，不能确认具体值，也不能排除恰好等于 52。

## 2. Client measurement path

### 2.1 为什么 short 流相同

`arrival_times` 用 `random.Random(seed).expovariate(rate)`（[workload_probe.py:38–57](scripts/workload_probe.py#L38-L57)）；short 用 `args.seed`、long 用 `seed+100003`，两条独立 RNG 流，long 不改变 short 的 `planned_t`。prompt 侧 short 用 `seed+11`、long 用 `seed+29`（[workload_probe.py:190–197](scripts/workload_probe.py#L190-L197)）。逐条验证：11,442 对 / 3,565 对 `max abs(Δplanned_t)=0`、prompt token mismatch 0；内容一致性由确定性 `RandomDataset` 推断。

### 2.2 planned_t → dispatch_t → socket

主循环到点后先查 `len(pending)>=max_inflight`，再建 task（[workload_probe.py:233–245](scripts/workload_probe.py#L233-L245)）；`send_one()` 第一行 `time.monotonic()` 即 `dispatch_t`，随后才构造 payload / `session.post()`（[workload_probe.py:99–114](scripts/workload_probe.py#L99-L114)）。因此 `dispatch_lag` 不含 TCP/frontend；本轮 P99 1.29–2.38 ms，但出现过 197.7/204.0 ms 停顿（§2.5）。Poisson 有限窗使 baseline 实际 615 条/60 s=10.25，不是发送失败。

### 2.3 TTFT / TPOT / SSE gap

- probe：按行读 SSE，空行派发 data 事件；第一个含 `choices` 的事件为 TTFT（content 可空）；`usage.completion_tokens` 为 tokens；最后一个 data（含 usage 尾包）为 e2e；TPOT=`(e2e−ttft)/(tokens−1)`（[workload_probe.py:85–147](scripts/workload_probe.py#L85-L147)）。
- 官方：`iter_any()` + `StreamedResponseHandler` 按 `\n\n` 切 SSE（[endpoint_request_func.py:20–58](vllm/benchmarks/lib/endpoint_request_func.py#L20-L58)、[endpoint_request_func.py:269–316](vllm/benchmarks/lib/endpoint_request_func.py#L269-L316)）；TTFT 同样是首个 choices；ITL 是所有 choices 事件的相邻差、池化；TPOT 同公式，latency 含 usage（[serve.py:298–305](vllm/benchmarks/serve.py#L298-L305)）。
- **SSE 事件不保证 1 token**；probe 的 `sse_choice_event_gap_p95` 是“每请求 P95 的跨请求 P95”，不是 token ITL。mixed 里 gap P95 +56.9% 而 TPOT +11.2%，说明客户端事件处理更忙，不等于每 token 变慢。
- probe 的 success 要求 200 + choices + usage + 可解析 data + `[DONE]`（[workload_probe.py:129–134](scripts/workload_probe.py#L129-L134)）；本轮 0 error。

### 2.4 window / drain / guard

- `T=duration`；`completed_in_window` 要求 `end_t<T`；`success_after_drain` 是最终成功总数。本轮窗口后完成 12/8/21/16/17/11/15（1.95%/0.60%/0.75%/0.14%/0.06%/0.31%/0.33%），Drain 0.59–2.50 s。
- 尾延迟对最终成功统计，**含 Drain**；窗口吞吐不含。`joint_slo_cohort` 分母是成功数（不含失败/drop），失败多时会偏高；本轮 0 失败。官方 `benchmark_duration` 在 `gather` 之后取（[serve.py:578–593](vllm/benchmarks/serve.py#L578-L593)），含 Drain，两套吞吐不能直接对齐。
- guard drop 只记 `status='client_guard_drop'`、不发送（[workload_probe.py:237–242](scripts/workload_probe.py#L237-L242)），计入 scheduled 但不计入 injected；本轮 0 次。HTTP 拒绝是 `status!=200`；本轮 53,407 条全 200。

### 2.5 两次客户端停顿

- `shape_long_prefill`：id 335–337 TTFT 226.6/201.6/232.5 ms；紧随的 id 338–340 `dispatch_lag` 197.7/149.6/67.0 ms，metrics running 8→13、waiting=0。
- `control_2028`：id 11 TTFT 264.3 ms；id 12/13 lag 204.0/118.7 ms；metrics 间隔 0.516→1.012 s=**0.495 s**（正常 0.26 s）。
- 这些请求 E2E 正常、TPOT 偏低（3.5–4.8 ms），符合“事件循环停 ~0.2–0.25 s、首事件延迟处理、后续 dispatch 推迟”。根因（GC/CPU/OS）未记录；不影响 P99，但拉低 joint cohort（0.99775/0.99972）。

## 3. Metrics / quality checks

- 每 0.25 s 抓 `/metrics`（与负载同 loop）；`waiting/running/preemptions/token counters` 求和、`kv` 取 max（[workload_probe.py:60–82](scripts/workload_probe.py#L60-L82)）。单 engine 下等价于全局；多 engine 时 kv max 是上界。
- 语义：`num_requests_running` 是 execution batch 里的请求数，不是 GPU 利用率；`num_requests_waiting` 只是 scheduler 队列（[loggers.py:188–204](vllm/v1/metrics/loggers.py#L188-L204)）；`kv_cache_usage_perc` 是 block 占用（[loggers.py:244–249](vllm/v1/metrics/loggers.py#L244-L249)）；preemptions 是 Counter，需差分（[loggers.py:270–290](vllm/v1/metrics/loggers.py#L270-L290)）。

| run | 样本 | 中位/最大间隔 (s) | waiting 峰/>0 | running 峰 | KV 峰 | preempt Δ | max in-flight | prompt Δ/T | gen Δ/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 227 | 0.269/0.290 | 0/0 | 14 | 0.0222 | 0 | 15 | 1,388 | 1,312 |
| long_prefill | 506 | 0.262/0.277 | 0/0 | 15 | 0.0698 | 0 | 17 | 5,191 | 1,282 |
| long_decode | 1,049 | 0.270/0.287 | 0/0 | 52 | 0.1585 | 0 | 52 | 1,360 | 5,133 |
| control | 2,149 | 0.269/0.280 | 0/0 | 29 | 0.0464 | 0 | 33 | 2,700 | 2,547 |
| mixed | 4,403 | 0.268/0.285 | 0/0 | 35 | 0.0897 | 0 | 38 | 5,268 | 3,181 |
| control_2028 | 683 | 0.264/**0.495** | 0/0 | 28 | 0.0433 | 0 | 29 | 2,684 | 2,535 |
| mixed_2028 | 680 | 0.267/0.283 | 0/0 | 36 | 0.0715 | 0 | 38 | 5,441 | 3,215 |

- 质量核对：全部 200、`success=scheduled`、0 fail/drop、preempt Δ0；output token 精确 {128}/{512}；prompt 差分与 `injected×P_est` 差 <0.1%，gen 差分与逐请求 output 和差 ≤30；精确 max in-flight 与采样 running 峰差 0–4。
- 每 group 条数：615 / 1,332 / 2,807 / 11,442 / 23,327+5,798 / 3,565 / 3,565+956；长度 P50 = **136/520**（chat template +8），P05 仍是最大值，min 37–41 / 345–353（<2% 低尾），均值 135.4–135.7 / 518.4–519.1。
- 未控制：server CLI/`max_num_seqs` 未记录；prefix cache 未清；run 顺序固定、时长不等、single seed；客户端与 server 同机且两次停顿；0.25 s 采样漏峰；无 GPU util/MFU/nsys；模型 revision 未 pin。补救：记录 CLI、每 case 重启/清 cache、AB/BA + 多 seed、多 client + 独立 metrics、Day 5 nsys/ncu、固定 revision。

## 4. Findings

**Observed**：53,407 条全 200、0 fail/drop、preempt Δ0；9,697 个样本 waiting 全 0；running 峰 52、KV 峰 0.158；short 计划 11,442/3,565 对完全一致；4 条 TTFT>200 ms 与 197–204 ms `dispatch_lag` / 0.495 s metrics 间隔重合；SSE gap +56.9% vs TPOT +11.2%。

**Inferred**：2048 budget 假设下无 chunking 证据；偏移来自 prefill/decode 共享 step 与 in-flight 变化（输出 tok/s ÷ in-flight ≈ 1/TPOT）；`max_num_seqs ≥52` 且未观察到它是瓶颈；4 条违约最可能是客户端伪影；60 s baseline 的 P99 很弱。

**Unknown**：server CLI/`max_num_seqs`/prefix cache/graph 状态；per-step `num_scheduled_tokens`/`num_computed_tokens`；GPU 时间分解；停顿根因；多 seed/更长窗口/更多长度；512 budget A/B。
