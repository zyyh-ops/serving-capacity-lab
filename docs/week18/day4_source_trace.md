# Week 18 Day 4 — Scheduler/KV Source Trace

## 0. 结论先行

- `max_num_seqs` 只限制 **`running` 列表长度**（[scheduler.py:71](vllm/v1/core/sched/scheduler.py#L71)、[scheduler.py:335](vllm/v1/core/sched/scheduler.py#L335)），不限制 token、不代表 KV 驻留数、更不代表 GPU 饱和。所以"waiting 变少"最多说明**队列变少**，不能推出吞吐变高 —— 这正是 admission A/B 的实测结果（吞吐 +0.7%）。
- 出现"KV 用量 <1 但 waiting 在涨"的典型条件就是**条数上限先到**：256 臂 `running` 精确顶在 256.0、KV 仅 0.409、waiting 峰 24。512 臂同一位置不再 break，waiting 恒 0（另一个可能来源是 [scheduler.py:196](vllm/v1/core/sched/scheduler.py#L196) 的 token budget 耗尽，本 workload 下不会先触发）。
- 提高 `max_num_seqs` **不会改 CUDA graph**（capture 列表按 `min(max_num_seqs*2, 512)` 生成，256/512 同时 clamp 到 512），但**会改 runner 内存**：`profile_run` 的虚拟 batch 条数 `= min(max_num_batched_tokens, max_num_seqs)`，峰值 activation 1.39 → 2.77 GiB，KV blocks **8,254 → 7,448**。这是 admission A/B 的隐藏混杂。
- 抢占只在 **running 分配失败**时发生（[scheduler.py:252-291](vllm/v1/core/sched/scheduler.py#L252-L291)）：FCFS 弹尾部、释放全部 block、`num_computed_tokens=0`、插回 waiting 头部 ⇒ 代价模型是**全前缀重算**。
- 但本轮 token 计数与这个代价模型**不一致**：抢占 356/360 次，总 prompt token 却只有名义值的 1.00–1.01×。源码层面能给出的只是机制；"浪费多少"必须靠 per-step trace，不属于本轮的观测范围。

## Environment

| 项 | 值 |
|---|---|
| vLLM / 引擎 | **0.10.2**，V1（`VLLM_USE_V1=1`），`EngineCore_DP0`，单 engine |
| 硬件 / 精度 | RTX 4090 24 GB，driver 570.124.06，bf16，TP=PP=DP=1 |
| 模型 | `Qwen/Qwen3-0.6B` + tokenizer |
| 公共 CLI | `--dtype auto --max-model-len 2048 --gpu-memory-utilization 0.7 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching` |
| Admission 臂差异 | `--max-num-seqs 256` / `512`（其余相同） |
| KV 臂差异 | 两臂均 `--max-num-seqs 256`，`--num-gpu-blocks-override 1024` / `512` |
| block_size | **16**（`vllm:cache_config_info{block_size="16"}`） |
| 实际 blocks / tokens | 8,254 / 132,064 · 7,448 / 119,168 · **1,024** / 16,384 · **512** / 8,192 |
| `Maximum concurrency for 2048 tok` | 64.48x · 58.19x · 8.00x · 4.00x |
| 内存 profiler | weight 1.12 GiB；activation **1.39 / 2.77 / 1.39 / 1.39** GiB；non-torch 0.02；CUDAGraph 0.49 |
| 图 / 编译 | level 3、`cudagraph_mode=1`(PIECEWISE)、67 个 capture size（512…1）、`max_capture_size=512`、4 臂均命中图缓存（2.14–2.53 s） |
| Prefix cache | **关闭**（`enable_prefix_caching=False`，`prefix_cache_hits_total` 恒 0） |

- `--num-gpu-blocks-override` 只改**可用** block 数，KV 张量仍按 profiler 给满（日志 `Current kv cache memory in use is 15146510540 bytes` 但 usable=512）⇒ kv_512 有 93.8% 的预留 KV 内存实际不可用。
- 每条 CLI 与启动日志均在 `results/week18_day4/servers/<arm>.log(.cmd)`；`num_gpu_blocks` 在 [kv_cache_utils.py:799-805](vllm/v1/core/kv_cache_utils.py#L799-L805) 被 override，随后 [kv_cache_utils.py:864-868](vllm/v1/core/kv_cache_utils.py#L864-L868) 打印容量与最大并发。

## 1. Admission path

- **`max_num_seqs` → `max_num_running_reqs`**：`self.max_num_running_reqs = self.scheduler_config.max_num_seqs`（[scheduler.py:71](vllm/v1/core/sched/scheduler.py#L71)）；同处 `max_num_scheduled_tokens = max_num_batched_tokens`，每步 `token_budget` 从这里取（[scheduler.py:196](vllm/v1/core/sched/scheduler.py#L196)）。
- **waiting 准入的 break 点**：
  - [scheduler.py:333-336](vllm/v1/core/sched/scheduler.py#L333-L336)：`if not preempted_reqs:` 之后 `while self.waiting and token_budget > 0:`，一旦 `len(self.running) == self.max_num_running_reqs` 就 **break**。
  - [scheduler.py:484-486](vllm/v1/core/sched/scheduler.py#L484-L486)：waiting 侧 `allocate_slots` 返回 `None` 也 **break**（此处**不抢占**，只是本步不再接纳）。
  - 注意 333 行的 `if not preempted_reqs:`：**某一步只要发生过抢占，本步就不再接纳任何 new request**。
- **"Waiting 涨而 KV <1" 的条件**：就是 [scheduler.py:335](vllm/v1/core/sched/scheduler.py#L335) 的条数上限（或 196 行的 token budget 耗尽）。实测吻合：admission_256 `running` 峰 = **256.0（=cap）**、KV 峰 0.409、waiting 峰 24；admission_512 `running` 峰 261（**未触及** 512）、waiting 恒 0、KV 峰 0.466。
  - 旁证：`kv%` 打满时 waiting 涨的是另一条路径 —— running 侧分配失败触发抢占（§2），与 admission 无关。
- **提高 `max_num_seqs` 会改 graph 吗？** 不会。capture 列表默认 `min(max_num_seqs*2, 512)`（[config/scheduler.py:77-83](vllm/config/scheduler.py#L77-L83)），256 与 512 都落到 512 ⇒ 我方两臂的 67 个 size、0.49 GiB、图缓存命中完全一致。
- **会改 runner 内存吗？** 会，而且是本次实验的混杂源：`profile_run` 用 `_dummy_run(self.max_num_tokens)`（[gpu_model_runner.py:3031](vllm/v1/worker/gpu_model_runner.py#L3031)），其中 `num_reqs = min(num_tokens, max_num_reqs)`、`min_tokens_per_req = num_tokens // num_reqs`（[gpu_model_runner.py:2702-2705](vllm/v1/worker/gpu_model_runner.py#L2702-L2705)）；sampler/输出 buffer 也按 `max_num_reqs` 开（[gpu_model_runner.py:2919-2928](vllm/v1/worker/gpu_model_runner.py#L2919-L2928)）。2048 token 下 256 vs 512 条序列 ⇒ activation 1.39 → 2.77 GiB（[gpu_worker.py:222-266](vllm/v1/worker/gpu_worker.py#L222-L266)），直接扣掉 806 个 block。

## 2. KV allocation and preemption

- **`allocate_slots` 的失败条件**：`num_blocks_to_allocate > self.block_pool.get_num_free_blocks()` ⇒ `return None`（[kv_cache_manager.py:263-272](vllm/v1/core/kv_cache_manager.py#L263-L272)）。需要多少 block 由 `coordinator.get_num_blocks_to_allocate` 按 `num_computed_tokens + num_new_tokens + num_lookahead_tokens` 与 `max_model_len` 取 min 后算出。
- **抢占谁**：running 循环内分配失败时
  - FCFS（默认）：`preempted_req = self.running.pop()`，即**最晚入队**的那个（[scheduler.py:269](vllm/v1/core/sched/scheduler.py#L269)）；
  - `priority` 策略：`max(running, key=(priority, arrival_time))`（[scheduler.py:260-267](vllm/v1/core/sched/scheduler.py#L260-L267)）；
  - 若弹出的正是当前 request 本身，`can_schedule=False` 并 break（[scheduler.py:281-284](vllm/v1/core/sched/scheduler.py#L281-L284)）。
- **释放什么 / 计数怎么动**：`kv_cache_manager.free(preempted_req)`（[scheduler.py:271](vllm/v1/core/sched/scheduler.py#L271) → [kv_cache_manager.py:305-313](vllm/v1/core/kv_cache_manager.py#L305-L313)，整条请求的 block 全部释放，逆序 evict）→ `status=PREEMPTED`、**`num_computed_tokens = 0`**（[scheduler.py:273-274](vllm/v1/core/sched/scheduler.py#L273-L274)）→ `waiting.prepend_request()`（[scheduler.py:279](vllm/v1/core/sched/scheduler.py#L279)）。**V1 没有 swap 路径，只有 recompute。**
- **恢复时会发生什么**：被抢占者按 FCFS 回到队首，`num_computed_tokens=0`，重新走 waiting 准入；此刻 `get_computed_blocks` 会尝试前缀命中（[scheduler.py:378-382](vllm/v1/core/sched/scheduler.py#L378-L382) → [kv_cache_manager.py:153](vllm/v1/core/kv_cache_manager.py#L153)），但本实验 `--no-enable-prefix-caching` ⇒ 命中 0 ⇒ **整段前缀重算**。这就是"抢占 = 浪费 token"的源码依据。
- **本轮的反常**：按上面的模型，356 次抢占 × 520-token 前缀应让 prompt 总量比名义值多 **+74%**（+185k token）；实测只有 **1.00–1.01×**（250,603 vs 249,600；406,732 vs 406,232），`vllm:generation_tokens_total` 也是 1.00×。可能的解释：(a) 受害者被抢占时已算前缀极短；(b) token 计数器口径不含重算。**本轮无法判定**，需要 per-step `num_scheduled_tokens` / `num_computed_tokens` / `preempted_req_ids`。
- **per-step 日志**：无。INFO 只有 10 s 滚动行（`Running/Waiting/GPU KV cache usage`），抢占没有任何逐次打印 ⇒ 只有 `vllm:num_preemptions_total` 的差分可作证据。

## 3. Metrics semantics

- **`vllm:num_requests_running`**：`len(self.running)`（[scheduler.py:1182](vllm/v1/core/sched/scheduler.py#L1182)），gauge 定义于 [loggers.py:188-204](vllm/v1/metrics/loggers.py#L188-L204)、每步在 [:512-519](vllm/v1/metrics/loggers.py#L512-L519) `set`。语义是**被调度进 batch 的序列条数**，不是 KV 驻留条数、不是 GPU 利用率。反证：kv_512 的 `running` 峰 **14 > 8,192/1,024 = 8**（满长并发上界）—— 因为 chunked prefill 下请求可以只持有"本步算过的那部分" KV。
- **`vllm:num_requests_waiting`**：`len(self.waiting)`（[scheduler.py:1183](vllm/v1/core/sched/scheduler.py#L1183)），含被抢占回插的请求；**不含**本步因 budget 用尽而未调度的 running 请求。
- **`vllm:kv_cache_usage_perc`**：`1 - free/(num_gpu_blocks - 1)`（[block_pool.py:307-318](vllm/v1/core/block_pool.py#L307-L318)，减 1 是 null block），gauge 见 [loggers.py:244-249](vllm/v1/metrics/loggers.py#L244-L249) 与 [:512-519](vllm/v1/metrics/loggers.py#L512-L519)。**分母随臂而变**（8,254 / 7,448 / 1,024 / 512），所以 0.409 与 0.466 不可直接比；可比的绝对量是 54.0k vs 55.6k token（+2.9%）。`gpu_cache_usage_perc` 是同一量的 deprecated 别名。
- **`vllm:num_preemptions_total`**：Counter（[loggers.py:270-275](vllm/v1/metrics/loggers.py#L270-L275)），每步 `inc(iteration_stats.num_preempted_reqs)`（[:538-539](vllm/v1/metrics/loggers.py#L538-L539)），必须**差分**。本轮 0 → 356/360 单调无回绕，无 reset。
- **TTFT / TPOT / E2E**：客户端 SSE 口径。TTFT = 首个含 `choices` 的事件；E2E = 最后一个 data（含 `usage` 尾包）；`TPOT = (e2e − ttft)/(output_tokens − 1)`。服务端另有 `request_queue_time_seconds`(WAITING) / `request_prefill_time_seconds` / `request_decode_time_seconds`，两套口径**不能互相代入**（preflight 里 16 条请求 queue_time ≈ 0.0006 s，而客户端侧同时包含 HTTP+SSE 开销）。
- **采样**：`/metrics` 请求间隔 0.25 s，实际上本机中位 0.27 s；最大间隔 0.428/0.518/0.380/0.379 s ⇒ **峰有漏采可能**，`waiting_peak`/`running_peak`/`kv_peak` 都是下界。

## 4. Evidence hierarchy

**Observed（日志/指标/产物直接可读）**
- 4 条 CLI、`num_gpu_blocks` 8,254/7,448/1,024/512、`GPU KV cache size`、block_size 16、4 臂图配置一致、图缓存命中。
- activation 1.39 / 2.77 GiB 与由此产生的 KV 差异；`--num-gpu-blocks-override` 被打印为 `Overriding num_gpu_blocks=8254 with ... 512`。
- `waiting` 峰 24 / 0 / 216 / 311；`running` 峰 256 / 261 / 28 / 14；`kv` 峰 0.409 / 0.466 / 1.0 / 1.0；preemptions 0 / 0 / 360 / 356。
- 总 token：prompt 1.00–1.01× 名义、gen 1.00× 名义；480/480 与 2,987/2,987 全部 200、`finish_reason=length`、长度 mismatch 0。

**Inferred（机制推断，不可被本轮日志直接证实）**
- admission 臂的等待来自 [scheduler.py:335](vllm/v1/core/sched/scheduler.py#L335) 的条数上限（因 running 精确等于 cap 而 KV 未满）。
- KV 臂的 waiting 增长来自 [kv_cache_manager.py:270-272](vllm/v1/core/kv_cache_manager.py#L270-L272) 的 `None` + §2 的抢占-再接纳循环。
- 抢占的代价模型是整段前缀重算（源码如此），但本轮 token 计数不支持"每次全量重算"。

**Unknown（需要 Day5 补）**
- 每次抢占实际丢弃多少 token（需 per-step `num_computed_tokens` 与 preempted id）。
- `num_preemptions_total` 与 token 计数器的口径差在哪一层（scheduler / engine core / metrics）。
- GPU 利用率、step 时间分解（prefill/decode/sampler/主机开销）。
- 若开启 prefix cache，抢占后的重算能否被 prefix hit 吸收。
- 干净单变量 KV 曲面：两臂都固定 `max_num_seqs`、都用 `--num-gpu-blocks-override`。
