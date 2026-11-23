# Week 18 Day 5 — Source Trace

## 0. 结论先行

- **抢占只有一个入口，代价模型只有一种**：running 循环里 `allocate_slots()` 返回 `None` → 弹队尾 → `free()` 全部 block → **`num_computed_tokens = 0`** → 插回 waiting 头部（[scheduler.py:252-288](vllm/v1/core/sched/scheduler.py#L252-L288)）。V1 没有 swap，**抢占 = 整段前缀重算**。
- **恢复时确实会"先尝试前缀命中"**：waiting 侧看到 `num_computed_tokens == 0` 就走 `get_computed_blocks()`（[scheduler.py:378-382](vllm/v1/core/sched/scheduler.py#L378-L382)）；本实验 prefix cache 关闭 ⇒ 命中 0 个 block ⇒ 从 `start=0` 重算，这正是 trace 里 `start=0` 的含义。
- **trace 与源码的代价模型完全对上**：重复调度位置 **45,420** = Σ`before`（即 `previously_computed`）**45,420**，正好是 29.6% 的调度位置（见 §2.3），两个独立统计给出同一个数。
- **`vllm:prompt_tokens_total` 是"每请求一次"，不是"每次计算一次"**：`num_prompt_tokens += prompt_len` 只在 `is_prefilling` 为真时发生（[stats.py:114-118](vllm/v1/metrics/stats.py#L114-L118)），而 `is_prefilling` 在**每条请求产生第一个输出时**被无条件置 False（[output_processor.py:414](vllm/v1/engine/output_processor.py#L414)）⇒ 重算永远不进计数器。实测 Δ=53,292 = 103 次请求，而真实 prompt 计算 = 53,292 + 45,420。
- **nsys 这次给出可用的 kernel 级事实**：`admission_256_delay90_dur60` 的实际窗口是 **13:57:17.0 → 13:57:39.7Z（22.71 s / 1,619 step）**，server 全程存活（probe 3,839/3,839 全 200、`metrics_errors=0`），因此这份报告可以进结论；但仍要注意窗口只覆盖 probe 的 55%（起点在 probe 内 18.4 s 处）。

## 1. 环境与证据链

| 项 | 值 | 证据文件 |
|---|---|---|
| vLLM / 引擎 | **0.10.2**，`VLLM_USE_V1=1`，单 engine（`EngineCore_DP0`） | `results/week18_day5/servers/*.log` |
| 硬件 / 精度 | RTX 4090 24 GB，driver 570.124.06，CUDA 12.8，bf16，TP=PP=DP=1 | `contract.txt`、nsys `META_DATA_CAPTURE` |
| 模型 | `Qwen/Qwen3-0.6B`（28 层，`max_model_len=2048`） | `models.json` |
| 图 / 编译 | level 3、`cudagraph_mode=1`(**PIECEWISE**)、67 个 capture size（512…1）、图内存 0.49 GiB、命中图缓存 | `core.py:76` 那行 config dump |
| admission 两臂 | `--max-num-seqs 256 / 512`，**都** `--num-gpu-blocks-override 4096` ⇒ `num_gpu_blocks=4096`、65,536 tok、32.00x | `metrics_pre.txt` 的 `cache_config_info`、`kv_cache_utils.py:864-868` 日志 |
| KV 臂 | `--max-num-seqs 256` + `--num-gpu-blocks-override 512` ⇒ 8,192 tok、**4.00x** | `servers/kv_512.log` |
| 公共 CLI | `--dtype auto --gpu-memory-utilization 0.7 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching` | `*.log.cmd` |
| trace 工具 | `scripts/week18_day5/patch_scheduler.py`（可逆、仅 0.10.2、非 profile run）+ `analyze_trace.py` | 两个脚本 |
| trace 覆盖 | 16 warm-up + 103 被测请求；5,858 step；59,192 事件（< 100,000 上限，无截断）；单一 pid 9800 | `kv_512_trace.jsonl` / `_summary.json` |
| nsys | `launch_nsys.sh`（**重起** server，不是 attach）；`nsys_summaries.sh` 出 csv/sqlite；报告 `admission_256_delay90_dur60` | `results/week18_day5/nsys/` |
| nsys 采集窗口 | session start 13:57:16.93Z；kernel **13:57:17.02–13:57:39.73Z（22.71 s，1,619 step）**；probe 13:56:58.49–13:57:39.76Z（41.26 s）⇒ 覆盖后 55%，终点与 probe 对齐 | `sqlite TARGET_INFO_SESSION_START_TIME` + `CUPTI_ACTIVITY_KIND_KERNEL` |

- `--cuda-graph-trace=node` 会把 graph replay 展开成逐 node：这是 kernel 数（772,274）远多于 launch 类 API 调用数（182,418 + 96,910 + 37,526 = 316,854）的原因，也是 API 侧时间不可外推的原因。
- 抢占计数口径：`vllm:num_preemptions` 是 **Counter**（[loggers.py:270-275](vllm/v1/metrics/loggers.py#L270-L275)），每 step `inc(iteration_stats.num_preempted_reqs)`（[stats.py:154-155](vllm/v1/metrics/stats.py#L154-L155)），必须差分；本 run 0 → 70 单调无回绕。

## 2. 抢占与重算链

### 2.1 源码路径（逐步）

1. **分配失败 = 抢占触发器**：`allocate_slots` 里 `if num_blocks_to_allocate > self.block_pool.get_num_free_blocks(): return None`（[kv_cache_manager.py:263-272](vllm/v1/core/kv_cache_manager.py#L263-L272)）。块数需求由 `num_computed_tokens + num_new_tokens + num_lookahead_tokens` 与 `max_model_len` 取 min 得到（[kv_cache_manager.py:257-268](vllm/v1/core/kv_cache_manager.py#L257-L268)）。
2. **抢占谁**：FCFS 下 `preempted_req = self.running.pop()`（队尾、最晚入队，[scheduler.py:269](vllm/v1/core/sched/scheduler.py#L269)）；`priority` 策略下取 `(priority, arrival_time)` 最大者（[scheduler.py:260-267](vllm/v1/core/sched/scheduler.py#L260-L267)）。若弹出的正是当前请求本身 ⇒ `can_schedule=False` 并 break（[scheduler.py:281-288](vllm/v1/core/sched/scheduler.py#L281-L288)）。
3. **释放什么**：`kv_cache_manager.free(req)`（[scheduler.py:271](vllm/v1/core/sched/scheduler.py#L271) → [kv_cache_manager.py:305-313](vllm/v1/core/kv_cache_manager.py#L305-L313)，整条请求所有 block、逆序归还，供 cache 复用）。
4. **状态回滚**：`status = PREEMPTED`、**`num_computed_tokens = 0`**（[scheduler.py:273-274](vllm/v1/core/sched/scheduler.py#L273-L274)）、`waiting.prepend_request()`（[scheduler.py:279](vllm/v1/core/sched/scheduler.py#L279)）。**只有 recompute，没有 swap。**
5. **恢复**：重新进 waiting 后 `num_computed_tokens == 0` ⇒ 走 `get_computed_blocks()`（[scheduler.py:378-382](vllm/v1/core/sched/scheduler.py#L378-L382) → [kv_cache_manager.py:153](vllm/v1/core/kv_cache_manager.py#L153)）；`num_new_tokens = request.num_tokens - num_computed_tokens`（[scheduler.py:421](vllm/v1/core/sched/scheduler.py#L421)）⇒ 从 0 补齐，包含已经生成过的输出 token。`status == PREEMPTED` 时进 `scheduled_resumed_reqs`（[scheduler.py:514-520](vllm/v1/core/sched/scheduler.py#L514-L520)）。
6. **为什么 `start` 能当"逻辑位置"用**：trace 的 `start` 取的是 `_update_after_schedule` **之前**的 `req.num_computed_tokens`（[scheduler.py:630-646](vllm/v1/core/sched/scheduler.py#L630-L646) 才推进），所以它是本 step 实际要算的起始位置；prefix cache 关闭时，被抢占者在此处一定是 0。

### 2.2 trace schema（`patch_scheduler.py` 的 HELPER，与 jsonl 字段一致）

```
step      : step, n_running, n_waiting, total_tokens        # 每 step 一条，_update_after_schedule 之前
scheduled : step, request_id, start, n, request_tokens, prompt_tokens   # 区间 [start, start+n)
preempt   : step, request_id, previously_computed, request_tokens      # 在 free() 之前发射
```

`preempt.step` 用 `_W18_STEP + 1` 记录（因为 `_w18_log_step` 在 `_update_after_schedule` 处才自增），所以**抢占事件与"同一 step 的 scheduled 记录"编号对齐**；`previously_computed` 就是 `num_computed_tokens`，即**被丢弃的位置数**。

### 2.3 实测（非合成）与统计

请求 `chatcmpl-6790eb4a…`（prompt 520，最终 857 token）：

| step | 事件 | start | n | previously_computed |
|---:|---|---:|---:|---:|
| 802 | scheduled | 0 | 520 | — |
| 803–849 | scheduled | 520…566 | 1 | — |
| 850 | **preempt** | — | — | **567** |
| 923 | scheduled | **0** | 568 | — |
| 946 | **preempt** | — | — | **590** |
| 1003 | scheduled | **0** | 591 | — |
| 1051 | **preempt** | — | — | **638** |
| 1129 | scheduled | **0** | 639 | — |

- 该请求 Σ`n` = **2,826**，唯一位置 857 ⇒ **3.3× 冗余**；整跑 Σ`previously_computed` = 45,420 = `analyze_trace.py` 的 `repeated_logical_token_positions`（[analyze_trace.py:30-43](scripts/week18_day5/analyze_trace.py#L30-L43) 用 `seen_high[rid]` 的区间重叠计算）⇒ 两个独立算法给出同一个数，说明**没有其它重算来源**。
- 被抢占 42 个请求、70 次事件，最多 3 次；70 次抢占分布在 70 个不同 step（每 step 最多 1 个），恢复等待 P50 349 ms / 最长 2.63 s。

### 2.4 `prompt_tokens_total` 到底数什么

- 计数点：[stats.py:114-118](vllm/v1/metrics/stats.py#L114-L118) `if is_prefilling: self.num_prompt_tokens += prompt_len`；`is_prefilling` 初值 True（[output_processor.py:115](vllm/v1/engine/output_processor.py#L115)），**每次有输出就置 False**（[output_processor.py:414](vllm/v1/engine/output_processor.py#L414)）。
- 因此它数的是"**请求数 × 该请求的 prompt 长度**"，与被抢占/重算完全无关。实测：Δ 53,292 = trace 里 103 条被测请求 `prompt_tokens` 之和（517.4 平均，含 chat template 开销），而 warm-up 16 条 = 1,424（41×8 + 137×8）= pre 快照值，两项**精确闭合**。
- 所以 **计数器不能反推冗余计算**：`prompt 53,292 + gen 52,736 = 106,028`，而真实被调度位置 153,265 / 唯一 107,845，差的就是那 45,420（29.6%）。**Day4 的"token 计数看不到重算"这一反常，到这里定案：是口径问题，不是抢占没发生。**

### 2.5 trace 完整性

- 59,192 行全部可解析；单一 `pid=9800`（= `servers/kv_512.log` 里的 `EngineCore_DP0 pid=9800`）；`step` 1…5858 严格单调、无重置（排除"第二次 server 写同一文件"的污染，虽然 nsys run 的环境里也带着 `W18_SCHED_TRACE_FILE`）。
- 事件数 59,192 < `W18_SCHED_TRACE_MAX_EVENTS` 默认 100,000 ⇒ 未触发截断；无 GPU 同步、无逐请求 INFO 日志，只在 `_update_after_schedule` 前写一次盘。**打点自身开销未量化**：没有同 workload（P512/O512 + 512 blocks）的无 patch 对照，base 两臂是 P128/O128 + 4096 blocks，不能拿来当基线。
- 覆盖：被测窗口的 **103 条请求全部出现**（100%），另加启动后的 16 条 warm-up（共 119 条），因此统计口径里的"全部请求"包含 warm-up；计数器的 pre 快照正好等于这 16 条之和，可用于对齐窗口边界。

## 3. 主机 / GPU 链

- **step 的源码顺序**：`Scheduler.schedule()`（[scheduler.py:177](vllm/v1/core/sched/scheduler.py#L177)）产出 `SchedulerOutput` → worker 侧 `GPUModelRunner.execute_model()`（[gpu_model_runner.py:2000](vllm/v1/worker/gpu_model_runner.py#L2000)）：`_update_states` → `prepare_inputs_event.synchronize()`（**上一步的 CPU buffer 复用点**，[gpu_model_runner.py:2019-2030](vllm/v1/worker/gpu_model_runner.py#L2019-L2030)）→ `_prepare_inputs` / `_preprocess` → `BatchDescriptor(num_tokens, uniform_decode)` → `cudagraph_dispatcher.dispatch()`（[gpu_model_runner.py:2047-2050](vllm/v1/worker/gpu_model_runner.py#L2047-L2050)）→ `model.forward` → sampler → `ModelRunnerOutput` → 回到 engine 侧 `Scheduler.update_from_output()`（[scheduler.py:862](vllm/v1/core/sched/scheduler.py#L862)）。
- **CUDA Graph 与 step 时间的"量化"**：`num_input_tokens` 会被 `pad_for_cudagraph` 补齐到 67 个 capture size 之一（[gpu_model_runner.py:1745-1750](vllm/v1/worker/gpu_model_runner.py#L1745-L1750)、[:1331](vllm/v1/worker/gpu_model_runner.py#L1331)），PIECEWISE 下 `unified_attention` 被排除在 graph 外（`splitting_ops` 里有它）⇒ 每 step = 若干 graph replay + 28 个 eager attention。这解释了 nsys 的 37,526 次 `cudaGraphLaunch`（≈23 次/step）与 182,418 次 `cudaLaunchKernel` + 96,910 次 `cuLaunchKernel`（≈113 + 60 次/step）并存。
- **nsys 的进程与 NVTX**：`PROCESS_0` = `vllm serve … --max-num-seqs 256 --num-gpu-blocks-override 4096`；采集用 `--sample=none --cpuctxsw=none`，`COLLECT_THREAD_STATE_TRACE=false` ⇒ **没有任何 CPU 线程态/采样可归因**，而 `NVTX_EVENTS` 只有 1 条 `"CCCL"` ⇒ **没有任何 vLLM range 可以和 kernel 对齐**。"哪个 kernel 属于 prefill/decode/sampler"只能按 kernel 形状 + 每 step 计数推断（attention 45,808 ÷ 1,619 step = **28.3 个/step**，与 28 层吻合，多出的 476 个来自 chunked prefill 边界；`reduce_kernel`=1,619 与 `reshape_and_cache`=45,332=28×1,619 独立复核）。
- **GPU / API / wall 三种时间的关系**（本窗口 22.71 s）：

| 口径 | 值 | 性质 |
|---|---|---|
| GPU kernel busy | **16.237 s（71.5%）**，平均 10.03 ms/step | 设备真正在跑的时间，可加总但**不等于** step 时间（step P50 13.10 ms） |
| CUDA API 总和 | **≈8.82 s（38.8%）** | CPU 在 API 里花的**墙钟**，与 GPU 时间**并行重叠**，不能相加 |
| `cudaEventSynchronize` | **5.78 s（65.6% API 时间）**，1,619 次 = 3.57 ms/step | 是"**等**"，CPU 阻塞在同步点；次数与 step 数精确相等 |
| kernel 间隙 | **4.805 s（21.2%，1,614 个 >1 ms）** = 2.97 ms/step | 设备空闲，集中在 step 边界；kernel 之间 P99 仅 59 µs |

- **异步重叠的坑**：`cudaLaunchKernel` 平均 5.9 µs、`cudaGraphLaunch` 33.1 µs 都是**提交**耗时，GPU 侧真实执行在其后；`cudaEventSynchronize` 的 3.57 ms/step 说明引擎在 step 末**等结果**（async output proc + 下一步的 `prepare_inputs_event` 复用），所以"CPU 慢"与"GPU 慢"在这个窗口里无法用 wall clock 区分——只能用"kernel busy 71.5% + 边界空隙 21.2%（2.97 ms/step ≈ 3.57 ms 的等待）"这个组合说"**空白在 step 边界，不在 kernel 之间**"。
- **重做的这条 nsys run 是有效样本**：`admission256_nsys_client` 3,839/3,839 全 200、0 超时、0 drop、`metrics_errors=0`、`metrics_post.txt` 44.9 KB；吞吐 11,596.8 tok/s 相对 BASELINE 256 的 12,168.5 是 **-4.7%**，但 rate 是 95 vs 100，**11,596.8/(95/100)=12,207 ≈ 12,168.5** ⇒ 吞吐由需求决定，看不出 profiler 代价；真正可测的是尾部退化（TPOT P99 19.34 vs 16.81 ms、E2E P99 +14%、waiting 峰 3 vs 0）。

## 4. Observed vs inferred vs unknown

**Observed**
- 0.10.2 / PIECEWISE / 67 capture sizes / 4,096 blocks（admission 两臂）与 512 blocks（KV 臂）；CLI 与 `num_gpu_blocks` 逐臂可查。
- trace：59,192 事件、5,858 step、70 个 `preempt`、Σ`previously_computed`=45,420、重复位置 45,420（29.6%）、唯一 107,845、119 请求、单 pid、无截断。
- 计数器：`num_preemptions` 0→70；`prompt_tokens` 1,424→54,716（Δ 53,292 = trace 103 条 prompt 之和）；`generation_tokens` 512→53,248（Δ 52,736 = 103×512）。
- nsys（`admission_256_delay90_dur60`）：772,274 kernel / 33 个名字、1,619 个 `reduce_kernel`、37,526 `cudaGraphLaunch`、182,418 `cudaLaunchKernel`、1,619 `cudaEventSynchronize`/5.78 s、71.5% duty、1,614 个 >1 ms 空隙（4.805 s）、NVTX 仅 1 条；窗口 13:57:17.02–13:57:39.73Z（22.71 s）；`START_DELAY_MS=0`、`RUN_DURATION_MS=60000` 与 22.71 s 不符；同 run 客户端 3,839/3,839 全 200。

**Inferred**
- 抢占代价是"从 0 重算到当前 token 数"，因为 `num_computed_tokens=0` + prefix cache 关闭 + trace `start=0`，三条一致。
- 45,420 个重复位置与 Σ`previously_computed` 相等 ⇒ trace 的区间统计无重复计数、也无遗漏来源。
- 1,619 step（`reduce_kernel` 计数、`reshape_and_cache`=28×1,619、窗口内 gen token÷1,619=169 ≈ running 均值 175，三条互证）；约 23 次 graph launch/step 来自 PIECEWISE 的分段。
- nsys 窗口是从 probe 中段开始的手动开窗（起点在 probe 内 18.4 s、终点与 probe 结束对齐），所以报告里的 `RUN_DURATION_MS` 只是请求值、`stop` 提前结束。

**Unknown**
- patch 自身的开销（每 step 写盘 + `monotonic_ns`）：没有同 workload 的无 patch 对照，`kv512` 的 4.88 ms/step 里有多少是打点成本。
- 抢占 victim 选择在真实 trace 里的可预测性（FCFS 弹队尾 vs `priority` 策略），本 run 只用默认策略。
- 打开 prefix cache 后 `get_computed_blocks()` 能救回多少（本轮命中恒 0）。
- 无 CPU 采样 / 无 NVTX 的情况下，28.5% 空闲里"主机准备"与"等 GPU 完成"的比例。
- `--cuda-graph-trace=node` 对 graph 段执行时间与 API 时间的定量扰动（缺同 workload、同 seed、无 profiler 的对照）。
