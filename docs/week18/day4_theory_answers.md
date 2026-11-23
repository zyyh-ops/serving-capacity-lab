## 第 1 题：不是成功优化

- 这是**放宽准入**，不是提高容量：`max_num_seqs` 只是把 256→512，Waiting 掉到 5 是因为请求被塞进 Running（256→390），**排队的延迟变成了 batch 内的 TPOT 延迟**。
- 吞吐 90→91 req/s（+1%）已到平台，而 TPOT P99 20→25 ms（+25%）越过 15 ms SLO；offered 100 > completed 91，积压仍在增长。
- 判据应是 **joint goodput（TTFT≤200 ms 且 TPOT≤15 ms）随 offered λ 的曲线**，不是 Waiting 峰值。
- 验证实验：固定 100 req/s 开环、固定 seed/长度，只扫 `max_num_seqs∈{256,320,390,512}`，同时记录 P99 TTFT/TPOT、goodput、KV、preemption、每步 batch/token 数与 GPU 忙闲。若吞吐不涨而 TPOT 单调升，就是 compute/KV 受限，准入参数无收益。

## 第 2 题：57 / 456 / 56，不能全继续

- 每请求 `ceil(900/16)=57` blocks（912 slots，浪费 12）；8 个共 **456**；剩余 `512−456=`**56**。
- 再加 128 token：`900+128=1028 → ceil=65`，每请求多 **8** 块，合计再要 **64 > 56**（总量 520 > 512，缺 8 块）。
- 所以**不能**：块是逐步分配的（解码跨过 912/928/…/1024 边界时各要一块），池子耗尽那一步 `allocate_slots()` 返回 `None`（[kv_cache_manager.py](vllm/v1/core/kv_cache_manager.py#L270-L272)），必须抢占。最多 7 个请求能走完这 128 token，第 8 个被 preempt。
- 真实 V1 还要算多 KV group、null block、watermark、block 对齐浪费与 spec lookahead，57 是下界。

## 第 3 题：FCFS 抢占与重算

- **选 victim**：`allocate_slots` 失败进入 `while True`（[scheduler.py](vllm/v1/core/sched/scheduler.py#L252-L290)）。默认 FCFS 直接 `self.running.pop()`——队尾、即最新到达的请求；`PRIORITY` 策略取 `max(priority, arrival_time)`。循环直到分配成功，或 victim 就是自己（`can_schedule=False`，本步不调度）。
- **释放**：`kv_cache_manager.free()` → 各 group `free()` → `block_pool.free_blocks()`：`ref_cnt--`，归零的块按**尾块优先**回 free queue（[block_pool.py](vllm/v1/core/block_pool.py#L252-L267)），便于 prefix cache 先淘汰尾部；同时 `encoder_cache_manager.free()`。
- **状态**：`status=PREEMPTED`、`num_computed_tokens=0`、记事件、`waiting.prepend_request()` 插回**队首**（[scheduler.py](vllm/v1/core/sched/scheduler.py#L271-L280)）。V1 只有重算，没有 swap。
- **恢复**：下次从 waiting 队首取出，因 `num_computed_tokens==0` 会重新 `get_computed_blocks()` 查前缀缓存（[scheduler.py](vllm/v1/core/sched/scheduler.py#L378-L382)）；`num_new_tokens = request.num_tokens − num_computed_tokens`（含已生成 token，[scheduler.py](vllm/v1/core/sched/scheduler.py#L418-L421)）→ **全量 re-prefill**。进 `scheduled_resumed_reqs`，`resumed_from_preemption=True`，worker 整表替换 block_ids；进度只在 `_update_after_schedule` 里加（[scheduler.py](vllm/v1/core/sched/scheduler.py#L646)），保证本步 SchedulerOutput 带的是调度前的值。
- **为何 `preemptions_delta=10` 不能量化重算**：counter 只数事件（[loggers.py](vllm/v1/metrics/loggers.py#L270-L275)），每次代价 = 被抢时 `num_tokens` − 恢复时 prefix-cache 命中数，随请求长度、命中与否、是否被反复抢占而变；可能命中自己的前缀几乎零重算，也可能全量重算。要量化必须用 `resumed_from_preemption` + 恢复时 `num_computed_tokens` + 该次 scheduled tokens 逐请求统计。

## 第 4 题：不能断言 B 更多

- 绝对量：A = 1024×0.65 ≈ **666** blocks，B = 512×0.92 ≈ **471** blocks，**A 更多**。Usage 是“已用/总量”的比例，分母不同不可直接比；且 KV 显存按 `num_gpu_blocks` 预分配，A 预留得更多。
- 40 次 preemption 只说明 B **更紧**（512 块粒度粗，1 block≈0.2%，且更易打满），不说明占用更多。
- 区分四类现象：
  - 真实增长：同一进程内 `num_preemptions_total` 单调上升，且与 KV≈1、Running 平台、Waiting 上升同窗口对齐；
  - Counter reset：数值回退、或 `_created` 时间戳跳到进程启动时刻（multiprocess 导出的 `vllm:num_preemptions_created`）；
  - 指标缺失：抓取空档/NaN/字段不存在，看样本数与 `_created`，别把缺失当 0；
  - 采样误差：0.5 s 采样会漏掉瞬间峰值——若 `usage<1` 却有 preemption，即证明峰值被漏采。
- 需联合观察：`num_preemptions_total` 速率、高频 `kv_cache_usage_perc`、`num_requests_running/waiting`、`prompt_tokens_total`/`generation_tokens_total`（流量归一）、`request_success_total`、`prefix_cache_hits/queries`、block 总数配置，以及逐请求 trace（preempt/resume 与重算 token 数）。

## 第 5 题：三个竞争假设 + 最小 profiling

假设：
1. **GPU compute 受限**：forward 已饱和，加并发只加 batch → 吞吐平、TPOT 升（step time ∝ token 数）。
2. **Host 调度/launch 受限**：Python 调度 + kernel launch 间隙造成 GPU 气泡，吞吐平但 GPU 不忙。
3. **准入/调度策略受限**：A 的 Waiting=100 而 KV 仅 40%、Running 卡在 256 → `max_num_seqs` 卡的，不是算力；B 的 KV 65%、Running 400 → 转为 KV/带宽受限（可能伴随抢占）。

最小 profiling：
- 引擎侧每步打点：`schedule()` 耗时 vs `execute_model` vs sample；
- nsys/DCGM 看 GPU busy% 与 kernel 间隙：忙≈100% 且 step∝token → compute；忙低且有 gap → host/launch；
- 固定 λ 做两组对照：`max_num_seqs` 拉平（都 256）与改 `max_num_batched_tokens`，看吞吐/TPOT 是否变化；再看 KV/preemption 判断是否容量受限。

为什么 B 仍不算有效容量优化：容量是**满足 SLO 的最大 λ**。B 的 TTFT 500 ms > 200、TPOT 25 ms > 15，两项都不达标；goodput 仍 20 req/s，说明有效容量没动，只是把排队换成了更长的 decode 和更高 KV 压力（65%），余量更小。
