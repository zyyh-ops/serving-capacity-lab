## 第 1 题：CPU/GPU 时间边界

**不能相加。** 这四个数不是同一层、同一时钟下的互斥区间；直接把四个值相加会重复计数。按题面，串行 EngineCore 的 Step Duration 约为：

```text
1.3 + 4.8 ≈ 6.1 ms（再加输出/通信）
```

而不是 `1.3+4.8+3.9+0.08=10.08 ms`。

| 数值    | 实际口径                                                     | 重叠关系                                                     |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.3 ms  | `Scheduler.schedule()` 的 **CPU wall**，只包含 Python、KV 分配和队列扫描 | 在 async/多流下可与上一步 GPU 尾部重叠；同步模式下与 `execute_model` 串行 |
| 4.8 ms  | `execute_model()` 的 **Python wall**：preprocess + launch + sample + bookkeeping | 包含 0.08；默认在 `_bookkeeping_sync`→`_to_list()` 的 `transfer_event.synchronize()` 等 GPU，所以通常已包含 3.9 的主体 |
| 3.9 ms  | GPU timeline 上 kernel 的占用或首末跨度                      | 与 4.8 内的 CPU 预处理/launch 并行，末尾被 D2H sync 等待；多流 kernel 会互相重叠 |
| 0.08 ms | `cudaGraphLaunch` 的 **CPU API 时间**，异步提交              | 是 4.8 的子区间；只是单次提交时间，不代表图内 kernel 执行    |

**重叠关系**：`0.08 ⊂ 4.8`；`3.9` 与 4.8 的 CPU 部分并行，但在 `_to_list()` 的 D2H sync 处被等待；一步若有多次 graph launch，Host 总 launch 时间是多次之和，仍不能与 GPU 时间相加。若 `3.9` 是“kernel duration 之和”，多流重叠会重复计算；若它是“首末跨度”，又包含 kernel 间隙，要先明确 nsys 查询口径。

**为什么 `4.8 > 3.9` 不能直接推出 Host 瓶颈**：Python wall 里可能大部分是 **CPU 在等 GPU**（`cudaEventSynchronize`），不是 CPU 在干活。证明 Host 是主要瓶颈还需要：

1. **nsys 时序**：GPU 有 gap，且 gap 与 CPU 的 `cudaLaunchKernel`/`cudaGraphLaunch`/Python 区间对齐；下一 kernel 要等 Host 提交。计算 `GPU busy = Σkernel/窗口`，明显小于 1。
2. **CPU 关键路径**：`schedule + preprocess + launch + bookkeeping sync + output` 本身就是限制，且 CPU 单核接近 100%（py-spy/perf）；同时把“CPU busy”与“CPU 阻塞在同步”分开。
3. **反事实实验**：CUDA Graph/减少 launch/绑核后 step 明显下降，而 kernel 数和总 kernel 时间几乎不变；或增加 GPU 工作量（更多 token）step 不变，直到 GPU 追上。
4. **排除饥饿**：`waiting=0`、KV 分配成功、无网络/D2H 阻塞；否则低 GPU busy 是“没活干”，不是 Host 慢。

## 第 2 题：精确统计 Recompute

| 轮次       | 执行区间               |          Token 次数 |
| ---------- | ---------------------- | ------------------: |
| 初始       | `[0,640)`              |                 640 |
| 第一次恢复 | `[0,512)`、`[512,700)` | 512 + 188 = **700** |
| 第二次恢复 | `[0,600)`、`[600,760)` | 600 + 160 = **760** |
| **合计**   |                        |            **2100** |

- **总执行 Token 次数 = 2100**
- **最终唯一逻辑位置 = `[0,760)` = 760**
- **重复执行量 = 2100 − 760 = 1340**

更直观地看：位置 `0–639` 执行 3 次，`640–699` 执行 2 次，`700–759` 执行 1 次。被重复覆盖的位置有 **700 个**（0–699），额外执行次数是 **1340**。

**为什么不是 `preemptions_delta × prompt_length`**：这里 `preemptions_delta=2`，若 `prompt_length=760`，则 `2×760=1520`；若按初始 640，则 `2×640=1280`，都不等于 1340。原因是：

- 每次重算量 = **被抢时已经算过的 prefix**，第一次是 640，第二次是 700，不是固定 prompt 长度；
- 重算包含已生成的 token（若有），不只是 prompt；
- Prefix Cache 命中会减少重算，甚至为 0；
- Counter 只数“抢占事件”，不数 token 位置。

源码上，抢占时 `kv_cache_manager.free()`、`num_computed_tokens=0`、请求插回 waiting 队首；恢复时 `num_new_tokens = request.num_tokens - num_computed_tokens`，`resumed_from_preemption=True` 只标记这一步是恢复。

## 第 3 题：GPU Compute、HBM 与 Host 瓶颈

| 竞争假设                          | 机制/对应                                               | 最小验证实验                                                 | 判据                                                 |
| --------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **H1 GPU compute 受限**           | A 中 kernel 连续，Tensor Core/SM 可能饱和               | 有 `ncu` 权限时抓一步 decode：`sm__pipe_tensor_op_hmma_cycles_active`、`sm__throughput`、`sm__warps_active`；或放大 batch/token，看 step 是否线性增长 | Tensor Core/SM 利用率高，step ∝ token 数             |
| **H2 HBM/KV 带宽受限**            | A 中 kernel 连续，但卡在 DRAM/L2，SM 在等数据           | `ncu` 抓 `dram__throughput`、`lts__throughput`、`dram__bytes`；改变 context 长度/batch，比较 bytes 与时间 | DRAM 接近峰值，step ∝ 读写字节数                     |
| **H3 Host launch/scheduler 受限** | B 中 kernel 间有空档，CPU 的 `schedule()`/launch 跟不上 | nsys 对照 CUDA API 行与 GPU 行，算 gap；`--enforce-eager` vs CUDA Graph；绑核/降 Python 开销 | gap 与 CPU 区间对齐；Graph 后 step 降、kernel 不变   |
| **H4 Admission/饥饿**             | B 中 GPU 空档是因为没有可发的请求，不是 CPU 慢          | 看 `waiting/running/KV/preemptions`；抬高 `max_num_seqs` 或 KV blocks | waiting>0、running 顶 cap；抬高后 GPU 变忙、吞吐上升 |

补充 H5：每步 D2H 采样同步也可能造成空档；测 `_bookkeeping_sync`/`transfer_event.synchronize` 的耗时，开 async scheduling 对比。

**高 GPU Busy ≠ Compute-bound**：Busy 只说明“有 kernel 驻留”，可能是 HBM-bound、低 occupancy、大量小 kernel/copy，甚至 kernel 在等内存；Tensor Core 利用率未知。连续 timeline 也可能是许多依赖小 kernel 串行，并不等于算力饱和。

**低 GPU Busy ≠ CPU-bound**：可能是没活可发（admission/KV 分配失败/waiting）、CPU 在等 D2H/网络、kernel 依赖串行、GPU 降频，CPU 自己也可能空闲。要证明 CPU-bound，必须看到 CPU 关键路径饱和、GPU gap 与 CPU 工作对齐，并用“减少 CPU 工作后 gap 消失、吞吐上升”的反事实确认。B 中“Scheduler 耗时与 gap 相关”只是相关性，还要确认 gap 发生在 schedule 期间，而不是 schedule 在等 GPU。

## 第 4 题：干净的 Admission A/B

**消除了哪些混杂**：

- 两臂都固定 `--num-gpu-blocks-override 4096`，**KV 可用块数/容量**这个 Day4 最大混杂消失；
- 两臂 `Preemptions=0`，排除抢占/recompute 差异；
- Workload、seed、model、graph、`max_num_batched_tokens` 固定，吞吐差异只剩 admission cap。

**仍存在哪些混杂**：

- `max_num_seqs` 仍会改变 `profile_run` 的虚拟 batch、activation 峰值和物理显存占用；`--num-gpu-blocks-override` 只限制“可用 KV 块”，不保证两臂真实预留内存相等；
- 4096 blocks 是人为上限，可能本身成为约束；结论只适用于这个 KV 容量；
- 单 seed、单 30 s 窗口、Poisson 噪声、A→B 顺序、热/降频、warmup 差异；
- P99 只是尾指标，不等于 joint goodput；TTFT P99 210 ms 仍高于 200 ms，TPOT P99 18 ms 仍高于 15 ms，尾延迟仍在 SLO 线外；
- 客户端口径与服务端 `/metrics` 口径、drain cohort 是否计入，都会影响 goodput。

**是否证明扩大 Admission Limit 提高 SLO 容量？——没有。** 这组数据只证明：

- 在 λ=100 这个点上，256 的 cap 导致 `waiting=20`，512 把队列消掉，TTFT P99 从 280 降到 210；
- 但 output throughput 只从 11,800 到 11,950（**+1.3%**，接近噪声），TPOT P99 **18/18 完全不变**；
- 这说明瓶颈在 decode 计算/每 token 成本，不在 admission；512 只是把排队从 waiting 挪进 batch，没有提高“单位时间满足 SLO 的请求数”。

**SLO 容量**应定义为“在给定 joint SLO 下能稳定承载的最大 offered λ”，单点 A/B 不够。决策应这样做：

1. 定义 **joint goodput**：`TTFT ≤ 200 ms 且 TPOT ≤ 15 ms` 的请求数 / 统计窗口；
2. 对 256/512 两臂扫 λ（如 80/90/100/110/120），每臂至少 3 个 seed，AB/BA 交换顺序，固定长度、warmup、KV/graph 配置；
3. 画 **goodput vs λ**（以及 vs 实际吞吐），看曲线 knee 是否右移；P99 只作 guardrail；
4. 只有当 512 在相同 λ 下可重复提高 joint goodput、且 TPOT 尾不退化时，才采用 512；否则保留 256。若 TPOT 是 binding constraint，应优先优化 decode（batch、attention/KV、量化、SLO-aware 调度），而不是继续加 cap。

## 第 5 题：Preemption Counter 与实际计算工作量

**两个 Counter 不能直接推算重算量：**

- `num_preemptions_total` 只在每次抢占事件上加 1，没有 request id、没有 token 数、没有位置；
- `vllm:prompt_tokens_total` 是**每个请求首次产出时加一次 `prompt_len`**，不是每 step 的 scheduled token，也不计 re-prefill；
- 因此 `480×520=249,600` 只是“480 个请求 × 名义 prompt 长度”。若 400 次抢占每次重算 520，应该多 `400×520=208,000`，但 `prompt_tokens_total` 根本不会变；
- 每次抢占的真实代价 = `被抢时 num_computed_tokens − 恢复时 prefix-cache 命中数`，可能 0 到几百 token，随请求长度、命中、是否反复抢占而变。

**基于 `[start,end)` 的统计方法：**

```text
每步、每请求：
  start = SchedulerOutput 里的 num_computed_tokens（调度前）
  end   = start + num_scheduled_tokens[req_id]
  diff[start] += 1
  diff[end]   -= 1

按 req_id 分开，扫描 0..num_tokens：
  cur += diff[i]
  unique   += (cur > 0)
  repeated += max(0, cur - 1)
```

- `total = Σ(end−start)`
- `unique = |∪[start,end)|`
- `repeated = total − unique`
- 位置被执行 k 次，贡献 `k−1` 次重复；多次抢占的所有轮次都放进同一张 per-request 覆盖图，**不要在 resume 时清零**。
- 若要算“每次 resume 的重算”：记录被抢时 snapshot `preempt_at`，恢复时 `resume_start`，该次重算 = `preempt_at − resume_start`；新增 token 不算重算。

**三类误差：**

1. **Prefix Cache**：恢复时 `num_computed_tokens` 已经是命中后的起点。若 trace 仍从 0 记 `[0,end)`，会把已缓存的前缀算成重算；别的请求命中的共享前缀也不应算作本请求的重算。必须记录 `num_cached_tokens`/prefix hit。
2. **Speculative Decoding**：`num_scheduled_tokens` 包含 draft token，但被拒绝的 draft 会在 `update_from_output` 里从 `num_computed_tokens` 减掉。在 schedule 时记 `[start,end)` 会把 rejected draft 算成已执行；应使用 step 后的 `num_computed_tokens`，或减去 `num_rejected`。
3. **异步未完成 Step**：`_update_after_schedule` 在真正执行前就把 `num_computed_tokens` 往前推；async 模式下 `execute_model` 可能先返回，D2H 在 `get_output()` 才完成。trace 在 schedule 时写，会把“已提交未完成”甚至“失败/被 abort”的 step 算进去；只统计有完成 ack 的 step，并在窗口结束时处理 in-flight。

另外，chunked prefill 的部分区间、请求 abort/finish、CUDA Graph padding、多 engine/DP 聚合、trace `MAX_EVENTS` 截断，也都会让统计偏高或偏低；报告时要说明是“逻辑调度位置”的下界，不是 GPU FLOPs。
