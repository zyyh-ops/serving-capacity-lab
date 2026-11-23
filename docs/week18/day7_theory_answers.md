# Week 18 Day 7

## 第 1 题：SLO Capacity 的跨实验复核

**不能证明 90 req/s 是最大稳定 SLO Capacity。**

- A 的 90 点：21,545 条中 663 条 TPOT>15 ms（3.08%，Wilson 95% CI 2.85%–3.32%），240 s 内已经越线；80 点 19,082/19,082 全达标。故 A 只支持“80 安全、90 边界”。
- B 的 90 点：3,588/3,588 全达标，但只有 40 s、单 seed、KV 固定 4096、prefix cache off；0/3,588 的 95% 上界约 0.11%，仍不能外推到 240 s 或生产。
- 100 点（A，30 s）goodput 18.03、TPOT p50 18.87 ms；105 点（B，25 s）goodput 19.04 ⇒ 拐点在 90–100 之间，105 已确认过载，95 未做干净容量点。

**不能合并成一条曲线**：两组的 KV blocks、prefix cache、窗口长度（240 s vs 40/25 s）和指标口径（arrival-cohort vs completion-window）都不同；同一个 x=90 有两个 y（A 3.1% 违约、B 0 违约），已经不是一条函数。源码上，prefix cache 改变 `get_computed_blocks()` 的命中（`kv_cache_manager.py:168-190`），KV blocks 改变 `allocate_slots()` 是否返回 None 及是否触发抢占（`scheduler.py:252-284`、`kv_cache_manager.py:270-272`），同一 λ 下的 running/KV/step time 因此不同。

**最小认证方案**：

1. **锁配置**：vLLM 0.10.2、Qwen3-0.6B、RTX 4090、max_model_len 2048、max_num_seqs 256、max_num_batched_tokens 2048、gpu_mem_util 0.7、`--num-gpu-blocks-override 4096`、prefix cache off（或每轮 reset）、chunked prefill on、cudagraph piecewise、temperature 0、ignore_eos、P128/O128（另加 P512/O128 混合）。
2. **点与重复**：80/85/90/95/100，3 seeds（2027–2029），每点 ≥300 s（95/100 用 600 s）；每 seed 用同一 Poisson 计划，AB/BA 顺序，arm 间重启 server；16 条 warmup 后丢弃前 10 s。
3. **指标**：逐请求 TTFT/TPOT；主口径 arrival-cohort（含 drain）joint SLO，辅口径 completion-window goodput；同时报违约率、P99 的 per-seed bootstrap CI、违约率的 Wilson CI（90 点 300 s 约 2.7 万条，可把 3% 的 CI 压到 ±0.2% 内）。
4. **决策规则**：λ* = 满足“每个 seed joint SLO ≥99%、违约率 95% 上界 ≤1%、preemptions=0、KV p95<80%、waiting p95<1 s、恢复<30 s”的最大 λ；生产容量取 **0.8λ***。若 95 通过而 100 失败→λ*=95；95 失败→λ*≤90；90 在 300 s 仍有 >1% 违约→λ*=80（补测 85）。105 只做过载/恢复测试，不进入容量曲线。

## 第 2 题：KV Preemption 与冗余计算

- **重复调度率** = 45,420 / 153,265 = **29.63%**。
- **调度放大倍数** = 153,265 / 107,845 = **1.421×**（相对名义 107,964 为 1.420×）。

**唯一位置 = 107,964 − 119**：prefix cache off 时每条请求都从位置 0 开始；最后一个生成的 token 只产出、不再作为输入被调度，所以每条请求贡献 `num_tokens−1` 个唯一位置，119 条共少 119。成立前提：prefix cache off、无 spec decode、按请求计唯一位置（不做跨请求去重）、trace 覆盖完整生命周期（本 run 59,192 事件 < 100k、step 1..5858 单调）、每条请求至少有 1 个 output token。

**29.6% 不能直接写成 GPU FLOPs 浪费比例**：它数的是“逻辑位置次数”，不是 FLOPs/时间。prefill 一个位置与 decode 一个位置的 FLOPs 和带宽差几个数量级；attention 随上下文长度增长；chunked prefill、batch、cudagraph 让同一位置的实际成本不同。45,420 恰好等于 Σ`previously_computed`，是调度冗余量，不是 GPU 时间。`prompt_tokens_total` 只在首次 prefill 计一次（`stats.py:117-118`、`loggers.py:540-541`；实测 Δ=53,292=103×prompt），看不到重算；`preemptions × prompt_len` 也不对，因为每次抢占的 `previously_computed` 不同（例：567/590/638），且 70 次分散在 42 条请求上。

**开 Prefix Cache 后**：抢占后块可能仍在缓存，`get_computed_blocks()` 命中后 `start>0`（`kv_cache_manager.py:179-190`、`scheduler.py:378-382`）。要记录 `num_cached_tokens`/hit 长度，只把 `[start, start+n)` 中未命中的部分计为“实际计算”；重复位置按每请求已执行区间的交集算；跨请求共享的前缀不能按请求重复计。建议同时报“唯一逻辑位置（数据）”和“唯一实际计算位置（算力）”两套数。

**开异步调度后**：`_update_after_schedule()` 在 schedule 时就把 `num_computed_tokens` 前移（`scheduler.py:643-646`），GPU 可能尚未执行。要把 scheduler step 与 model runner 实际执行的 step 关联，只对 `execute_model` 确认执行的 token 计数；排除 abort/未执行就被抢占的请求；用聚合计数并确保退出前 drain，避免 100k 事件截断或丢事件。

## 第 3 题：Nsight 因果归因与时间线陷阱

**不能证明 CPU Scheduler 是主瓶颈。**

- GPU busy 71.5% 只说明“有 kernel 在跑”，28.5% 是残差，不是 CPU 归因；22.71 s 窗口还缺 ramp-up、含 drain，不是稳态。
- `cudaEventSynchronize` 1,619 次/5.78 s 恰好每 step 一次，对应 `_to_list()` 的 pinned copy + event sync（`gpu_model_runner.py:3730-3742`）：host 在等采样 token 从 GPU 拷回。这段等待与 GPU 计算重叠，不能和 4.805 s 的 GPU gap 相加，否则重复计算。
- 4.805 s 的 >1 ms gap 集中在 step 边界（1,614 个 ≈ 1,619 step；kernel 间 gap P99 仅 59 µs），形态与“每 step 一次同步/准备/launch”一致；但同步本身是 GPU 依赖，不是 CPU 工作量，gap 是 host 造成还是 GPU 尾部，需要 host 侧时间线。
- 没有 vLLM NVTX range、没有 CPU 线程状态采样（`--sample=none --cpuctxsw=none`），无法定位 scheduler/Python/SSE 的时间；attention 60.8% + GEMM 30.9% 只说明 GPU kernel 时间占 91.7%，不能用它证明 host 是主因。
- `--cuda-graph-trace=node` 把 graph replay 展开成逐 node（37,526 次 `cudaGraphLaunch`、约 196 launch/step），API 时间被放大；不完整窗口也会让“稳态”结论失真。

**竞争假设与最小实验**：

| 假设 | 预测 | 最小实验 |
|---|---|---|
| H1 Host launch/step 串行 | host busy 与 gap 同增；注入延迟 1:1 | 在 `schedule()` 注入 0.5/1/2 ms；async scheduling on/off、enforce-eager on/off 对照 |
| H2 GPU compute | step 时间随 batch/FLOPs 增长，SM 忙 | max_num_seqs 128/256/512，固定 λ；ncu 看 SM/tensor 利用率 |
| H3 KV memory BW | attention 时间随 KV 长度/占用增长，DRAM 近峰 | P128 vs P512、KV blocks 512/1024/4096，同 token/s；ncu DRAM 计数器 |
| H4 请求饥饿 | waiting=0、running<cap，gap 随到达间隔 | λ=50/80/95/100 sweep，看 GPU busy 是否随负载上升 |
| H5 前端/采样 CPU | host 时间在 SSE/detokenize/sampler | NVTX 包住 schedule/execute/sample/detokenize + py-spy/perf |

**最小 Nsight/CPU 方案**：同配置同 seed，warmup 后取 60 s 稳态；`nsys profile --trace=cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree --cuda-graph-trace=graph`；在 `Scheduler.schedule`、`GPUModelRunner.execute_model`、sampler、detokenizer/SSE 打 NVTX；导出 sqlite 后按 step 对齐 host start/launch/first-kernel/last-kernel，分别算 host busy、host blocked、GPU busy、gap。用 profiler on/off 同 seed 量化扰动；`node` 只做单独对照。

## 第 4 题：Admission Control 与双分母决策

**25 s 过载（Offered 2,601）**：

| 指标 | Open | RPS Gate |
|---|---:|---:|
| Offered-level 达标 | 441/2,601 = **16.95%** | 1,967/2,601 = **75.62%** |
| Gate 拒绝率 | 0 | 634/2,601 = **24.37%** |
| Accepted-level | 16.95% | **100%** |
| Completion-window Goodput | 19.04 | **77.52** |

净增 good = 1,526，goodput ×4.07。注意 441/25=17.64 ≠ 19.04，1,967/25=78.68 ≠ 77.52：窗口 goodput 与 arrival-cohort 不是同一批请求，必须分开报。

**为什么过载 Gate 有收益、混合 Cost Gate 净损失**：

- 105 req/s 已超过 SLO 服务能力（running 253/256、TPOT p50 16.93 ms>15、KV 0.799）；边际请求的达标概率只有约 17%，拒绝它相当于把容量让给能达标的请求，所以 offered-level good 从 17.0% 升到 75.6%。代价是 24.4% 需求被丢、offered 兑现率只有 75.6%。
- 60 req/s 混合负载没有过载：Open 3,588/3,588 全达标、waiting=0、goodput 59.08。Cost Gate 仍拒 472（13.2%），goodput 降到 51.22（−13.3%）；这是配额算术（60×(0.75×384+0.25×768)=28,800/s > 24,000/s），不是容量判决。且 `P+2O` 把 long 高估：long 拒 27.8% vs short 8.5%（3.3×），而实测 TPOT p99 6.89 vs 6.93、TTFT 只差 4–5 ms。结论：admission 只在 λ>μ_SLO 时创造价值；有余量时拒绝 = 净损失 + 不公平。

**`local_reject` ≠ HTTP 429**：`local_reject` 在 `session.post()` 之前发生，请求从未到 vLLM；server 没有 429、没有 `Retry-After`、没有错误率/可用性记账。真实 429 是服务端拒绝，计入可用性/错误预算，客户端会重试/退避；local reject 把丢弃隐藏在客户端，还会造成重试放大（offered 分母不变，实际负载可能反弹）。

**评价标准**：最大化 offered-level 净价值

`NGV = Σ_c w_c·G_c − Σ_c k_c·R_c − c_retry·RetryLoad − c_slo·Violations − c_fair·FairnessGap`

约束：accepted SLO ≥ 目标、各类拒绝率 ≤ 上限、P99/恢复时间、可用性 ≥ 目标；公平性看 long/short 拒绝率比与 Jain 指数；重试放大看 retries/reject、有效 offered 倍数、goodput per original offered。落地：默认 open；只有预测到 TPOT/队列将越线时才 gate；用分层/加权桶（WFQ）而非单桶；用实测 GPU time 校准权重；网关侧 429 + `Retry-After`；shadow mode 量“误拒率”；同时报 Good/Offered、Accepted SLO、拒绝率和恢复时间。

## 第 5 题：Release 证据审计与技术叙事

**A. 不能原样发布（“生产 SLA”不成立）。** 修订：*“在 Qwen3-0.6B、单卡 RTX 4090、P128/O128、temperature=0 的受控实验中，90 req/s 在 40 s、4096 KV blocks、prefix cache off、单 seed 下 3,588/3,588 达标；在 240 s、KV auto、prefix cache on 的另一配置下，90 req/s 有 3.1% TPOT 违约。因此 80 req/s 是已验证安全点，90 req/s 只是边界候选，不是生产 SLA。”*

**B. 可以发布；建议补 25 s/单 seed/非 429。** 原文数字准确（634/2,601=24.4%，19.04→77.52）。修订：*“25 s 单 seed 过载 pilot（offered 105 req/s）中，客户端 78 req/s 令牌桶把完成窗口 goodput 从 19.04 提高到 77.52 req/s；同时 634/2,601=24.4% 的 offered 请求在到达 vLLM 前被客户端拒绝，offered-level 达标率 75.6%；这是客户端 local reject，不是 HTTP 429。”*

**C. 可以发布。** 表述已限定“逻辑 Token 位置次数/调度次数”，29.6% 计算正确。建议补配置：*“在 512-block KV 压力实验（prefix cache off、单 engine、无 spec decode）中，70 次 preemption 对应 45,420 个重复调度逻辑 token 位置次数，占总调度次数 153,265 的 29.6%；这不是 GPU FLOPs 或 GPU 时间占比。”*

**D. 不能以该措辞发布（因果不成立）。** 修订：*“Nsight 22.71 s 窗口测得 GPU busy 71.5%；1,614 个 >1 ms 的 step 边界间隙合计 4.805 s，与每步 3.57 ms 的 `cudaEventSynchronize` 一致。由于只有 1 条 NVTX、未采 CPU 线程状态，28.5% 空闲不能归因于 CPU Scheduler；请求饥饿、KV 带宽和 profiler 扰动仍未排除。”*

**面试叙事（约 90 秒）**：问题（90 能否上线）→ 方法（锁配置、多 seed、cohort 口径）→ 证据（80 安全、90 边界、100 过载）→ 决策（80 上线、90 只做压测、gate 只用于过载）→ 局限（单卡/单模型/短窗/客户端拒绝）。

我最近做了一个基于 vLLM 0.10.2 的真实模型 Serving 性能分析项目，使用单卡 RTX 4090 部署 Qwen3-0.6B，重点研究 SLO 容量、Scheduler 和 KV Cache 的性能瓶颈。

首先我设计了 Open-loop Benchmark，在不同到达率下测量 TTFT、TPOT、吞吐和 SLO Goodput，发现即使完成吞吐仍然增长，TPOT 也可能已经越过 SLO，因此最大吞吐不能直接等同于稳定服务容量。

然后我分别调整 `max_num_seqs` 和 KV Blocks，观察到准入限制与 KV 容量不足具有不同的退化表现。针对 KV 抢占，我进一步修改 Scheduler，记录请求级执行区间，在一次压力实验中发现 70 次抢占产生了 45,420 个重复调度的逻辑 Token 位置。

此外，我使用 Nsight Systems 分析 GPU Kernel 和 Host 时间线，发现较长执行空隙集中在 Step 边界，但也明确保留了具体 Host 瓶颈尚未被完全证明的限制。

最后，我设计了基于 Token Bucket 的过载保护实验。在一次短时过载下，SLO Goodput 从 19.04 提升到 77.52 req/s，但代价是主动拒绝约 24.4% 的请求。同时，我还通过混合负载实验发现，未经校准的成本模型会造成不必要的拒绝。

整个项目的重点是建立从负载、调度、GPU 执行到容量决策的可验证分析链路，而不是只追求最高吞吐数字。
