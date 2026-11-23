# Week 18 Day 5 — Bottleneck Attribution

> 三个 mode **分开报，不合并成一条性能曲线**：**BASELINE**（`admission256_base` / `admission512_base`，无 trace 无 profiler）、**SCHED_TRACE**（`kv512_trace_client`，scheduler 打点）、**NSYS_PROFILE**（`admission_256_delay90_dur60` + `admission256_nsys_client`，nsys 下重起 server）。

## 0. 结论先行

1. **固定 4096 blocks 后，256 → 512 cap 对吞吐零效果**：12,168.5 → 12,194.1 tok/s（**+0.2%**）；两臂 waiting 峰 **0**、preemptions **0**、running 峰 251/220 均未触 cap ⇒ 该 λ=100 下 `max_num_seqs` 不是约束。goodput 75.1 → 85.7 的差来自 **TPOT**（13.50 → 12.96 ms），不是吞吐。
2. **KV 压力是唯一被证实的机制，但它制造的是尾部而不是吞吐损失**：512 blocks（8,192 tok，4.00x）下 70 次抢占、重复调度位置 **45,420 / 153,265 = 29.6%**、恢复等待 P50 **349 ms**、TTFT P50 7434.7 ms。
3. **nsys 22.7 s / 1,619 step 窗口：GPU busy 71.5%，孔隙全在 step 边界**。kernel 间空隙 P99 只 **59 µs**，但 **1,614 个 >1 ms 的边界空隙合计 4.805 s（21.2%）**，数量≈step 数；CPU 侧 `cudaEventSynchronize` **1,619 次 / 5.78 s（API 时间的 65.6%，3.57 ms/step）** ⇒ 空转来自"**按 step 串行**"，不是逐 kernel 发射（全部 launch 类 API 只占窗口 **12.4%**）。
4. **这条 nsys run 这次有效**（3,839/3,839 全 200、0 超时、`metrics_errors=0`、`metrics_post.txt` 44.9 KB），但**它的吞吐不能拿来跟 BASELINE 比**：rate 95 vs 100，11,596.8 / 12,168.5 = **0.953 ≈ 95/100** ⇒ 吞吐是被需求限制的，看不到 profiler 的吞吐代价；可见的代价在尾部——TPOT P99 **19.34 vs 16.81（+15%）**、E2E P99 2,551 vs 2,232（+14%）、waiting 峰 3 vs 0。
5. **计数器推不出重算**：`vllm:prompt_tokens_total` Δ = **53,292**，正好 = 103 × trace 里的 `prompt_tokens`（每请求一次），29.6% 的重算完全不在其中。

## A. 干净 Admission 256 / 512（同一批 usable KV blocks）

| 项 | max_num_seqs=256 | max_num_seqs=512 | 差 | 说明 |
|---|---:|---:|---:|---|
| `num_gpu_blocks`（`cache_config_info`） | **4,096** | **4,096** | 0 | 两臂都 `--num-gpu-blocks-override 4096` |
| KV tokens / max concurrency | 65,536 / 32.00x | 65,536 / 32.00x | 0 | block_size 16 |
| 注入（dispatch proxy, req/s） | 99.567 | 99.567 | 0 | 同 seed 2027、同计划 |
| 完成（窗口, req/s） | 95.07 | 95.27 | +0.2% | 服务能力几乎相同 |
| output（窗口, tok/s） | 12,168.5 | 12,194.1 | **+0.2%** | 吞吐不变 |
| goodput（窗口, req/s） | 75.1 | **85.7** | +14.1% | cohort 2,388 / 2,700 |
| TTFT P50 / P99 (ms) | 63.1 / **170.0** | 64.9 / **230.8** | +2.8% / +35.8% | 512 臂尾部更差 |
| TPOT P50 / P99 (ms) | 13.50 / 16.81 | **12.96 / 15.83** | -4.0% / -5.8% | 512 臂并发更低 ⇒ step 更短 |
| E2E P99 (ms) | 2,231.7 | 2,201.8 | -1.3% | — |
| waiting 峰（采样） | **0** | **0** | — | 从未排队 |
| running 峰（采样） | 251 | 220 | -12.4% | 都由需求决定，非 cap |
| KV 峰（fraction） | 0.803 | 0.720 | — | 都远未满 |
| preemptions Δ | **0** | **0** | 0 | 无 KV 压力 |
| 失败 / 超时 / drop | 0 / 0 / 0 | 0 / 0 / 0 | — | 全部 HTTP 200，2,987/2,987 |
| dispatch lag P99 (ms) | 27.9 | 7.6 | — | 客户端侧，不入服务端判决 |

**怎么读这张表**：这是 **null result**。cap 从 256 抬到 512，吞吐、完成率、KV 峰全在噪声内（单 seed，+0.2% 不能当收益）。唯一系统性变化是 **TPOT**：running 251 的 batch 比 220 更长，TPOT P50 慢 0.54 ms，于是 joint goodput 掉 10.6 个 req/s；而 TTFT P99 反而涨 61 ms，两条尾朝相反方向抖 ⇒ **两臂 goodput 差不能当作"提高 cap 改善 SLO 容量"的证据**。

## B. KV 抢占 trace（`kv_512_trace_client`：512 blocks / 8,192 tok）

| 证据 | 值 |
|---|---|
| `/metrics` `num_preemptions_total` Δ | **70**（0 → 70） |
| trace `preempt` 事件 | **70**（与计数器**逐一对齐**） |
| 被抢占请求数 / 每请求次数 | **42**（1 次 19 个、2 次 18 个、3 次 5 个） |
| 调度事件 / step | 53,264 / 5,858（step dt P50 4.88 ms，P99 12.57 ms） |
| 逻辑位置：调度 / 唯一 / 重复 | 153,265 / 107,845 / **45,420（29.6%）** |
| 重复位置 vs Σ`previously_computed` | 45,420 **完全相等**（无第三方重算来源） |
| 唯一位置 vs 计数器（prompt 53,292 + gen 52,736 + warm-up 1,936） | 107,845 vs 107,964（**-0.1%**） |
| 抢占→恢复等待 | P50 **349 ms**，均值 803 ms，最长 2,633 ms |
| trace 完整性 | 59,192 事件 < 100,000 上限；单一 pid 9800；step 1…5858 单调无重置；119 请求 = 103 被测 + 16 warm-up |

**一个真实例子**（`chatcmpl-6790eb4a…`，非合成）：step 802 预填 `[0,520)` → 逐 token 到 567 → step 850 被抢占（`previously_computed=567`）→ 等 48 step → step 923 从 **`start=0` 重算 568 个位置** → step 946 再抢占（590）→ step 1003 重算 591 → step 1051 再抢占（638）→ step 1129 重算 639。**该请求共被调度 2,826 个位置，而它的 token 只到 857**，约 **3.3×** 冗余。

**已知局限**：只统计"被调度的逻辑位置"，不是 FLOPs/时间；只在 prefix cache 关、单 engine、无 spec decode 下成立；窗口外的生命周期会被低估（本 run 无截断，但 2.63 s 的最长恢复等待说明排队才是主要代价，而不是那 45,420 个位置本身）。

## C. Nsight Systems（`admission_256_delay90_dur60`，重做版）

**采集事实**：`NSYS_DELAY=90 NSYS_DURATION=60` 起 nsys server（`--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --trace-fork-before-exec=true --cuda-graph-trace=node`）。导出元数据是 `START_DELAY_MS=0` / `RUN_DURATION_MS=60000`，但**实际窗口由 kernel 时间戳决定**：session start **13:57:16.93Z**，kernel 活动 **13:57:17.02 → 13:57:39.73Z = 22.71 s**，而 probe 是 **13:56:58.49 → 13:57:39.76Z（41.26 s）** ⇒ **窗口只覆盖后 55%，起点在 probe 内 18.4 s 处，终点与 probe 结束对齐**（大概率是手动 `nsys stop` 早于 60 s）。元数据里的 `START_DELAY_MS` 与 `RUN_DURATION_MS` 都不能当窗口用。前两次尝试（13:05、13:37）都因为**采集窗口到点时 nsys 顺手把 server 杀了**而报废（后一次直接导致 `run_probe.sh` 的 `curl: (7)`）；这次 probe 之后 `metrics_post.txt` 仍是有效 44.9 KB ⇒ server 活过了 probe 结束，报告可用。

| 指标 | 值 |
|---|---|
| 窗口内并发（78 个采样） | running 0 → **250** → 0，均值 **175**；≥150 占 68%、≥200 占 33%；kv 峰 **0.792**；waiting 峰 3；preemptions 0 |
| GPU kernels / 不同 kernel 名 / 每 step | **772,274** / 33 / **477** |
| 推断 step 数 | **1,619**（`reduce_kernel`=1,619；`reshape_and_cache`=45,332=28×1,619；另用窗口内 gen token 273,601÷1,619=169 tok/step ≈ 采样 running 175 复核） |
| step 时间（相邻 `reduce_kernel` 间隔） | P50 **13.10 ms**、P95 21.66、P99 25.33、max 31.58 |
| GPU busy / duty cycle | **16.24 s / 71.5%**（平均 10.03 ms busy / step） |
| kernel 间空隙 | **1,614 个 >1 ms，合计 4.805 s（21.2%）**；P50 **0.06 µs**、P99 **59 µs**、max 10.25 ms |
| launch 类 API | `cudaGraphLaunch` 37,526（33.1 µs/次，1.24 s）· `cudaLaunchKernel` 182,418（5.9 µs，1.07 s）· `cuLaunchKernel` 96,910（5.1 µs，0.49 s）→ 合计 **2.81 s = 12.4% 窗口**，约 196 次/step |
| `cudaEventSynchronize` | **1,619 次 / 5.78 s（占 API 时间 65.6%）**，均值 **3.57 ms**（max 11.44 ms） |
| kernel 时间分类（**不可相加当 step 时间**） | attention(split-KV) **9,866 ms / 60.8%**（45,808 次，215 µs/次，28.3/step）· GEMM 5,024 / 30.9%（239,815 次，148/step）· triton pointwise/reduce 661 / 4.1%（364,275，225/step）· torch elementwise+copy 544 / 3.3% · `reshape_and_cache` 140 / 0.9% |
| NVTX | 仍只有 1 条（"CCCL"），**没有 vLLM range** |
| run 有效性 | **有效**：3,839 请求全 200、0 超时、0 drop、`metrics_errors=0`、`metrics_post.txt` 44.9 KB |

**GPU 是连续忙碌还是周期性空闲？** 尺度分明：step 内 28 个 attention 与 ~148 个 GEMM kernel 之间几乎不留缝（**P99 59 µs**），而每 step 之间有一个 **~3.0 ms** 的边界空隙（1,614 个 >1 ms 空隙 ÷ 1,619 step；4.805 s ÷ 1,619 = 2.97 ms），与 `cudaEventSynchronize` 的 **3.57 ms/step** 吻合 ⇒ **空转是主机侧"等上一步 + 准备下一步"，不是 kernel 发射抖动**。

**profiler 到底贵不贵？** 吞吐看起来 -4.7%（11,596.8 vs 12,168.5），但 rate 是 95 vs 100：**11,596.8 / (95/100) = 12,207 ≈ 12,168.5** —— 吞吐完全由需求决定，看不出 profiler 的吞吐代价。相反尾部有可测退化：TPOT P99 **19.34 vs 16.81 ms（+15%）**、E2E P99 +14%、waiting 峰 3（base 为 0）；`--cuda-graph-trace=node` 会把 graph replay 展开成逐 node（本 run 37,526 次 `cudaGraphLaunch`），这与 TPOT 尾变差方向一致，但 rate/seed 不同，**只能算弱证据**。

**这里无法证明的**：nsys 只采 CUDA activity，没有 DRAM/L2/Tensor-Core 计数器，所以 60.8% 的 attention 份额**不能**拆成"算"还是"读 KV"；`--cuda-graph-trace=node` 的逐 node 展开会让 API 侧时间（含 graph launch 33.1 µs/次）系统性高于未开 profiler 的情况；窗口不含前 18 s（ramp）且尾部含排空，**不是严格稳态**。

## D. 四个竞争假设：控制与判决

| 假设 | 支持证据 | 反证 / 边界 | 判决 |
|---|---|---|---|
| **H1 GPU 计算/激活** | duty **71.5%**；attention+GEMM = **91.7%** kernel 时间；step P50 13.1 ms（avg running 175），与 10.03 ms busy/step 相称 | 28.5% 空闲说明**没打满**；无 tensor-core 计数器，"忙"≠"算满" | **主导，但非唯一** |
| **H2 HBM / KV 带宽** | 每 step 28.3 个 split-KV attention（215 µs/次，随 KV 长度增长）独占 **60.8%**；窗口内 kv 峰 0.792（≈5.2 万 token 常驻），每 step 必须整段读一遍，结构上就是读带宽敏感 | 未采集 DRAM 计数器；**不能**把 60.8% 归为带宽 | **最可能的第二瓶颈，未证** |
| **H3 Host 调度/发射** | 每 step ~196 次 launch、`cudaEventSynchronize` 5.78 s（**25.5% 窗口**，3.57 ms/step）、1,614 个 step 边界空隙 4.805 s（**21.2%**） | 单次 `cudaLaunchKernel` 仅 **5.9 µs**，全部 launch 类 API 合计 **2.81 s = 12.4%** 窗口 ⇒ 逐 kernel 发射不是主因 | **否（逐 kernel）；是（step 级串行）** |
| **H4 Scheduler 准入与重算** | 512 blocks 下 70 次抢占、**29.6%** 冗余位置、恢复等待 349 ms、TTFT P50 7.4 s | 4096 blocks 下 **0 抢占、waiting 峰 0**，admission 臂根本没被准入卡过 | **只在 KV 受限时成立；且是尾部放大器，不是吞吐杠杆** |

## E. Evidence strength

**Observed**：A 表全部行（两臂 `num_gpu_blocks=4096`、12,168.5 / 12,194.1 tok/s、TTFT 170.0 / 230.8 ms、preemptions 0/0）；B 的 70=70、45,420、29.6%、107,845、349 ms 与那个请求的时间线；C 的 772,274 kernel / 1,619 `reduce_kernel` / 1,619 次 sync 共 5.78 s、71.5% duty、1,614 个 >1 ms 空隙合计 4.805 s、分类份额（60.8% / 30.9%）、NVTX 仅 1 条、窗口 13:57:17.0–13:57:39.7Z（22.71 s）对应 probe 41.26 s、窗口内 78 个采样 running 0→250→0、nsys 客户端 3,839/3,839 全 200 且 `metrics_errors=0`；`START_DELAY_MS=0` 与 `RUN_DURATION_MS=60000` 与实际窗口不一致；`prompt_tokens_total` Δ=53,292。

**Inferred**：1,619 个 step（三条互证：`reduce_kernel` 计数、`reshape_and_cache` 45,332=28×1,619、窗口内 gen token 273,601÷1,619=169 ≈ 采样 running 均值 175）；空闲集中于 step 边界而非 kernel 之间（>1 ms 空隙数 1,614 ≈ step 数 1,619，且每 step 有一次 3.57 ms 的 `cudaEventSynchronize`）；profiler 的**吞吐**代价可忽略（11,596.8/(95/100)=12,207 ≈ 12,168.5），代价体现在 TPOT/E2E 尾部（+15%/+14%）；cap 不是约束（waiting=0 且 running 未触 cap）；H2 的"读带宽敏感"由 kernel 形状与 KV 规模外推。

**Unknown**：无 per-kernel 计数器时的 compute vs bandwidth 分解；28.5% 空闲中"等 GPU 完成"与"等 CPU 准备"各占多少（需要 CPU 采样/NVTX 对齐，而 NVTX 是空的）；窗口缺前 18.4 s（ramp）且尾部含排空，**不是严格稳态**；512 blocks 下的稳态吞吐上限（该 run 完成率 3.1 req/s 主要被 7–15 s 的 TTFT 拖住）；单 seed + 30/41 s 有限窗的稳态性；`--cuda-graph-trace=node` 对 step 时间的定量扰动（缺同 workload、同 seed、无 profiler 的对照）。
