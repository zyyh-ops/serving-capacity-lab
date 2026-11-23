# Week 18 Day 4 — Admission vs KV Capacity Analysis

## 0. 结论先行

1. **Admission A/B 不是吞吐实验**：`max_num_seqs` 256 → 512，waiting 峰 **24 → 0**、TTFT P99 **257.2 → 188.7 ms（-26.6%）**，但 completed **93.67 → 94.33 req/s（+0.7%）**、output **11,989 → 12,075 tok/s（+0.7%）**。goodput 34.8 → 39.17（+12.6%）**全部来自 SLO 命中率，不是吞吐**。
2. 两臂 goodput 都由 **TPOT 支配**：TPOT P50 16.08/15.40 ms 全部越过 15 ms SLO，TPOT 命中率 41.0%/44.8%，而 TTFT 命中 97.7%/99.2%。这一对臂处在 **decode 计算受限**（running≈180、preemptions 0、KV 41–47%），`max_num_seqs` 从未成为吞吐瓶颈。
3. **KV A/B 是容量崩溃实验**：512 → 1024 blocks（8,192 → 16,384 tokens），吞吐只 **×1.55**（2,035 → 3,162 tok/s）、TTFT P50 **39.5 → 16.6 s（-58%）**、goodput 0.325 → 0.60。两臂 KV 都打满 1.0、preemptions 356/360、waiting 单调涨到 311/216 → **KV 容量受限**，且不是线性收益。
4. 半容量反而 **TPOT 更好**（P50 4.83 vs 6.02 ms）：running 只有 9.8，step 更短。`throughput ≈ running/TPOT` 两侧都对得上（2,029 vs 实测 2,035；3,277 vs 实测 3,162），说明这些数字自洽。
5. **抢占只有指标级证据**：计数器 0 → 356/360，无 per-step/per-request 日志；而 token 计数显示总 prompt = 1.00–1.01× 名义、总 gen = 1.00× 名义，**看不到 356 次抢占应有的 re-prefill 放大**（每次重算 520 token 应 +74%）。这条反差必须由 Day5 的 per-step trace 解开。
6. **混杂（必须声明）**：admission 两臂同时改了 KV blocks（**8,254 → 7,448，-9.8%**，来自 `profile_run` 的 activation 1.39 → 2.77 GiB），所以 `kv%` 分母不同，跨臂百分比不可直接比。

## A. Measurement contract

| 项 | admission_256 | admission_512 | kv_1024 | kv_512 |
|---|---|---|---|---|
| Workload | P128/O128, λ=100, T=30 s | 同左 | P512/O512, λ=12, T=40 s | 同左 |
| Seed / vLLM | 2027 / **0.10.2** | 同 | 同 | 同 |
| 硬件 | RTX 4090 24 GB，driver 570.124.06，bf16，V1 | 同 | 同 | 同 |
| Server 关键参数（差异项） | `max_num_seqs=256`，无 block override | `max_num_seqs=512`，无 override | `max_num_seqs=256` + `--num-gpu-blocks-override 1024` | 同上 + `512` |
| 注入 / 完成（窗口） | 99.57 / 93.67 | 99.57 / 94.33 | 12.0 / 6.175 | 12.0 / 3.975 |
| arrival 窗口 → 排空 | 30.0 s → 30.94 s（0.94 s） | 30.0 → 30.92（0.92 s） | 40.0 → **73.7 s**（33.7 s） | 40.0 → **119.4 s**（79.4 s） |
| 窗口末在途 | 177 | 157 | 233 | 321 |
| client dispatch lag P99 | 6.14 ms | 8.59 ms | 2.00 ms | 2.35 ms |
| guard drops / 失败 / 超时 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| metrics 采样 | 110 样本，中位 0.279 s，max 0.428 s，errors 0 | 107，0.284/0.518，0 | 274，0.270/0.380，0 | 445，0.269/0.379，0 |
| 实际 KV blocks / tokens | **8,254 / 132,064** | **7,448 / 119,168** | **1,024 / 16,384** | **512 / 8,192** |

- block_size = 16；prefix cache **关闭**；chunked prefill 开；`max_num_batched_tokens=2048`；`max_model_len=2048`；`gpu_memory_utilization=0.7`。
- 全部 4 个 run 的 output 长度精确（mismatch 0）、`finish_reason=length`、HTTP 200；scheduled = attempted = success。
- 排空口径：尾延迟与 goodput 的 cohort 统计含 drain，`*_in_window` 只算 `end_t < T`；KV arm 的 480 条请求**最终全部完成**（success_after_drain = 480）。

## B. Admission A/B

| Metric | max_num_seqs 256 | max_num_seqs 512 | Difference | Interpretation |
|---|---:|---:|---:|---|
| Injected req/s | 99.567 | 99.567 | 0 | 同一份 Poisson 计划、同 seed |
| Completed req/s | 93.667 | 94.333 | **+0.7%** | 完成率几乎不变 |
| Output tok/s | 11,989 | 12,075 | **+0.7%** | **decode 已达天花板** |
| Goodput req/s（窗口） | 34.8 | 39.167 | **+12.6%** | 增量来自 SLO 命中（cohort 1,221 → 1,332） |
| TTFT P50 / P99 (ms) | 67.4 / 257.2 | 68.1 / 188.7 | P50 0，**P99 -26.6%** | 中位不变，只修尾部 |
| TPOT P50 / P99 (ms) | 16.08 / 19.49 | 15.40 / 18.71 | -4.2% / -4.0% | 两臂都 >15 ms SLO |
| Running 峰 / 均值 | 256.0 / 183.3 | 261.0 / 176.6 | 256 臂**顶到 cap** | 512 臂 cap 未触及 |
| Waiting 峰 / >0 占比 | 24.0 / 4.5% | **0 / 0%** | 队列消失 | 唯一的真实机制差异 |
| KV 峰 | 0.409（≈54.0k tok） | 0.466（≈55.6k tok） | 绝对量 +2.9% | 分母不同，百分比不可比 |
| Preemptions Δ | 0 | 0 | 0 | 无 KV 压力 |
| SLO 命中：TTFT / TPOT / joint | 97.7% / 41.0% / 40.9% | 99.2% / 44.8% / 44.6% | — | **TPOT 是唯一约束** |

**"减少 waiting 是否等于提高可持续吞吐？"** —— **否**。waiting 从 24 归零，token 吞吐只 +0.7%。这个 cap 换来的是**尾部延迟**（TTFT P99 -68 ms、TPOT P50 -0.68 ms）与因此的 goodput +12.6%。同时两臂在 running≈180 时 TPOT 已≈15–16 ms，说明 4090 上 0.6B 的 128-token continuation 在此并发下已饱和；继续抬高 cap 只会把请求堆进同一个 step，而不是产出更多 token。

**graph 参数变了吗？** —— **没有**。4 个 run 的 `compilation_config` 完全一致：67 个 capture size（512,504,…,8,4,2,1）、`cudagraph_mode=1`（PIECEWISE）、`max_capture_size=512`、图内存 0.49 GiB、启动时均 `Directly load the compiled graph(s) ... from the cache`（2.14–2.53 s）。原因在 `cuda_graph_sizes` 的实现：默认取 `min(max_num_seqs*2, 512)`，256 与 512 都被 clamp 到 512。**真正被改变的是 runner 内存**（见 A 表末行的混杂）。

## C. KV capacity A/B

| Metric | 1024 blocks（16,384 tok） | 512 blocks（8,192 tok） | Difference | Interpretation |
|---|---:|---:|---:|---|
| Injected req/s | 12.0 | 12.0 | 0 | 同计划 |
| Completed req/s（窗口） | 6.175 | 3.975 | **-35.6%** | 触发容量坍塌 |
| Output tok/s | 3,162 | 2,035 | **-35.6%** | 容量 ×2 只换吞吐 ×1.55 |
| Goodput req/s | 0.60 | 0.325 | **-45.8%** | 几乎无请求能满足 200 ms TTFT |
| TTFT P50 / P99 (ms) | 16,565 / 31,343 | 39,502 / 75,905 | **×2.38 / ×2.42** | 纯排队 |
| TPOT P50 / P99 (ms) | 6.02 / 11.68 | **4.83 / 9.50** | -19.7% / -18.7% | 小 batch ⇒ 更快 step |
| Running 均值 / 峰 | 19.7 / 28 | 9.8 / 14 | -50% | 容量直接决定并发 |
| Waiting 均值 / 峰 | 103.9 / 216 | 156.4 / 311 | +51% / +44% | 95%/98% 采样 >0 |
| KV 峰（fraction） | **1.0**（22.6% 采样 ≥0.99） | **1.0**（12.4% ≥0.99） | 均打满 | 两臂都饱和 |
| Preemptions Δ | **360** | **356** | ≈ 相同 | 与容量**不成比例** |
| Output length mismatch | 0 | 0 | — | 480/480 均为 512 tok，`length` |
| Error/timeout/client drops | 0/0/0 | 0/0/0 | — | 全部 200，drain 后 480/480 |

**自洽性校验**（这三条把数据锁在一起）：
- `throughput ≈ running / TPOT`：19.7/6.02 ms = 3,277 ≈ 3,162；9.8/4.83 ms = 2,029 ≈ 2,035。
- Little 定律 `E2E ≈ (running+waiting)/完成率`：(19.7+103.9)/6.175 = **20.0 s** vs E2E P50 **20.2 s**；(9.8+156.4)/3.975 = **41.8 s** vs **43.2 s**。
- 两臂总工作量恒定：480 × 512 = 245,760 output token，最终全部完成（drain 33.7 / 79.4 s）。

**抢占发生在哪？** KV 全程 92.6–99.9%，waiting 从 45/90 单调涨到 216/311，running 反而停滞在 10–28 → **准入被 KV 分配失败挡住**，不是被计算挡住。抢占速率与到达同阶（前 5 s 14–17 次，尾部 2–3 次/5 s），**两臂都在 ≈0.74 次/请求**，说明抢占频率由"KV 满 + 新到达"驱动，与容量无关。

**是重算还是只看得到指标？** 只有**指标级**证据：没有任何 per-preemption 日志行（INFO 只有 10 s 滚动汇总），也没有 per-request `num_computed_tokens`。更关键的是一个**反向证据**：从服务端日志重建的总 prompt token = 250,603（kv_512）/ 406,732（admission_256），与名义值（249,600 / 406,232）只差 **0.1–0.4%**；`vllm:generation_tokens_total` 计数器同样 = 1.00× 名义。若 356 次抢占每次重算 520-token 前缀，prompt 总量应 **+74%**（+185k token）—— 完全没有出现。因此：**"抢占 Δ>0" 不能直接换算成"浪费了多少 token"**，两者差异需 Day5 用 per-step trace 判定（是受害者被抢占时几乎无已算前缀，还是 token 计数器不含重算）。

## D. Diagnosis / alternatives

- **H1（admission cap 是瓶颈）：对吞吐不成立，对尾部成立。** 支持：waiting 24→0、TTFT P99 -26.6%、running 在 256 臂精确顶到 256.0。反证：token 吞吐 +0.7%、preemptions 0、KV 仅 41–47%、两臂 TPOT 都已在 15 ms 线上。⇒ cap 只影响**谁在排队**，不影响**单位时间算多少 token**。
- **H2（KV 稀缺是瓶颈）：强支持。** 支持：两臂 KV=1.0、waiting 单调涨、running 被压在 10–28、E2E≈(running+waiting)/速率、容量 ×2 → 吞吐 ×1.55 且 TTFT 减半。反证/边界：吞吐并未随容量线性（0.5×容量只掉 36%），因为小 batch 的 TPOT 更好；且 drain 后 480/480 全成功，说明这是**时延/容量**问题而非**正确性**问题。
- **H3（GPU 计算/主机开销）：分臂而治。** KV 臂**不是**计算受限（decode batch 10–20、TPOT 4.8–6.0 ms、GPU 远未喂满）；admission 臂**是**计算受限（batch≈180、TPOT 15–16 ms、吞吐与 cap 无关）。主机侧无证据：dispatch lag P99 ≤8.6 ms、metrics errors 0、0 drop、0 超时。但没有 GPU util/MFU，无法给出"利用率百分比"。
- **必须记住的混杂**：`max_num_seqs` 512 使 `profile_run` 的虚拟 batch 变成 512 条（`num_reqs = min(2048, max_num_seqs)`），activation 峰值 1.39 → 2.77 GiB，KV blocks 8,254 → 7,448。**这不是纯单变量对照**。
- **Day5 profiling 要定的事**：(a) per-step `num_scheduled_tokens` / `num_computed_tokens` / 被抢占 request id，回答"抢占到底浪费多少 token"；(b) step 时间分解（prefill vs decode vs sampler vs 主机），确认 admission 臂的 15 ms 是 GPU 饱和还是调度/采样开销；(c) 干净 KV A/B —— 两臂都用 `--num-gpu-blocks-override` 且 `max_num_seqs` 固定，避免 profiling 反噬；(d) 多 seed + 更长窗口，避免有限窗 Poisson 与"未到稳态"的干扰。

## E. Result quality

- [x] Cross-variant identical workload and seed：同 seed 2027、同 P/O、同 λ；**但 admission 两臂的 KV blocks 不同（-9.8%）**，非纯单变量。
- [x] 完整 server 日志 + 实际 block 数：4 个 arm 均有 `vllm serve` 命令行、`GPU KV cache size`、`num_gpu_blocks`、图捕获与内存 profiler 输出。
- [x] 无 client guard drop（0/0/0/0），四条臂均未作废。
- [x] Errors/timeouts 披露：失败 0、超时 0、HTTP 全 200、`metrics_errors=0`。
- [x] Counter reset / missing metrics 披露：preemptions 计数 0 → 356/360 单调无回绕；`generation_tokens` 单调；无缺失样本导致的空洞。
- [x] 输出长度与 finish reason 已核对：mismatch 0，480/480 与 2,987/2,987 均 `finish_reason=length`。
- [x] Observed / inferred / unknown 已分离（见下）。

**Observed**：§A–§C 表中所有数字；waiting 24→0 与 TTFT P99 -26.6%；吞吐 +0.7%；KV 臂 ×1.55；preemptions 0/0/356/360；token 总量 1.00×；4 臂 graph 配置一致。

**Inferred**：admission 臂已到 decode 计算天花板（因吞吐与 cap 无关、TPOT 钉在 15 ms、KV 未满、preemptions 0）；KV 臂的吞吐由容量→并发→TPOT 决定（因三条恒等式同时成立）；goodput 差异由 TPOT 而非 TTFT 决定。

**Unknown**：抢占的真实重算代价（per-step trace 缺）；GPU 利用率/MFU 与 step 时间构成；prefix cache 若开启会如何改变 KV 臂；单一 seed 与 30/40 s 有限窗的稳态性；`max_num_seqs` 与 KV 两轴同时变化时的二维曲面。
