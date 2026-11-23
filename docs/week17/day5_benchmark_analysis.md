# Day 5：CUDA Graph 实验分析（no_graph vs piecewise）

## 0. 结论先行

| 指标 | no_graph | piecewise | 变化 |
|---|---|---|---|
| 单请求 96 token 端到端 p50 | 1,066.3 ms | **414.4 ms** | **2.57×** 更快 |
| 8 并发（768 token）端到端 p50 | 1,183.3 ms | **509.4 ms** | **2.32×** 更快 |
| 8 并发输出吞吐 p50 | 649.0 tok/s | **1,507.5 tok/s** | **2.32×** 更高 |
| 首 token 延迟 p50（TTFT） | 27.4 ms | 24.1 ms | 1.14× 更快（**基本没变**） |
| 正确性 | 5,280 token 哈希 | 完全一致 | ✅ |
| CUDA Graph 显存 | 0.00 GiB | 0.09 GiB（94 MiB） | +0.38% 显存 |
| Nsight：图内执行的 kernel 占比 | 0% | **79.1%** | — |
| Nsight：相邻 kernel 间隔 p50 | 18.11 µs | **0.06 µs** | 约为原来的 1/300 |
| Nsight：GPU 忙占比 | 13.7% | 31.5% | +17.8 pp |

**一句话**：这是典型的 **launch-bound** 场景 —— 0.6B 模型在 4090 上每个 decode step 要发 ~420 个 kernel、GPU 只忙 2 ms，其余时间全在等 CPU 逐个 launch。CUDA Graph 把 29×11.6 个 kernel 压成 29 次 `cudaGraphLaunch`，端到端直接减半以上；而 prefill 因为超过 `max_capture_size=16` 全程无图，所以 TTFT 几乎没改善。

---

## 1. 实验条件

### 1.1 硬件 / 软件

| 项 | 值 | 来源 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090，24 GB（25,512,443,904 B），AD102，128 SM，L2 72 MiB，2.52 GHz | nsys `TARGET_INFO_GPU` |
| CPU | Intel Xeon Platinum 8358P，128 核 | nsys `TARGET_INFO_SYSTEM_ENV` |
| 驱动 / CUDA | 570.124.06 / 12.8（`CudaDriverVersion=12080`） | nsys |
| OS / 容器 | Ubuntu 24.04.1 LTS，容器内 `NVIDIA_VISIBLE_DEVICES=6`（映射为 CUDA 0） | nsys |
| vLLM / PyTorch | 0.10.2 / 2.8.0+cu128，V1 引擎（`VLLM_USE_V1=1`） | 引擎日志、`environment/环境配置.md` |
| Nsight Systems | 2024.6.2（CLI） | nsys |
| 模型 | `Qwen/Qwen3-0.6B`，bf16，28 层，权重 1.12 GiB | 引擎日志 |

### 1.2 两个实验条件（唯一变量：`cudagraph_mode`）

| | no_graph | piecewise |
|---|---|---|
| 启动方式 | `bash experiments/day5_serve.sh no_graph` | `bash experiments/day5_serve.sh piecewise` |
| `--compilation-config` | `{"level":3, "cudagraph_mode":0}` | `{"level":3, "cudagraph_mode":1}` |
| 语义 | `CUDAGraphMode.NONE`：**torch.compile 照旧、piecewise 切分照旧**，只是不录图/不回放 | `CUDAGraphMode.PIECEWISE`：每段单独录图，attention 留在图外 |
| 其它 flag | `--max-model-len 512 --max-num-batched-tokens 256 --max-num-seqs 8 --gpu-memory-utilization 0.7 --no-enable-prefix-caching --generation-config vllm --dtype auto` | 同左 |
| `cudagraph_capture_sizes`（自动推导） | `[16, 8, 4, 2, 1]` | 同左（`max_capture_size=16`） |
| 图捕获 | 跳过（`gpu_model_runner.py:3046` WARNING） | `5/5`，1 s，0.09 GiB |

> **重要澄清**：`no_graph` **不是** `--enforce-eager`。它还是一次完整的 level-3 piecewise 编译（`torch.compile takes 5.32 s`、`Directly load the compiled graph(s) ... 2.259 s`），只是编译产物外面没包 `CUDAGraphWrapper`。真正的 eager（`day5_serve.sh eager`）本次**没有跑**，所以本文所有结论只覆盖「有图 / 无图」这一对，不外推到 eager。
>
> 对应 `day5_theory_answers.md` 第 4 题的 A/B/C 三档：本文只有 **B（编译、无图 = no_graph）** 和 **C（piecewise 图 = piecewise）**，**A（eager）缺数据**。

### 1.3 压测客户端（`experiments/day5_bench.py`）

| 项 | 值 |
|---|---|
| 接口 | `POST http://127.0.0.1:8000/v1/completions`（OpenAI 兼容），httpx，`max_connections=16` |
| 采样 | `temperature=0`、`ignore_eos=true`、`max_tokens=96`（强制走满 96 步，排除 EOS 提前结束的干扰） |
| prompt | 同一句「Explain GPU inference scheduling, paged KV cache and CUDA Graph.」×8，batch 内再追加 `\nRequest variant: i.`（≈100 token，>16 且 <256） |
| 预热 | 5×batch1 + 3×batch8，**不计入统计** |
| 计时 | 10×batch1、5×batch8、5×streaming；`p50 = statistics.median` |
| 校验 | `request_id` 唯一、`finish_reason == "length"`、`completion_tokens == 96`、输出 sha256 |

### 1.4 Nsight 采集（`experiments/day5_nsys_offline.py`）

| 项 | 值 |
|---|---|
| 方式 | 进程内 `torch.cuda.profiler.start()/stop()` 只框住**最后一次** `llm.generate`；前 5 次预热在框外 |
| 抓到的范围 | 1 个 prefill + 95 个 decode step（96 个输出 token） |
| 工具 | `nsys profile` → `.nsys-rep`，`nsys export --type sqlite` → `.sqlite`（本文全部数字从 sqlite 的 `CUPTI_ACTIVITY_KIND_*` 表现算） |
| 窗口长度 | no_graph 1,454 ms / piecewise 610 ms（**含 nsys 插桩开销**，绝对值比 §3 的 HTTP 数字高，只用于看结构） |
| 未归档 | 当时执行的 nsys 命令行本体没有存进仓库，只有产物；复现命令见 §8 |

---

## 2. Correctness

两个模式各 55 个请求（10 次单请求 + 5×8 并发 + 5 次流式），共 **5,280 个输出 token**。逐请求对 completion 文本取 sha256：

| 检查 | no_graph | piecewise | 结论 |
|---|---|---|---|
| `finish_reason == "length"` | 55/55 | 55/55 | ✅ 都跑满 96 步 |
| `completion_tokens == 96` | 55/55 | 55/55 | ✅ 无提前截断 |
| `request_id` 批内唯一 | 5/5 批 | 5/5 批 | ✅ 客户端并发真打进去了 |
| 单请求输出（10 次） | 1 种文本 `557c37bf…` | 同 | ✅ 完全确定 |
| 8 并发输出（40 次） | `0a7e905f…`×30、`db1f5e19…`×5、`3d9bd751…`×5 | **同一分布** | ✅ |
| 流式输出（5 次） | `0a7e905f…` | 同 | ✅ |
| **跨模式逐请求哈希** | — | **完全相同（多集合全等）** | ✅ |

三条结论：

1. **图回放与无图执行逐 bit 一致**。`temperature=0` 下 greedy 路径对数值扰动敏感，能完全对齐说明 padding 出来的那些行没有污染 logits —— 这与 `logits_indices` 只取真实 token 位置（`gpu_model_runner.py:2101`）、`attn_metadata` 按真实 token 数构建（`:2026`，早于 `_preprocess` 的 padding）在机制上吻合。
2. **8 并发的 40 个请求只产生 3 种文本**是正常的：8 条 prompt 只差一个 `variant: i` 后缀，greedy 解码在几步后就收敛到同一路径；关键是这个分布**在两个模式下逐条相同**，说明 batch 内的 padding 和请求顺序没有引入差异。
3. 本实验只验证了「同一模型、同一 benchmark 输入」下的等价性，**没有**验证 `cudagraph_capture_sizes` 之外的形状（例如 ≤16 的 chunked prefill 尾块 → size=4 的图）。

---

## 3. 延迟

单位 ms。`p50 = median of 10（单请求）/ 5（并发）`。

| 指标 | no_graph p50 | mean | min | max | std | piecewise p50 | mean | min | max | std | 加速 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 单请求 96 token | 1,066.34 | 1,067.32 | 1,051.12 | 1,093.34 | 12.63 | **414.39** | 417.90 | 411.47 | 434.58 | 7.40 | **2.57×** |
| 8 并发 768 token | 1,183.28 | 1,186.16 | 1,176.33 | 1,197.25 | 9.14 | **509.44** | 510.88 | 506.42 | 518.79 | 4.27 | **2.32×** |
| 流式端到端 | 1,055.82 | 1,055.93 | 1,050.42 | 1,060.90 | 3.38 | **406.03** | 409.31 | 405.62 | 418.57 | 4.96 | **2.60×** |
| 首 token（TTFT） | 27.42 | 28.28 | 23.30 | 35.35 | 3.99 | 24.07 | 26.01 | 21.22 | 33.30 | 4.29 | 1.14× |
| 8 并发单请求延迟 | 1,162.4 | — | 1,130.3 | 1,194.8 | — | **501.4** | — | 488.7 | 517.1 | — | 2.32× |

拆到「每 token」和「每个 decode step」（`(e2e - TTFT) / 95`）：

| | no_graph | piecewise | 加速 |
|---|---|---|---|
| 每输出 token（e2e/96） | 11.11 ms | 4.32 ms | 2.57× |
| 每 decode step | 10.94 ms | 4.11 ms | **2.66×** |
| 单请求吞吐 | 90.0 tok/s | 231.7 tok/s | 2.57× |

三个观察：

1. **省下来的几乎全在 decode 的 step 间隔上**，不在 kernel 计算里（§6 证明 kernel 总时长只差 3.5%）。因此「每 decode step」的加速（2.66×）比端到端（2.57×）略高 —— 端到端里还含一段没被优化的 prefill。
2. **TTFT 只快 14%**，与「prefill 无图」完全一致。剩下的差异来自 nsys/CPU 抖动，量级在标准差（4 ms）以内。
3. **并发代价变大**：8 并发相对单请求的延迟放大，no_graph 是 1.11×（1,183/1,066），piecewise 是 1.23×（509/414）。因为图模式把「固定开销」压掉了，batch 内多出来的 7 条请求的边际成本占比就变高了 —— 这反而是好事，说明瓶颈从固定开销换成了真实计算。

---

## 4. 吞吐

| 指标 | no_graph | piecewise | 变化 |
|---|---|---|---|
| 单请求 | 90.0 tok/s | 231.7 tok/s | +157% |
| 8 并发 p50 | 649.0 tok/s | **1,507.5 tok/s** | **+132%** |
| 8 并发 min / max | 641.5 / 652.9 | 1,480.4 / 1,516.5 | — |
| batching 增益（并发/单请求） | 7.21× | 6.51× | — |

- 8 并发达到单请求的 6.5×（理想 8×），说明 batch=8 时合批仍然很划算；no_graph 的 7.21× 更高，是因为它的单请求基线被 launch 开销压得太低（分母被夸大）。
- piecewise 的吞吐抖动更大（std 12.5 vs 5.0 tok/s），但绝对值仍比 no_graph 的最好成绩（652.9）高 2.27×，不存在「偶尔跑快」的解释空间。

---

## 5. 显存开销

引擎启动日志（`gpu_worker.py:391`，两个模式逐字对照）：

| 项 | no_graph | piecewise | 差值 |
|---|---|---|---|
| 权重 | 1.12 GiB | 1.12 GiB | 0 |
| 峰值激活 | 0.07 GiB | 0.07 GiB | 0 |
| 非 torch 内存 | 0.02 GiB | 0.02 GiB | 0 |
| **CUDAGraph 内存** | **0.0 GiB** | **0.09 GiB** | **+0.09 GiB** |
| 设备空闲/总量 | 23.32 / 23.76 GiB | 同 | 0 |
| KV cache 实占 | 16,568,667,340 B（15.43 GiB） | **完全相同** | 0 |
| KV 块数 / token 容量 | 9,029 块 / 144,464 token | **完全相同** | 0 |
| 建议 `--kv-cache-memory` | 16,411,380,940 | 16,312,814,796 | **−98,566,144 B（94.0 MiB）** |

解读：

1. **图本身只要 94 MiB**：`16,411,380,940 − 16,312,814,796 = 98,566,144 B ≈ 94.0 MiB`，占 24 GiB 显存的 **0.38%**，占 15.43 GiB KV 的 **0.59%**。日志把它取整成 `0.09 GiB`，比精确值（0.0918 GiB）略小。
2. **145 张图 → 0.65 MiB/张**：5 个尺寸 × 29 段 = 145 张，`94.0 MiB / 145 ≈ 663.8 KiB/张`。这个数字随层数线性增长（每层多一段图），换 7B/14B 模型时会大一个量级，不能拿 94 MiB 外推。
3. **KV 容量一点没少**：两个模式的 9,029 块 / 144,464 token / `282.16x`（512 token 请求的最大并发）完全一致。原因在 `gpu_worker.py:355-367`：KV cache 大小是用「可用内存 − 非 KV 占用（权重/激活/非 torch） − **150 MiB 冗余缓冲**」算出来的，而 94 MiB 的图正好落在缓冲里。换句话说，**图池是从那 150 MiB 安全垫里扣的，不是从 KV 里扣的**（代价是总占用略高于 0.7 的目标线；本次配置下恰好成立，图再大就会挤 KV）。
4. **启动代价**：`init engine` 9.95 s → 10.42 s（+0.47 s），其中图捕获本身 `Graph capturing finished in 1 secs`。torch.compile 时间两者基本相同（5.32 s / 5.25 s），因为编译缓存命中（`Directly load the compiled graph(s) ... 2.2 s`）。

---

## 6. Nsight 观察

方法：`nsys export --type sqlite`，直接查 `CUPTI_ACTIVITY_KIND_KERNEL` / `_RUNTIME` / `_MEMCPY`（时间单位 ns）。窗口 = 1 次 prefill + 95 次 decode（96 个输出 token），**无预热**。

### 6.1 总量：GPU 干的活一模一样，差别全在「怎么发」

| 指标 | no_graph | piecewise | 变化 |
|---|---|---|---|
| kernel 总数 | **40,468** | **40,468** | **完全相同** |
| 其中由图回放执行 | 0（0%） | **32,015（79.1%）** | — |
| kernel 总时长（GPU 忙） | 199.24 ms | 192.20 ms | −3.5% |
| 窗口跨度（wall） | 1,454.23 ms | **610.49 ms** | **−58.0%** |
| GPU 忙占比 | 13.7% | **31.5%** | +17.8 pp |
| 相邻 kernel 间隔 p50 | **18.11 µs** | **0.06 µs** | 1/300 |
| 相邻 kernel 间隔 p90 / 均值 | 77.09 / 31.01 µs | 50.50 / 10.34 µs | — |
| 间隔 >100 µs 的次数 | 925 | 328 | — |
| 间隔 >1 ms 的次数 | 7 | 4 | — |
| CPU 侧 CUDA API 调用数 | 44,242 | **17,737** | **−59.9%** |
| CPU 侧 API 总耗时（含插桩开销） | 270.02 ms | 160.88 ms | −40.4% |

**kernel 数完全相同（40,468）是这组实验最硬的一条对照**：CUDA Graph 没有减少任何一个 kernel、没有融合任何算子、也没有改变计算图，它只改变了「kernel 是谁发出来的」。所以 2.32× 的加速不可能来自算法，只能来自 launch 路径。

### 6.2 发射路径：从 40,468 次逐 kernel launch 变成 2,755 次 graph launch + 8,453 次裸 launch

| API | no_graph | piecewise | 说明 |
|---|---|---|---|
| `cudaLaunchKernel` | 18,840 | 8,200 | runtime API：cublasLt GEMV + flash + elementwise（torch 侧算子） |
| `cuLaunchKernel` | 21,628 | 253 | driver API：几乎全是 Triton kernel（triton 的 C launcher 走这条） |
| `cudaGraphLaunch` | **0** | **2,755** | 每 1 次 = 一张图 = 平均 **11.6 个 kernel** |
| `cudaMemcpyAsync` | 3,332 | 3,332 | H2D/D2H，两模式相同 |
| `cudaMemsetAsync` | 124 | 124 | 相同 |
| `cudaStreamIsCapturing` | 96 | 2,851 | 回放/录制前查 stream 是否在 capture，≈30 次/step（与 29 段图同量级） |
| `cudaEventSynchronize` | 96 | 96 | 每 step 一次同步点，可当 step 计数 |

三条自洽性检查（都能对上，说明数字没算错）：

- `18,840 + 21,628 = 40,468` = kernel 总数 → no_graph 下**每个 kernel 恰好一次 launch**；
- `8,200 + 253 = 8,453`，而 `40,468 − 32,015 = 8,453` → piecewise 下**没进图的 kernel 数正好等于直接 launch 数**；
- `32,015 / 2,755 = 11.62` → 每张图平均 11.6 个 kernel。

### 6.3 谁在图里、谁在图外

| kernel 家族 | 个数 | GPU 时间（no_graph / piecewise） | 在哪 |
|---|---|---|---|
| `cublasLt gemvx`（线性层 GEMV） | 10,736 | 142.1 / 140.0 ms（**71% / 73%**） | **图内**（图外只剩 96 个） |
| Triton fused（RMSNorm / SiLU / 残差） | 21,600 | 25.1 / 21.9 ms | **图内**（图外 225 个） |
| `flash_fwd_splitkv*`（attention） | 4,508 | 24.1 / 22.9 ms | **图外**（splitting op） |
| `reshape_and_cache_flash`（写 KV） | 2,688 | 4.7 / 4.3 ms | **图外** |
| 其它（elementwise/memset） | 936 | 3.2 / 3.0 ms | 大部分图外（672 个） |

- **attention 一定在图外**：`cudagraph_mode=PIECEWISE` 的 `splitting_ops` 就是所有 attention op（`compilation.py:336-345`），切分点不录图，因为它的 kernel 参数随 `seq_lens`/`block_table`/`num_splits` 变。7,196 个 attention 相关 kernel（4,508 + 2,688）**全部**是直接 launch。
- **线性层一定在图内**：10,736 个 gemvx 里 10,640 个由图回放，这正是省下 launch 的大头（每层 4 个 linear × 28 层 = 112 个/step）。
- `reshape_and_cache` 恰好 **2,688 = 28 层 × 96 step**，是核对「28 层 / 96 步」这两个数字最方便的一把尺子。

### 6.4 step 结构：95 个 decode step，每步 29 次 graph launch

对 `cudaGraphLaunch` 的时间戳按 >1 ms 间隔切分：

```
组数 = 95，每组 launch 数 = {29}
```

- **95** = 96 个输出 token 里扣掉 prefill 那一步（第 1 个 token 由 prefill 的采样产生，之后 95 次 decode）；
- **29** = 28 个 attention 切分点 → 29 段图（Qwen3-0.6B 是 28 层）；
- prefill（≈100 token > `max_capture_size=16`）**一次图都没用**，所以这 95 组里没有任何一组属于 prefill。

每个 decode step 的发射量对比：

| | no_graph | piecewise |
|---|---|---|
| kernel 总数 / step | ≈421 | ≈421 |
| 由图回放 | 0 | ≈337（29 张图） |
| 直接 launch | ≈421 | ≈89（attention 相关 + elementwise） |
| 发射调用 / step | ≈421 | **≈118**（29 graph + 89 kernel） |

### 6.5 时间去哪了（这是整份分析的核心）

按「step 周期」（相邻 `cudaEventSynchronize` 间隔）和「GPU 忙/闲」拆：

| | no_graph | piecewise |
|---|---|---|
| step 周期 p50 | **14.60 ms** | **6.09 ms** |
| step 周期 p90 / max | 15.07 / 23.92 ms | 6.18 / 15.19 ms |
| GPU 忙 / step | 2.08 ms（14.2%） | 2.00 ms（32.8%） |
| GPU 空闲 / step | **12.52 ms** | **4.09 ms** |

- no_graph：每个 step 里 GPU 有 12.5 ms 在等。间隔的中位数是 18 µs —— 这正好是「一次 `cudaLaunchKernel` 的 CPU 成本 + 一次 kernel 启动延迟」的量级（40,468 次 launch 的 CPU 时间 222.8 ms ≈ 5.5 µs/次，但真正卡住 GPU 的是驱动的 submit/同步路径，nsys 插桩会放大它）。
- piecewise：图内的 kernel 间隔中位数 **0.06 µs**（同一个 `cudaGraphLaunch` 里 kernel 背靠背，硬件级连续），剩下的 4.09 ms 空闲集中在**每步 29 次 graph launch 之间**和 **step 与 step 之间**（scheduler、采样同步、Python 开销）。也就是说优化后的瓶颈已经从「kernel 级」上移到了「step 级」。
- 因此下一步能继续榨的只有两处：把 attention 也塞进图（FULL 图，需要 backend 支持 + 固定 metadata），或者减 step 边界上的 CPU 工作（异步调度 / 更少的同步点）。这与把 `cudagraph_mode` 再往上调（`FULL_AND_PIECEWISE`）的方向一致，但本次**没有做过**该实验。

### 6.6 测量口径提示

1. nsys 会让每次 API 调用变慢（`cudaEventSynchronize` 96 次共 13.2/13.8 ms），所以 §6 的绝对时间（610/1454 ms）比 §3 的 HTTP 数字（414/1066 ms）高 35–50%。**只用比值，不用绝对值**。
2. `.sqlite` 里 kernel 时间是 CUPTI 报告的 device duration；「GPU 忙」是本文把区间并集算出来的（本实验全程单 stream，并集≈求和，`streamId` 去重后为 1）。
3. `CUDA_GRAPH_NODE_EVENTS` 表只有 5,452 条，远小于图内 32,015 个 kernel，说明该 trace 没有为每次回放展开所有 node（`--cuda-graph-trace=node` 才会），本文的图内/图外归属是用 `graphNodeId is not null` 判的，不依赖这张表。

---

## 7. 能得出的结论 / 不能得出的结论

**能得出：**

1. 在这个模型/这张卡上，decode 是**彻底 launch-bound** 的：GPU 只忙 13.7%，且 kernel 数（40,468）在开图前后完全相同。CUDA Graph 的收益 = 消灭 launch 间隔，实测端到端 2.57×、decode step 2.66×。
2. 收益的边界很清晰：**只有命中 `cudagraph_capture_sizes`（本次 1/2/4/8/16）的 batch 才有图**，prefill（≈100 token）完全吃不到，所以 TTFT 只快 14%。
3. 图模式的成本很低：94 MiB（0.38% 显存）、1 s 捕获、0.47 s 启动，且因为落在 150 MiB 冗余缓冲里，KV 容量一块没少。
4. 正确性无损：5,280 个 token 跨模式逐请求哈希全等。

**不能得出（本次未覆盖）：**

1. **padding 的实际浪费没被测到**：所有请求的 batch 都正好命中 1、8，`pad_for_cudagraph` 一次都没真的「向上取整」。要量化浪费得跑 3/5/7 并发（3→4、5→8、7→8）。
2. **eager 基线缺失**：`day5_serve.sh eager` 没跑，所以「no_graph vs eager」的差距未知，本文的 2.57× 不能当成「CUDA Graph 相对 eager 的收益」。
3. **大模型不适用**：0.6B + 4090 是极端 launch-bound；GPU 越忙、模型越大、batch 越大，这个倍数越接近 1（那时 kernel 计算才是瓶颈）。
4. **FULL 图未验证**：`cudagraph_mode=1` 只覆盖 mixed（prefill-decode）那套 key，`FULL_AND_PIECEWISE`/`FULL_DECODE_ONLY` 的收益与兼容性都没测。
5. **batch > 8 未测**：`max_num_seqs=8`、capture size 最大 16。9–16 并发会命中 size=16 的图（开始出现真正的 padding 浪费），17 以上会掉回 eager —— 两段都没验证。

---

## 8. 复现命令

```bash
# 1) 起服务（两个模式各一次，注意清干净上一个进程）
bash experiments/day5_serve.sh no_graph   # → results/day5_no_graph_server.log
bash experiments/day5_serve.sh piecewise  # → results/day5_piecewise_server.log

# 2) HTTP 压测（服务就绪后）
python experiments/day5_bench.py --mode no_graph   > results/day5_no_graph.json
python experiments/day5_bench.py --mode piecewise  > results/day5_piecewise.json

# 3) Nsight（离线跑，一次 generate，不含预热）
#    注：当时实际执行的命令行没有归档，下面是等效复现写法
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
     -o results/day5_piecewise_profile \
     python experiments/day5_nsys_offline.py --mode piecewise
nsys export --type sqlite results/day5_piecewise_profile.nsys-rep   # → .sqlite

# 4) 从 sqlite 复算本文数字（最小集）
#    表：CUPTI_ACTIVITY_KIND_KERNEL / _RUNTIME / _MEMCPY、StringIds
#    - kernel 数与图内 kernel：select count(*), count(graphNodeId) from CUPTI_ACTIVITY_KIND_KERNEL
#    - 发射 API 分布：CUPTI_ACTIVITY_KIND_RUNTIME join StringIds on nameId, group by name
#    - 相邻 kernel 间隔 p50：按 start 排序后求 start[i+1]-end[i]
#    - graph launch 分组：按 >1ms 间隔切分 cudaGraphLaunch 的 start
```

---

## 附录：原始数字

### A.1 压测 summary（JSON 原样）

```
no_graph   single_e2e_p50_ms=1066.337  batch8_e2e_p50_ms=1183.275
           batch8_output_tok_s_p50=649.046  first_content_p50_ms=27.422
piecewise  single_e2e_p50_ms= 414.394  batch8_e2e_p50_ms= 509.440
           batch8_output_tok_s_p50=1507.536 first_content_p50_ms=24.074
```

### A.2 单请求 10 次（ms）

```
no_graph   1067.65 1071.02 1051.75 1056.82 1051.12 1063.08 1069.96 1083.44 1065.02 1093.34
piecewise   415.38  434.58  426.58  411.56  424.12  411.47  413.41  414.94  413.14  413.85
```

### A.3 8 并发 5 次（wall ms / tok·s⁻¹）

```
no_graph   1197.25/641.47  1196.69/641.77  1183.28/649.05  1176.33/652.88  1177.26/652.36
piecewise   509.44/1507.54  506.42/1516.53  511.38/1501.81  508.36/1510.73  518.79/1480.38
```

### A.4 引擎内存日志（两行原文的关键片段）

```
no_graph   ... 1.12 GiB for weight, 0.07 GiB for peak activation, 0.02 GiB for
           non-torch memory, and 0.0 GiB for CUDAGraph memory. Replace
           gpu_memory_utilization config with `--kv-cache-memory=16411380940` to fit ...
piecewise  ... and 0.09 GiB for CUDAGraph memory. ... `--kv-cache-memory=16312814796` ...
```

### A.5 Nsight 采集窗口

```
no_graph    kernels=40,468  span=1,454.23 ms  busy=199.24 ms  gpuBusy=13.7%
piecewise   kernels=40,468  span=  610.49 ms  busy=192.20 ms  gpuBusy=31.5%
            in-graph kernels=32,015  cudaGraphLaunch=2,755  (95 组 × 29)
```
