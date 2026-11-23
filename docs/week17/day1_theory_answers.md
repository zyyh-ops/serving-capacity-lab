## 第 1 题：为什么要分 API Server、Engine Core 和 GPU Worker

- **API Server**：负责 HTTP/SSE、鉴权限流、请求校验与 tokenize、路由，属于 I/O 密集层，可以多副本部署。
- **Engine Core**：负责调度、KV 管理、请求状态机，是纯 CPU 的串行决策中心。
- **GPU Worker**：负责模型前向、kernel、NCCL、显存，绑定 GPU，并按 TP/PP 拆成多进程。
- **职责全放在一个进程或模块中会出现的问题**：GIL 和事件循环被长请求阻塞；一处崩溃会导致整个服务停止，无法隔离或重启；不能独立扩缩容；无法做跨机 TP；CPU 抖动影响调度精度；升级、多模型、多租户都难做；显存与协议层耦合后难以测试。

## 第 2 题：Offline 与 Online 的区别

- **Offline**：输入静态数据集，在进程内作为库调用（`LLM.generate`），优先考虑吞吐，可以开很大的 batch，不关心 TTFT，跑完即止。
- **Online**：请求动态到达，以服务形式部署，优先满足 SLA（TTFT、TPOT、尾延迟），需要连续批处理、抢占、优先级、流式输出、限流和可观测性。
- **两者架构不同，是因为一个在进程内运行、一个前后端分离**；**优化重点也不同**：前者看总 tokens/s，后者看延迟约束下的 goodput（有效吞吐）。

## 第 3 题：为什么对版本兼容敏感，以及为什么建议使用独立环境

- vLLM 含大量**预编译 CUDA/C++ 扩展**，会强绑定 torch C++ ABI、CUDA toolkit、驱动、Python ABI 和 GPU 架构；版本错配就会报 `undefined symbol`、编译失败甚至静默错误。
- **独立环境**：固定 torch 和 CUDA 版本，避免与其他项目冲突，保证可复现，升级时不会污染系统 Python。

## 第 4 题：四者职责与依赖

- **Scheduler**：决定每一步让哪些请求进入 batch、prefill 还是 decode、chunked prefill、抢占与优先级、token 预算。
- **KV Cache Manager**：按 block 分配和回收显存，复用 prefix cache，执行 swap，计算剩余容量并触发 preempt。
- **ModelRunner**：组装输入（slot mapping、位置、attn metadata），执行 forward 和采样，管理 CUDA graph。
- **Attention Backend**：在 paged KV 上执行 prefill/lack decode kernel（FA、FlashInfer、Triton）。
- **依赖链**：Scheduler 依据 KV Manager 的容量决定调度，把调度结果交给 ModelRunner，ModelRunner 按 slot mapping 写入 KV，并调用 Attention Backend 读取 block table；Backend 的 page size 等能力反过来约束 KV 和调度策略。

## 第 5 题：权重放得下，仍会显存或 KV 不足

- 权重之外还有 activation、CUDA graph 捕获、NCCL/workspace、显存碎片、同卡其他进程；`gpu_memory_utilization` 是上限而非保证。
- `max_model_len` 过大：启动时按最坏情况要能装下一整段 KV，池子被压小甚至直接报错；`max_num_seqs` 过大、TP 太小、KV 用 fp16 同样会挤爆。
- **关系**：每 token KV ≈ 2 × 层数 × KV 头数 × head_dim × 字节数；单请求上限 = per-token × `max_model_len`；总池 ≈ per-token × 可用显存，并发上限 ≈ 池 ÷（per-token × 平均长度）。
- 缓解：降 `max_model_len`/`max_num_seqs`、加 TP、开 FP8 KV、`enforce_eager`。