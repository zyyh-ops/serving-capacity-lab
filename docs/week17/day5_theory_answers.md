### 第 1 题：CUDA Graph 的执行语义

- **Capture**：在 `capture_model` 中，对每个 bucket 执行 dummy run，并在 `torch.cuda.graph` 内把这段 forward 的 kernel、参数指针和依赖关系录成图，再按 `BatchDescriptor` 缓存；录制下来的是执行图，不是本次请求的结果。
- **Replay**：调用 `cudagraph.replay()` 一次提交整张图，按录制顺序执行，省掉 Python 代码和逐个 kernel 的 launch；输出复用 graph pool 中的 buffer。
- **输入必须放在固定地址**：图里存的是裸指针，`replay()` 不接受新 tensor；所以 vLLM 用预分配的持久 buffer（`input_ids`、`positions` 等，或者用 `cudagraph_copy_inputs` 先拷进固定 buffer。
- **ID/position 可变**：它们只是 buffer 里的数值，长度由 bucket 固定；host 在 replay 前写入新值，同一张图就能服务不同请求。
- **修改底层地址的后果**：replay 仍读旧指针 → 读到过期数据或已释放内存（错误结果、非法访问、静默破坏内存）；若拷贝在别的 stream 且没有 event 依赖，还会与 replay 竞争。只有 DEBUG 下才会在 replay 前断言地址。
- **为什么不同于 Kernel Fusion**：图只是把原 kernel 原样打包，kernel 数、FLOPs、访存都不变；fusion 是编译器合并 op、减少 kernel 和访存。

### 第 2 题：Shape Bucketing 与 Padding

1. **32**：最小且大于等于 23 的 bucket（`bs_to_padded_graph_size`。
2. **9**：32 − 23。
3. **在代码里，pad 行是假 token**：`slot_mapping` 填 `-1`，不写 KV；`logits_indices` 只取真实请求的末 token，采样只遍历 `num_reqs`，pad 行没有所属请求可以挂。partial prefill 的采样也会被丢弃并回退 generator。
4. **过密**：每个 bucket、每个 piece 都要 capture 和编译，导致启动更慢，graph 显存和编译缓存也更大；**过疏**：padding 比例高、白白计算 token，超过最大 bucket 就直接退回 eager。
5. **80 大于 64**：该步不进行 padding，而是走 eager；要避免这种情况，要么调大 max capture size（显存允许），要么用 chunked prefill 或限制 `max_num_batched_tokens`，把一步拆成不超过 64 个 token。

### 第 3 题：PIECEWISE 与 FULL CUDA Graph

- **区别**：PIECEWISE 在 attention 处切图，只 capture 非 attention 的编译子图，attention 每步 eager 执行、metadata 可以动态；FULL 把整个 forward 连同 attention capture 成一张图，attention metadata 也要放进固定 buffer。
- **为什么要 piecewise compilation**：capture 的单位是编译后的子图；只有按 `splitting_ops` 切开，每个 piece 才能各自包 `CUDAGraphWrapper`；`splitting_ops` 为空时，PIECEWISE 会被当成 FULL。
- **为什么 PIECEWISE 更容易**：attention 的 grid、num_splits、scheduler metadata 常依赖运行期数据，capture 时必须把它们固定成捕获时的值；PIECEWISE 把它留在图外，FULL 则要用 FA3 AOT metadata、固定 `max_num_splits`、capture ≤992 等约束。
- **Scheduler、KV Block Allocation、API Server 是否进入同一张图**：不会。图只包 `self.model` 的 forward；调度和 block 分配在 EngineCore 的 Python 代码里，API server 在另一个进程，都不进图。

### 第 4 题：性能实验中的控制变量

- **A 与 B**：Inductor 融合 norm、residual、silu 等 elementwise，生成更少更大的 Triton kernel，去掉逐 op 的 Python/ATen dispatch；kernel 数和访存都降。
- **B 与 C**：再把剩余 kernel 的 launch 和 CPU dispatch 压成每个 piece 一次 replay；scheduler、metadata、sampling 仍在图外。
- **为什么不是纯 Kernel Launch Overhead**：C 还要按 bucket padding（非整桶时多算 token），capture 时可能选不同的 kernel 路径，省掉的还有 Python/guard；所以 `T_B − T_C` = launch + dispatch + padding ± kernel 差异，可能高估也可能低估。
- **Warmup 为什么必须分离**：首次包含 Inductor 编译和 autotune、graph capture、allocator/context 初始化；仓库日志里编译约 5 秒、capture 约 1 秒，算进去就是一次性成本，不是稳态。
- **Batch=1 与 Batch=8 的加速比为什么可能不同**：launch 是每步固定成本，batch=8 每步 GPU 工作更大、固定开销占比更小，graph 的相对收益就小；padding 比例和每步 host 工作量也不同。

### 第 5 题：Nsight Timeline 与 Host/GPU Overhead

- **为什么不矛盾**：graph 只减少 host 提交次数，GPU 仍执行同样多的 kernel；PIECEWISE 下每层切成多个 piece，每个 piece 一次 replay、attention 还要单独 launch，不是整步一次图。TPOT = GPU kernel + 采样/同步 + host gap。
- **CUDA API 时间与 GPU Kernel 时间的区别**：前者是 CPU 在 driver 里提交/同步的耗时（异步，提交完就返回）；后者是 GPU 实际执行时长。
- **为什么减少 CPU Launch 次数不意味着 GPU 计算量减少**：replay 执行同一张 kernel DAG，FLOPs 和访存完全不变。
- **如何判断 GPU 是否在等待 Host 提交工作**：看 nsys——GPU 行有空洞，且下一个 kernel 的 launch API 在前一个 kernel 结束后才发出，就是 host-bound；也可比较 kernel 时长之和与步进 wall time。
- **为什么不能把所有 Kernel Duration 相加当作完整 E2E Latency**：kernel 可能跨流重叠（会重复计数），还有空洞、采样 D2H 同步、detokenize、网络和排队；相加只是 GPU busy time。
- **Graph 改善 Host Dispatch 后优先查什么**：attention/KV 带宽、batch 小时的权重带宽、sampling+D2H 每步同步、scheduler/InputBatch 的 CPU 时间、padding 浪费、piecewise 碎片化（很多小图 + eager attention）。
