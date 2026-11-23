# Day 5：Capture / Dispatch / Replay / Shape Padding 的真实调用链（vLLM 0.10.2 / V1）

> 目标：不解释「CUDA Graph 是什么」，只回答「这四件事在 vLLM 里到底是谁在什么时候调的、传了什么、日志/nsys 里能看见什么」。
> 所有行号都能在本仓库 `vllm/` 快照里直接核对（清单见 §0.1）。

---

## 0. 一页调用链

**启动期（只做一次）：把形状录成图**

```
GPUWorker.compile_or_warm_up_model()                    gpu_worker.py:320
 ├─ compile_sizes → _dummy_run(size)（预热未捕获尺寸）   :331-335
 ├─ kernel_warmup(self)                                  :340
 └─ GPUModelRunner.capture_model()                       :344 → gpu_model_runner.py:3044
     ├─ cudagraph_mode == NONE → 警告 + return 0          :3045-3049   ← 本次 no_graph 走这条
     ├─ initialize_cudagraph_capture()                    :3051 → :3225
     │   ├─ 按 attention backend 能力降级 mode（FULL→PIECEWISE）  :3235-3286
     │   └─ cudagraph_dispatcher.initialize_cudagraph_keys(mode, uniform_decode_query_len)  :3290-3292
     ├─ set_cudagraph_capturing_enabled(True)             :3077
     ├─ freeze_gc() + graph_capture(device)               :3078
     ├─ compilation_cases = reversed(self.cudagraph_batch_sizes)   :3083  ← 从大到小
     └─ _capture_cudagraphs(cases, mode, uniform_decode)  :3084 → :3122
         └─ for num_tokens in cases:                      :3138
             ├─ _dummy_run(num_tokens, mode=NONE) × cudagraph_num_of_warmups   :3139-3152
             └─ _dummy_run(num_tokens, mode=PIECEWISE)                        :3153-3157
                 └─ _dummy_run()                          :2620
                     ├─ cudagraph_dispatcher.dispatch(BatchDescriptor)        :2793-2800
                     ├─ set_forward_context(..., batch_descriptor=desc)       :2802-2808
                     └─ self.model(...)
                         └─ CUDAGraphWrapper.__call__     cuda_graph.py:106
                             ├─ entries[desc] 不存在 → 新建           :121-126
                             ├─ validate_cudagraph_capturing_enabled()  :137
                             ├─ torch.cuda.graph(cudagraph, pool=...)   :158
                             ├─ output = self.runnable(*args, **kwargs) :160
                             └─ entry.cudagraph / entry.output = ...    :172-173
```

**运行期（每个 model step）：先把形状对齐，再查表，再回放**

```
GPUModelRunner.execute_model()                           gpu_model_runner.py:2000
 ├─ _prepare_inputs(scheduler_output) → attn_metadata     :2026   （用真实 token 数建 metadata）
 ├─ _preprocess() → num_input_tokens = pad_for_cudagraph(total)   :1744-1765  ← Shape Padding
 ├─ BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)   :2043-2048
 ├─ cudagraph_dispatcher.dispatch(desc) → (runtime_mode, desc')        :2049-2050 ← Dispatch
 ├─ set_forward_context(..., cudagraph_runtime_mode=mode, batch_descriptor=desc')  :2054-2060
 └─ self.model(...)                                       :2064
     └─ CUDAGraphWrapper.__call__                         cuda_graph.py:106
         ├─ 从 forward_context 读 desc / mode              :107-109
         ├─ mode == NONE 或 != 本 wrapper 的 mode → 直接跑 runnable   :111-119
         ├─ DEBUG 下断言输入地址与捕获时一致                :182-190
         └─ entry.cudagraph.replay(); return entry.output   :192-193  ← Replay
```

四者是一条流水线：**Shape Padding 决定 key** → **Dispatch 用 key 查出 mode** → **Capture 在启动期把每个 key 录成一张图** → **Replay 在运行期把命中的图重放一次**。

### 0.1 本文可核对的源码范围

本仓库快照只有 13 个文件，CUDA Graph 相关的是：

| 文件 | 关键内容 |
|---|---|
| `vllm/compilation/cuda_graph.py` | `CUDAGraphEntry` / `CUDAGraphWrapper`（**Capture + Replay 的唯一实现处**） |
| `vllm/compilation/backends.py` | `VllmBackend` / `split_graph` / piecewise 后处理里**安装 wrapper**（`:347-364`） |
| `vllm/config/compilation.py` | `CUDAGraphMode`（`:35-70`）、`init_with_cudagraph_sizes()`（`:486-534`）、`splitting_ops`（`:536-576`） |
| `vllm/v1/worker/gpu_model_runner.py` | `capture_model` / `_capture_cudagraphs` / `_dummy_run` / `execute_model` / `_preprocess` |
| `vllm/v1/worker/gpu_worker.py` | `compile_or_warm_up_model`（cuda graph 的调用者） |
| `vllm/v1/attention/backends/flash_attn.py` | 只用来核对「pad 行不进 KV」这条边界（`:493-497`、`:531-535`） |

**不在快照里、但被调用到的**（读代码时只能看到调用点和返回值语义）：

- `vllm/v1/cudagraph_dispatcher.py`：`CudagraphDispatcher.dispatch()` / `initialize_cudagraph_keys()`。日志 `[cudagraph_dispatcher.py:102]` 就来自这里。
- `vllm/config/vllm.py`：`VllmConfig.pad_for_cudagraph()`（查 `bs_to_padded_graph_size` 表）。
- `vllm/compilation/cuda_piecewise_backend.py`：`PiecewiseBackend`，即 wrapper 里真正被录进图的 `runnable`。
- `vllm/forward_context.py`：`BatchDescriptor` / `set_forward_context()`。
- `vllm/compilation/decorators.py`：`@support_torch_compile` 触发后端编译。

## 1. Shape Padding：把 batch 对齐到「已捕获的形状」

### 1.1 捕获尺寸表在启动时就算好了

`CompilationConfig.init_with_cudagraph_sizes()`（`compilation.py:486-534`）做三件事：

```python
self.cudagraph_capture_sizes.sort(reverse=True)            # :518 降序
self.max_capture_size = self.cudagraph_capture_sizes[0]    # :519-520

# 预计算 batch size → 被 pad 到的图尺寸，数组下标直接当 hash 用
self.bs_to_padded_graph_size = [0 for i in range(self.max_capture_size + 1)]   # :523-525
for end, start in zip(self.cudagraph_capture_sizes,
                      self.cudagraph_capture_sizes[1:] + [0]):                 # :526-527
    for bs in range(start, end):
        if bs == start:
            self.bs_to_padded_graph_size[bs] = start                           # :529-530
        else:
            self.bs_to_padded_graph_size[bs] = end                             # :531-532
self.bs_to_padded_graph_size[self.max_capture_size] = self.max_capture_size     # :533-534
```

即：**每个 bs 都映射到「≥ bs 的最小已捕获尺寸」**（[16,8,4,2,1] 下：1→1，2→2，3→4，5→8，9→16）。列表最后一位补 `0`，保证 `bs` 小于最小捕获尺寸时也能落进第一个区间。

日志实测（两个模式相同）：

```
cudagraph_capture_sizes: [16, 8, 4, 2, 1]      max_capture_size: 16
```

### 1.2 运行期唯一的 padding 点

`GPUModelRunner._preprocess()`（`gpu_model_runner.py:1744-1765`）：

```python
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and not envs.VLLM_DISABLE_PAD_FOR_CUDAGRAPH
        and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):   # 1745-1747
    num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)  # 1750-1751
else:
    num_input_tokens = num_scheduled_tokens      # eager 分支，1747 条件不成立时不 pad
num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)   # 1763-1765（DP 专用）
```

三个条件必须同时成立才 pad，任何一个不满足就直接用真实 token 数：

1. `cudagraph_mode != NONE` —— 这就是 `no_graph` 与 `piecewise` 的差别之一；
2. `VLLM_DISABLE_PAD_FOR_CUDAGRAPH` 未设置（逃生开关）；
3. `num_scheduled_tokens <= cudagraph_batch_sizes[-1]` —— 注意 `cudagraph_batch_sizes` 是**升序**的（`:300-307`，`reversed(config)`），所以 `[-1]` 就是最大捕获尺寸 `16`。**超过 16 的 batch 一律不 pad、也不会有图**。

接着 `num_input_tokens` 只用于切片张量，不改变 metadata：

```python
input_ids = self.input_ids.gpu[:num_input_tokens]      # :1798  持久缓冲，不是新张量
positions = self.positions.gpu[:num_input_tokens]      # :1804
```

### 1.3 为什么必须 pad、代价落在哪

- **必须 pad**：`torch.cuda.graph` 录下的是「具体形状的 kernel 参数」。同一个图只能重放同一形状，所以 `num_tokens` 必须先落到表里的某个值。`cudagraph_copy_inputs=false`（`:285-292`）时连输入地址都不能变，只换内容。
- **代价只落在图内算子**：`attn_metadata` 在 `_prepare_inputs`（`:2026`）里就按**真实** token 数建好了，`num_input_tokens` 是之后才被撑大的。多出来的行只参与图内的 GEMM / RMSNorm / 激活，不参与 attention —— 因为 attention 本身就是 `splitting_ops`，被切在**图外**（见 §4.4）。
- **pad 行不会写进 KV cache**：attention 侧留了一句注释解释这件事（`flash_attn.py:493-497`）——「key/value 是 padded 的，`slot_mapping` 不是；`reshape_and_cache_flash` 用 `slot_mapping` 的 shape 决定实际 token 数」。后面 `flash_attn_varlen_func(q=query[:num_actual_tokens], ...)`（`:531-535`）也只算真实行。所以 padding 只浪费线性层的算力，不会污染 KV、也不会污染 logits。
- **padding 是幂等的**：`_preprocess` 已经把 `num_input_tokens` 对齐过一次，`dispatch` 再拿它查表时通常原样返回；真正需要「向上取整」的是 FULL 图那条路径（未捕获尺寸 → 最近上界）。

### 1.4 本次实验里 padding 的真实取值

| 场景 | `num_scheduled_tokens` | `num_input_tokens` | 命中图 |
|---|---|---|---|
| 单请求 decode（server / offline） | 1 | 1 | ✅ size=1 |
| 8 并发 decode（server batch8） | 8 | 8 | ✅ size=8 |
| prefill（prompt ≈100 token，单请求） | ≈100 | ≈100（>16，不 pad） | ❌ 走 eager |
| chunked prefill 尾块（≤16 token） | 3 | 4 | ✅ size=4（本次未触发） |

这里有一个容易误读的地方：**本次所有 speedup 都只来自 decode**。`max_num_batched_tokens=256` 但 `max_capture_size=16`，所以 prefill 段（100+ token）永远在 16 以上，`_preprocess` 不 pad、`dispatch` 也不会给图。nsys 里 `cudaGraphLaunch` 只覆盖 95 个 decode step 就是这个原因（§3.4）。

## 2. Capture：`capture_model` → `_capture_cudagraphs` → `_dummy_run` → `torch.cuda.graph`

### 2.1 入口与三种「录哪些批次」

```python
def capture_model(self) -> int:                                          # gpu_model_runner.py:3044
    if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning("Skipping CUDA graph capture. ...")               # :3045-3049
        return 0
    else:
        self.initialize_cudagraph_capture()                              # :3051
```

注意 `return 0` 是**在** `initialize_cudagraph_capture()` 之前——所以 `cudagraph_mode=NONE` 时 dispatching keys 也不会被初始化，这是 §3.2 那个 warning 的根因。

`initialize_cudagraph_capture()`（`:3225-3292`）先根据 attention backend 的 `cudagraph_support` 能力**降级/改写** mode（FULL→FULL_AND_PIECEWISE 或 FULL_DECODE_ONLY，`:3235-3286`），再填 key 表（`:3290`）。

`capture_model` 主体（`:3074-3120`）：

```python
set_cudagraph_capturing_enabled(True)                    # :3077 之后才允许 capture
with freeze_gc(), graph_capture(device=self.device):     # :3078 关 GC，进 capture 上下文
    if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
        compilation_cases = list(reversed(self.cudagraph_batch_sizes))   # :3083 16,8,4,2,1
        self._capture_cudagraphs(compilation_cases,
                                 cudagraph_runtime_mode=cudagraph_mode.mixed_mode(),
                                 uniform_decode=False)                    # :3084-3087
    if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL \
            and cudagraph_mode.separate_routine():                        # :3091-3092
        ...  # 只有 FULL decode 才额外录一套 uniform_decode=True 的图
```

- **`reversed` 是为了 pool 复用**：注释写得很直白（`:3075-3076`），先录大形状，小形状能复用大形状已经占住的显存池。
- **PIECEWISE 只有 mixed 一套 key**：`:3091` 的条件对 `cudagraph_mode=1` 不成立，所以本次只捕获 `uniform_decode=False` 的那一套。decode step 能命中，靠的是 dispatch 对 key 的归一化（§3.4 的反证）。

### 2.2 每个尺寸录几次前向

```python
for num_tokens in compilation_cases:                              # :3138
    for _ in range(self.compilation_config.cudagraph_num_of_warmups):
        self._dummy_run(num_tokens, cudagraph_runtime_mode=CUDAGraphMode.NONE, ...)  # :3139-3152
    self._dummy_run(num_tokens, cudagraph_runtime_mode=cudagraph_runtime_mode, ...)  # :3153-3157
```

`cudagraph_num_of_warmups=1`（日志里能看到），所以每个尺寸是 **1 次 eager 预热 + 1 次带图前向**。预热那次用 `NONE`（`:3140-3144` 的注释强调「预热用 NONE」与「attention 要不要预热」是两件正交的事），效果上是让 cublasLt heuristic、Triton autotune、workspace 分配在录图之前先发生，避免把「第一次运行」的初始化代理一起录进去。

### 2.3 图到底是被谁「录」下来的

`_dummy_run` 只负责把输入准备好、把 context 设好，然后 `self.model(...)`（`:2809-2815`）。真正录图的是编译期装到子模块上的 `CUDAGraphWrapper`：

```python
# backends.py:347-364
if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
    static_graph_wrapper_class = resolve_obj_by_qualname(
        current_platform.get_static_graph_wrapper_cls())
    self.module.__dict__[target] = static_graph_wrapper_class(
        runnable=piecewise_backend,                  # 编译后的一个 piece
        vllm_config=self.vllm_config,
        runtime_mode=CUDAGraphMode.PIECEWISE,        # 注意：一律标 PIECEWISE
        cudagraph_options=CUDAGraphOptions(
            debug_log_enable=piecewise_backend.is_first_graph,
            gc_disable=not piecewise_backend.is_first_graph,
            weak_ref_output=piecewise_backend.is_last_graph))
else:
    self.module.__dict__[target] = piecewise_backend  # NONE：不包 wrapper，纯 eager 调用
```

两点值得注意：

1. **wrapper 安装发生在 torch.compile 的后处理**，与 `capture_model` 是两件事。所以 `no_graph` 里模型依然是「被切成 29 段 + Inductor 编译好」的，只是每段没包 wrapper、直接 Python 调过去（这正是它慢但不算 eager 的原因）。
2. `runtime_mode` 恒为 `PIECEWISE`，即使外层是 FULL 图；FULL 与 PIECEWISE 的区分靠 forward context 里的 `cudagraph_runtime_mode`（`cuda_graph.py:109-119` 会比对），两套 wrapper 嵌套时靠这个 disambiguate。

录图本体（`cuda_graph.py:128-180`）：

```python
cudagraph = torch.cuda.CUDAGraph()                              # :143
with ExitStack() as stack:
    if self.cudagraph_options.gc_disable:                       # 除第一段外都关 GC
        stack.enter_context(patch("gc.collect", lambda: None))
        stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))
    with torch.cuda.graph(cudagraph, pool=self.graph_pool):     # :158 全局 pool，跨段共享
        output = self.runnable(*args, **kwargs)                 # :160 真前向
        if self.cudagraph_options.weak_ref_output:
            output = weak_ref_tensors(output)                   # :168 只留弱引用省显存
entry.output = weak_ref_tensors(output)                         # :172
entry.cudagraph = cudagraph                                     # :173
compilation_counter.num_cudagraph_captured += 1                 # :175
return output                                                   # :180 强引用返回给上层（内存归 pytorch 管）
```

`gc_disable` / `weak_ref_output` 的意思在注释里：piecewise 一层一张图，每张都跑一遍 GC 会让 capture 慢到不可接受（`:146-155`）；weak ref 只在最后一段安全，因为最后一段的输出不会再被别的图消费（`:161-167`）。

### 2.4 日志对照（这就是 §2.1-2.3 的实测证据）

`results/day5_piecewise_server.log`：

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████| 5/5 [00:00<00:00, 36.40it/s]
INFO [gpu_model_runner.py:3118] Graph capturing finished in 1 secs, took 0.09 GiB
```

- `5/5` = `cudagraph_capture_sizes` 的 5 个尺寸，**每个尺寸内部还有 29 个 piecewise 子图**（tqdm 只按尺寸计数）；总量 = 5×29 = **145 张图**，平均 0.09 GiB / 145 ≈ **0.65 MiB/图**。
- 进度条名字里的 `mixed prefill-decode` + `PIECEWISE` 直接对应 `:3134-3136` 的 `desc` 字符串，也就是本次只录了 mixed 那一套。

`results/day5_no_graph_server.log`：

```
WARNING [gpu_model_runner.py:3046] Skipping CUDA graph capture. To turn on CUDA graph capture,
        ensure `cudagraph_mode` was not manually set to `NONE`
```

没有进度条、没有 `Graph capturing finished`、`CUDAGraph memory 0.0 GiB` —— 三处对得上。

## 3. Dispatch：每个 step 把 `(num_tokens, uniform_decode)` 换成 `runtime_mode`

### 3.1 调用点只有一处（运行期）

`execute_model()`（`gpu_model_runner.py:2043-2050`）：

```python
uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
    num_scheduled_tokens == self.input_batch.num_reqs * max_query_len)   # :2043-2046
batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                   uniform_decode=uniform_decode)         # :2047-2048
cudagraph_runtime_mode, batch_descriptor = \
    self.cudagraph_dispatcher.dispatch(batch_descriptor)                  # :2049-2050
```

- `uniform_decode_query_len` = 1（无投机解码时，`:377-378`），所以纯 decode step 的 `uniform_decode=True`；prefill / 混合批为 `False`。
- 返回值里的 `batch_descriptor` **覆盖**了原值：dispatch 返回的 descriptor 才是后面 `set_forward_context` 写进 context、也是 wrapper 查表用的 key。

### 3.2 key 表什么时候被填；没填会怎样

key 表在 `initialize_cudagraph_capture()` 末尾填（`:3290-3292`）：

```python
self.cudagraph_dispatcher.initialize_cudagraph_keys(
    self.compilation_config.cudagraph_mode, self.uniform_decode_query_len)
```

而这个函数只在 `capture_model()` 的 `else` 分支里被调用（`:3048-3051`）。所以 `cudagraph_mode=NONE` 时：**图没捕获 + key 表空**，`dispatch` 每次都退化成 eager，并且打一条 warning（`no_graph` 日志）：

```
WARNING [cudagraph_dispatcher.py:102] cudagraph dispatching keys are not initialized.
        No cudagraph will be used.
```

`piecewise` 日志里没有这行 —— 因为 key 表已初始化，且每步都能查到。

### 3.3 结果从哪进模型：`set_forward_context`

```python
with set_forward_context(attn_metadata, self.vllm_config,
                         num_tokens=num_input_tokens,
                         num_tokens_across_dp=num_tokens_across_dp,
                         cudagraph_runtime_mode=cudagraph_runtime_mode,
                         batch_descriptor=batch_descriptor), ...:      # :2054-2060
    model_output = self.model(...)                                     # :2064
```

这是 **dispatch 与 wrapper 之间唯一的数据通路**（contextvar / thread-local，不在函数参数里）。`CUDAGraphWrapper.__call__` 开头立刻读它：

```python
forward_context = get_forward_context()                    # cuda_graph.py:107
batch_descriptor = forward_context.batch_descriptor        # :108
cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode  # :109
```

所以「wrapper 不知道自己该不该回放」这个问题的答案是：**它不看自己，只看 context**（`:111-119`），mode 不匹配就直接透传给 `runnable`。这也是同一个模型里 FULL 图、PIECEWISE 图、eager 能共存的原因。

### 3.4 一个能反推 dispatch 行为的观测

`day5_piecewise_profile.sqlite` 里 `cudaGraphLaunch` 共 **2,755** 次，按 >1 ms 间隔切分正好是 **95 组 × 29 次**：

- 95 = 96 个输出 token 中扣掉 prefill 那一 step；
- 29 = 28 个 attention op 形成 28 个切分点 → 29 段图（Qwen3-0.6B 是 28 层，与 `reshape_and_cache` 的 2,688 = 28×96 次调用对得上）；
- 每组 29 次，说明**每个 decode step 都命中同一批 key，没有发生运行期补捕获**。

反证：若 dispatch 返回的 descriptor 与捕获时的 key 不一致，wrapper 会在 `entries` 里查不到、走到 `:128` 的捕获分支，而此时 `set_cudagraph_capturing_enabled(False)` 已生效（`:3111`），`validate_cudagraph_capturing_enabled()`（`:137`）会直接抛错。进程没炸，说明 key 严格一致。

## 4. Replay：`graph.replay()` 加两条隐含契约

### 4.1 `__call__` 的三个分支

```python
if cudagraph_runtime_mode == CUDAGraphMode.NONE or \
        cudagraph_runtime_mode != self.runtime_mode:
    return self.runnable(*args, **kwargs)          # ① NONE / 模式不匹配 → eager

if batch_descriptor not in self.concrete_cudagraph_entries:
    self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(...)   # ② 新 key：建空条目

entry = self.concrete_cudagraph_entries[batch_descriptor]
if entry.cudagraph is None:
    ... capture ...                                 # ③ 首次：现场录（启动期走的这条）
    return output

if self.is_debugging_mode:                          # ④ 已录：DEBUG 校验输入地址，然后回放
    assert new_input_addresses == entry.input_addresses
entry.cudagraph.replay()                            # :192
return entry.output                                 # :193
```

「懒捕获」是**存在**的（`:128`），但 vLLM 在启动期把所有 key 都录满、随后 `set_cudagraph_capturing_enabled(False)`（`:3111`），所以运行期走到 ③ 就是 bug 而不是优化。

### 4.2 契约一：输入必须在同一批地址上

`replay()` 不拷贝任何输入，图里记录的是**捕获时那些 tensor 的地址**。所以运行期必须有「形状与地址都固定」的输入缓冲，这就是 `__init__` 里预分配的那批持久缓冲（`gpu_model_runner.py:312-330`）：

```python
self.input_ids   = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions   = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)
self.seq_lens    = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
```

`_preprocess` 里的 `self.input_ids.gpu[:num_input_tokens]`（`:1798`）是**切片视图**，地址 = base + 0，与捕获时一致；每个 step 只改内容（`_prepare_inputs` 往这几个 buffer 里写）。`VLLM_LOGGING_LEVEL=DEBUG` 时的断言（`:182-190`）就是给这条契约上的保险 —— 默认日志级别下不打这个断言，违规只会「静默算错」。

若确实无法保证地址固定，才有 `cudagraph_copy_inputs=True`（`compilation.py:285-292`）：编译期额外插一层 `copy_and_call`（`backends.py:597-627`）把输入搬进内部缓冲，代价是多一次拷贝。

### 4.3 契约二：输出是图池里的同一块内存

`entry.output` 是弱引用（`:172`），`replay()` 后返回的是**同一批 tensor**（`:193`）。上层拿到后必须在本 step 内消费完（`hidden_states[logits_indices]` → `compute_logits`，`gpu_model_runner.py:2101-2102`），不能跨 step 持有。

### 4.4 为什么 attention 始终在图外

`cudagraph_mode=PIECEWISE` 时 `splitting_ops` 默认就是所有 attention op（`compilation.py:336-345`、`:536-576`，日志里 `"splitting_ops":["vllm.unified_attention", ..., "vllm.gdn_attention"]`）。切分点上的 attention 不参与录图，理由是它的 kernel 参数随 `seq_lens` / `block_table` / `num_splits` 变化，录不进固定形状。

实测正好对上：`cudaLaunchKernel` 直接 launch 的 8,453 个 kernel 里，与 attention 相关的 **7,196 个**（2,688 个 `reshape_and_cache_flash` + 4,508 个 `flash_fwd_*`）全部在图外；反过来说 gemvx（线性层）10,736 个里有 10,640 个在图内、Triton（norm/激活/残差）21,600 个里有 21,375 个在图内。

## 5. 四者关系（一张表）

| 阶段 | 触发时机 | 入口 | 关键数据结构 | 本次实测证据 |
|---|---|---|---|---|
| Shape Padding | 每个 step，`_preprocess` | `pad_for_cudagraph(total)`，`gpu_model_runner.py:1750` | `bs_to_padded_graph_size[]`（`compilation.py:523-534`） | decode 1→1、8→8；prefill ≈100→≈100（不 pad） |
| Capture | 启动期一次 | `capture_model`，`:3044` | `CUDAGraphEntry{descriptor, cudagraph, output}`（`cuda_graph.py:24-31`） | `5/5` 进度条 + `took 0.09 GiB` + 145 张图 |
| Dispatch | 每个 step，`execute_model` | `cudagraph_dispatcher.dispatch`，`:2050` | `BatchDescriptor(num_tokens, uniform_decode)` + `cudagraph_keys` | `no_graph` 的 `keys are not initialized`；`piecewise` 每步 29 次 graph launch |
| Replay | 每个 step，`self.model(...)` 内部 | `CUDAGraphWrapper.__call__`，`cuda_graph.py:192` | `forward_context.descriptor` → `entries[desc]` | 2,755 次 `cudaGraphLaunch`；图内 32,015 个 kernel（79%） |

## 6. 三个坑 + 一句话结论

**坑 1：`no_graph` ≠ `enforce_eager`。** `cudagraph_mode=0` 只是不录图，torch.compile（level 3）+ piecewise 切分照旧（`backends.py:366`），所以它比真 eager 快，但每个 piece 仍是 Python 调用 + 每个 kernel 一次 launch。

**坑 2：`max_capture_size` 决定了「有没有图」，不是 `max_num_batched_tokens`。** 本次 `max_num_batched_tokens=256` 却只捕到 16，于是 100+ token 的 prefill 全程无图。想让 prefill 也吃上图，得把 `cudagraph_capture_sizes` 显式往上铺。

**坑 3：`0.09 GiB` 这个数字不能外推。** 它 = 5 个尺寸 × 29 段，随层数线性长；同时它必须小于 `gpu_worker.py:360` 预留的 150 MiB 冗余，否则 KV cache 会被挤掉（本次 9029 块 / 144,464 token 在两个模式下完全一致）。

> **一句话**：`pad_for_cudagraph` 把不规则的 batch 压成 5 个形状，`dispatch` 用这 5 个形状查出「该用哪套图」，`CUDAGraphWrapper` 在启动期把这些形状录成 145 张图、在运行期只做 `replay()` 一件事 —— 于是每 step 的 CPU 调用从 ~420 次 kernel launch 降到 29 次 graph launch，GPU 空转从 86% 降到 69%。
