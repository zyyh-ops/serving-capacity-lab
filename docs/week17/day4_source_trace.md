# Day 4：Worker → GPUModelRunner → Qwen3/FlashAttention → Sampler 源码追踪（vLLM 0.10.2 / V1）

---

## 0. 一页调用链

```
EngineCore.step()                              core.py
 └─ Executor.execute_model(SchedulerOutput)    TP=1 → UniProcExecutor，直接调 Worker
    └─ Worker.execute_model()                  gpu_worker.py:425
       └─ GPUModelRunner.execute_model()       gpu_model_runner.py:2000
          ├─ _update_states()                  :491   ← 请求生命周期 + InputBatch 对齐
          ├─ _prepare_inputs()                 :862   ← positions / slot / qsl / attn_metadata
          ├─ _preprocess()                     :1736  ← padding、input_ids/positions 切片
          ├─ set_forward_context(attn_metadata) :2054
          ├─ self.model(...)                   :2064  ← Qwen3 forward（每层 Attention）
          ├─ hidden_states[logits_indices]     :2101
          ├─ self.model.compute_logits()       :2102  ← lm_head → logits
          ├─ _sample() → Sampler.forward()     :2118 / sampler.py:70
          ├─ _bookkeeping_sync()               :2120  ← 唯一 GPU→CPU 同步点
          └─ ModelRunnerOutput(...)            :2150  ← outputs.py:85
```

`Worker.execute_model` 本身几乎不做事，只有三件：**PP 收中间张量、转发、PP 发中间张量**。

---

## 1. `Worker.execute_model()` → `GPUModelRunner.execute_model()`

```python
# vllm/v1/worker/gpu_worker.py:424
@torch.inference_mode()
def execute_model(self, scheduler_output):
    intermediate_tensors = None
    forward_pass = scheduler_output.total_num_scheduled_tokens > 0
    if forward_pass and not get_pp_group().is_first_rank:        # 非首段：从上一段收
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group()))
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)  # ★真实入口
    if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
        return output                                            # 末段：直接回 ModelRunnerOutput
    ...                                                          # 中间段：send_tensor_dict 后返回 None/空 output
```

链的上一层是 `Executor`：TP=PP=1 时 `UniProcExecutor` 在**同进程**直接调用，没有 RPC；TP>1 才变成 `collective_rpc("execute_model", ...)`（day2 §3.3）。所以本实验（`tensor_parallel_size=1`）里 `Worker` 与 `GPUModelRunner` 是同一个 pid 内的两层 Python 调用。

`GPUModelRunner.execute_model` 的六个阶段（`gpu_model_runner.py:2000-2169`）：

| 阶段 | 行号 | 关键动作 | 输出 |
|---|---|---|---|
| Preprocess | 2006 | `_update_states`；`total==0` 时提前返回 `EMPTY_MODEL_RUNNER_OUTPUT`（2007-2012） | InputBatch 已对齐 |
| | 2026 | `_prepare_inputs` | `attn_metadata`（dict）、`logits_indices` |
| | 2041 | `_preprocess` | `input_ids/positions` 切片、`num_input_tokens` |
| Forward | 2054-2070 | `set_forward_context(...)` + `self.model(...)` | `hidden_states [num_tokens, H]` |
| Postprocess | 2101-2102 | `hidden_states[logits_indices]` → `compute_logits` | `logits [num_reqs, V]` |
| Sample | 2118 | `_sample` → `Sampler` | `SamplerOutput` |
| Bookkeep | 2120 | `_bookkeeping_sync` | CPU 侧 token id 列表 + logprobs |
| Output | 2150 | 组装 `ModelRunnerOutput`（异步调度时包成 `AsyncGPUModelRunnerOutput`） | 回 Scheduler |

两个易错点：

1. **`total_num_scheduled_tokens == 0` 的早退在 `_update_states` 之后**（2007）。所以「空 step」也会清理已完成请求的 batch 状态，只是不启动任何 kernel。
2. `enforce_eager=True` 时 `cudagraph_mode=NONE`，`_preprocess` 走 eager 分支（1745-1761）**不做 batch padding**，`num_input_tokens == total_num_scheduled_tokens`；只有开 CUDA graph 时才会 `pad_for_cudagraph`。本实验日志里的 `query_start_loc[-1] == total` 恒成立，正是这个原因。

---

## 2. `_update_states()` 对 `InputBatch` 的更新职责

`InputBatch`（`gpu_input_batch.py:72`）是**按 req_index 对齐的持久批缓冲**，核心是四个按行对齐的结构：

| 结构 | 形状 | 谁读 |
|---|---|---|
| `_req_ids` / `req_id_to_index` | `list[str]` / dict | 所有下游都用它定义「batch 内顺序」 |
| `token_ids_cpu` | `[max_num_reqs, max_model_len]` | `_prepare_inputs` 取 `input_ids` |
| `num_computed_tokens_cpu` | `[max_num_reqs]` | 算 `positions`、`seq_lens` |
| `block_table`（`MultiGroupBlockTable`） | `[max_num_reqs, max_num_blocks]` | `block_table_tensor` + slot mapping |
| `num_tokens` / `num_tokens_no_spec` | `[max_num_reqs]` | 判 chunked prefill 是否算完 |

`_update_states`（491-668）按固定五段推进：

```
① finished_req_ids      → requests.pop() (:502) + input_batch.remove_request() (:511)
② unscheduled_req_ids   → remove_request() (:529)
      = 被抢占 / 本步未调度的请求；只从 batch 里摘掉，CachedRequestState 保留，
        因为它的 block_ids 和 num_computed_tokens 还要用来恢复
③ scheduled_new_reqs    → 建 CachedRequestState（555-568）→ requests[req_id]（569）→ reqs_to_add
④ scheduled_cached_reqs → 逐请求更新（580-656）
      num_computed_tokens_cpu[idx] = req_data.num_computed_tokens   (:627)
      new_block_ids 非空 → block_table.append_row(new_block_ids, idx) (:630)
⑤ reqs_to_add → add_request() (:661) → condense() (:664)
                 → _may_reorder_batch() (:666) → refresh_metadata() (:668)
```

### 2.1 三个真正的职责边界

1. **块表语义由 `resumed_from_preemption` 决定**（606-616）。`False` → 把增量 `new_block_ids` **append** 到已有行（RUNNING 请求走这里）；`True` → 整表**替换**（被抢占后恢复）。这与 day3 §1.2 里 scheduler 侧 `req_to_new_blocks` 存增量/全量的不对称是同一个布尔的上下游。
2. **`token_ids_cpu` 只在非末段（PP）时由 scheduler 回填**（635-644）。单机 PP=1 时**不写**：采样出的 token 在 `_bookkeeping_sync`（1980-1983）里就已经写进 `token_ids_cpu` 了，scheduler 再送一遍是重复劳动。这也是「scheduler 不需要把 sampled token 发回 worker」的原因（注释 1958-1962）。
3. **物理 KV 块不在这里释放**。`_update_states` 只删 Python 侧状态和 batch 行；`kv_cache_manager.free()` 发生在 `Scheduler.update_from_output` 判停之后（day3 §4）。worker 对块表永远只做「加行/换行」，不做回收。

### 2.2 `condense()` 是「顺序会变」的根源

`condense()`（`gpu_input_batch.py:543-649`）把末尾非空行**下移**填空洞（`move_row`），然后截断尾部。因此**同一个请求在不同 step 的 req_index 可以不同**。这条约束往下传染成一条硬规矩：

> `_prepare_inputs` 里所有数组的「第 i 个元素」都必须按 `self.input_batch.req_ids` 的当前顺序生成，不能缓存上一次的 index。

`refresh_metadata()`（651-667）只在 batch 真有变化时才重建 `SamplingMetadata`（temperature/top_p/top_k/penalty/generators/logitsprocs），否则复用上一份——这是 `_sample` 之外唯一和采样相关的 CPU 开销。

---

## 3. `_prepare_inputs()`：Query Start / Positions / Slot Mapping

`_prepare_inputs`（862-1098）的顺序是刻意排的：**能提前的 H2D 先发，纯 CPU 计算在后，尽量拖到需要时才同步**。

```python
# gpu_model_runner.py:880
self.input_batch.block_table.commit_block_table(num_reqs)   # ① 先把块表 H2D，与后面 CPU 计算重叠

req_ids = self.input_batch.req_ids                          # ② batch 内顺序，唯一权威
tokens  = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]   # 883-886
num_scheduled_tokens = np.array(tokens, dtype=np.int32)
req_indices  = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)   # 890  [2,5,3]→[0,0,1,1,1,1,1,2,2,2]
cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)   # 895  [2,7,10], [0,1,0,1,2,3,4,0,1,2]

positions_np = self.positions.np[:total]
np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)   # 899-902 ★Positions

token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]      # 913-914
torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(), 0,
                   torch.from_numpy(token_indices),
                   out=self.input_ids.cpu[:total])                                        # 916-922 ★Input IDs

self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)   # 924 ★Slot Mapping
self.input_batch.block_table.commit_slot_mapping(total)                        # 926

self.query_start_loc.np[0] = 0
self.query_start_loc.np[1:num_reqs+1] = cu_num_tokens          # 930-931 ★Query Start
self.query_start_loc.np[num_reqs+1:].fill(cu_num_tokens[-1])   # 934  pad 成非降（FA 要求）
self.query_start_loc.copy_to_gpu()

self.seq_lens.np[:num_reqs] = num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens   # 938-940
```

### 3.1 三个数组的语义（一句话各自）

| 数组 | 定义 | 消费方 |
|---|---|---|
| `positions` | `num_computed_tokens[req] + 该请求内第 j 个 query 的偏移` | RoPE；决定 attention「我在哪」 |
| `query_start_loc` | `cu_num_tokens`，长度 `num_reqs+1`，首位恒 0 | `cu_seqlens_q`，切分 varlen batch |
| `slot_mapping` | 每个 scheduled token 对应的**物理 KV 槽位** | `reshape_and_cache_flash` 写 KV |

### 3.2 Slot Mapping 的真实算法（`block_table.py:123-129`）

```python
block_table_indices = req_indices * max_num_blocks_per_req + positions // block_size
block_numbers       = block_table_np.ravel()[block_table_indices]     # 逻辑页 → 物理块号
slot_mapping        = block_numbers * block_size + positions % block_size
```

注意它**不是** `token_indices // block_size`：`max_model_len` 未必被 `block_size` 整除（源码注释 97-99），所以行跨距用 `max_num_blocks_per_req` 而不是 `max_model_len / block_size`。

### 3.3 `logits_indices`：只对「每个请求的最后一个 query」算 logits

```python
logits_indices = query_start_loc[1:] - 1     # :967
```

chunked prefill 下一个请求可能只算了一半 prompt，但它**照样会产出一个 logits**（`query_start_loc[1:]-1` 指向该请求本步最后一个 token）。这个 token 的采样结果在 `_bookkeeping_sync`（1892-1901）里被判定为 `seq_len < req_state.num_tokens` 后丢弃，并且**回退 RNG offset**（`generator.set_offset(offset - 4)`）——否则「多采一次」会污染随机数序列。投机解码时这条路径换成 `_calc_spec_decode_metadata`（980-982）。

### 3.4 实测对齐（`results/day4_runner_trace.log` step 53）

```
scheduled_tokens = [1, 60, 67], total = 128
query_start_loc   = [0, 1, 61, 128]
seq_lens          = [54, 187, 67]
```

反推 `num_computed_tokens`：A `54-1=53`、B `187-60=127`、C `67-67=0`。即：

- A 是纯 decode（prompt 52，已生成 2 个），`positions = [53]`，`logits_index = 0`；
- B 是 chunked prefill 的尾段（prompt 187，已算 127、本步算 60 → 到 187 收尾），`positions = [127..186]`，`logits_index = 60`；
- C 是新请求的**首段** chunked prefill（prompt 82，本步只算 67，未算完），`positions = [0..66]`，`logits_index = 127`。

三者在**同一个 varlen batch、同一次 kernel launch** 里完成——这就是 `query_start_loc` + `positions` 存在的意义，也是它和「一次只跑一个请求」的 Triton kernel 最本质的区别（见 §6）。

---

## 4. Qwen3 Attention → FlashAttention Backend

### 4.1 静态接线：backend 是在 `__init__` 里定死的

```
Qwen3Attention.__init__                         qwen3.py:98-138
 ├─ QKVParallelLinear / RowParallelLinear        :98-113
 ├─ get_rope(...)                                :115
 ├─ Attention(num_heads, head_dim, scaling, ...) qwen3.py:123 → attention/layer.py:58
 │   ├─ get_attn_backend(head_size, dtype, ...)  layer.py:171 → selector.py:140
 │   │    └─ current_platform.get_attn_backend_cls(...)   selector.py:205
 │   │         └─ cuda.py:355-364  SM≥8.0 且 FA 可导入 → FLASH_ATTN_V1
 │   │            logger.info_once("Using Flash Attention backend on V1 engine.")  cuda.py:362
 │   ├─ impl_cls = FlashAttentionBackend.get_impl_cls()   flash_attn.py:67
 │   └─ self.impl = FlashAttentionImpl(...)               flash_attn.py:369
 └─ q_norm / k_norm = RMSNorm(head_dim)          :137-138
```

日志里 `Using Flash Attention backend on V1 engine.`（`results/day4_server.log`）就是 `cuda.py:362` 这一行，可以把它作为「backend 选型确实走了 FA」的物证。后端名字 `FlashAttentionBackend.get_name()` 返回 `"FLASH_ATTN_VLLM_V1"`（`flash_attn.py:64-65`）。

### 4.2 运行期调用链（每层每步走一遍）

```
Qwen2Model.forward                                qwen2.py:342
 └─ for layer in layers: layer(positions, hidden_states, residual)   qwen2.py:361-365
    └─ Qwen3DecoderLayer.forward                  qwen3.py:217
       ├─ input_layernorm → residual 更新          :224-229
       └─ self_attn(positions, hidden_states)      :230
          └─ Qwen3Attention.forward                qwen3.py:140
             ├─ qkv_proj → split(q,k,v)            :145-146
             ├─ q_norm / k_norm（per-head RMSNorm） :148-155
             ├─ rotary_emb(positions, q, k)        :156
             ├─ self.attn(q, k, v)                 :157
             │   └─ Attention.forward              layer.py:223
             │      ├─ use_direct_call = not opaque_attention_op()   layer.py:193 / cuda.py:480-481
             │      │     CUDA 上为 False → 走自定义 op 而不是直接 Python 调用
             │      ├─ torch.ops.vllm.unified_attention_with_output(...)   layer.py:280
             │      │   └─ unified_attention_with_output()          layer.py:505
             │      │      ├─ attn_metadata = forward_context.attn_metadata[layer_name]  :517-518
             │      │      ├─ kv_cache = self.kv_cache[forward_context.virtual_engine]   :520
             │      │      └─ self.impl.forward(self, q,k,v, kv_cache, attn_metadata)    :521
             │      │         └─ FlashAttentionImpl.forward          flash_attn.py:422
             │      │            ├─ reshape_and_cache_flash(k, v, k_cache, v_cache, slot_mapping, ...)  :498
             │      │            └─ flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=query_start_loc,
             │      │                  seqused_k=seq_lens, block_table=..., max_seqlen_q/k, ...)          :531
             └─ o_proj(attn_output)                :158
```

四个关键点：

1. **`attn_metadata` 是 dict，按 `layer_name` 索引**（`layer.py:269-270, 476-477`）。同一个 KV cache group 里的所有层**共享同一个 metadata 对象**（`gpu_model_runner.py:1089-1090`），因此 FA 的 AOT scheduler metadata 只算一次。
2. **KV 写入和 attention 在同一次 `impl.forward` 里**（`flash_attn.py:498` 先写、`:531` 再读），两者共用同一份 `slot_mapping` / `block_table`。这就是「paged decode」在真实引擎里的形态：写页与读页用同一套索引，不需要额外的物化步骤。
3. `reshape_and_cache_flash` 用 `slot_mapping` 的 shape 决定 token 数（注释 493-497），所以 KV 不需要 `[:num_actual_tokens]` 切片；而 `flash_attn_varlen_func` 的 q/out 必须切（`:532, :535`）。
4. **cascade attention 是另一个分支**（`:556-580`），当 `common_prefix_len > 0` 时把「共享前缀」和「各请求后缀」拆成两次 kernel。是否启用由 `_compute_cascade_attn_prefix_len`（`gpu_model_runner.py:1100-1193`）+ `use_cascade_attention`（`flash_attn.py:363`）决定。

### 4.3 metadata 两段式：`CommonAttentionMetadata` → `FlashAttentionMetadata`

```
_prepare_inputs                         gpu_model_runner.py:1041-1057
 └─ CommonAttentionMetadata(query_start_loc, seq_lens, num_computed_tokens_cpu,
                            block_table_tensor, slot_mapping, num_actual_tokens, ...)
     └─ attn_group.metadata_builder.build(common_prefix_len, common_attn_metadata)   :1084
        └─ FlashAttentionMetadataBuilder.build                        flash_attn.py:224
             ├─ aot_schedule? get_scheduler_metadata(...)             :271-286
             └─ FlashAttentionMetadata(num_actual_tokens, query_start_loc, seq_lens,
                                       block_table, slot_mapping, use_cascade, ...)   :344-360
```

`CommonAttentionMetadata` 是**与 backend 无关的公共字段**，`FlashAttentionMetadata` 才是 FA 私有字段（cascade 的 `cu_prefix_query_lens` / `suffix_kv_lens` / `scheduler_metadata` / `max_num_splits`，`flash_attn.py:112-141`）。这一层抽象是 0.10.x 才有的：换 backend 只需换 builder，`_prepare_inputs` 不动。

---

## 5. Hidden States → Logits → Sampler → ModelRunnerOutput

```
hidden_states = self.model(input_ids, positions, ...)        gpu_model_runner.py:2064
    # [num_input_tokens, hidden]；eager 时 num_input_tokens == total_scheduled_tokens
sample_hidden_states = hidden_states[logits_indices]         :2101  # 只留每请求最后一个 query 位
logits = self.model.compute_logits(sample_hidden_states, None)  :2102
    └─ Qwen3ForCausalLM.compute_logits                      qwen3.py:328-335
       └─ LogitsProcessor.forward                            logits_processor.py:57
          └─ _get_logits: lm_head.quant_method.apply(hidden_states)   :111   # [num_reqs, vocab]
             ├─ _gather_logits: tensor_model_parallel_gather   :90-102     # TP 下聚到 rank0
             └─ logits[..., :org_vocab_size]                   :119-120    # 去掉 vocab padding
sampler_output = self._sample(logits, spec_decode_metadata)  :2118
    └─ self.sampler(logits=..., sampling_metadata=self.input_batch.sampling_metadata)  :1835
       └─ Sampler.forward                                    sampler.py:70-126
          ├─ logits.to(float32)                              :87
          ├─ apply_allowed_token_ids / apply_bad_words        :89-91
          ├─ apply_penalties（repetition/frequency/presence） :98
          ├─ sample()                                        :139-192
          │    ├─ all_greedy → greedy_sample = argmax        :136-137, :156-163
          │    └─ else → apply_temperature → topk_topp_sampler  :168-181
          └─ SamplerOutput(sampled_token_ids=[num_reqs, 1])  :119-125
_bookkeeping_sync(...)                                       :2120 → :1868
output = ModelRunnerOutput(...)                              :2150
```

### 5.1 `_bookkeeping_sync`：整条链唯一的 GPU→CPU 同步点

注释直接写了「NOTE: GPU -> CPU Sync happens here. Move as many CPU operations as possible before this sync point.」（1909-1910）。同步之前的 logprobs 转换（1911-1913）是最后一件能异步做的事；之后全部是 CPU 循环：

1. 丢弃 partial prefill 的采样结果并回退 RNG（1888-1901）；
2. `sampled_token_ids` → `list[list[int]]`（1922-1938）；
3. 把采样结果写回 `input_batch.token_ids_cpu` 与 `req_state.output_token_ids`（1980-1987）——**下一步的 `input_ids` 就来自这里**，形成闭环；
4. `req_ids` / `req_id_to_index` 做 **copy**（1905-1907），因为异步调度下这些结构下一步会被改。

### 5.2 `ModelRunnerOutput` 的字段就是 Scheduler 需要的全部

```python
ModelRunnerOutput(                      # outputs.py:85
    req_ids, req_id_to_index,           # batch 顺序快照（下一步可能已被 condense 打乱）
    sampled_token_ids=[[...]],          # 每请求每步的 token（投机解码时多个）
    logprobs, prompt_logprobs_dict,
    num_nans_in_logits, kv_connector_output,
)
```

`scheduler.update_from_output` 拿它判停（max_tokens / EOS / stop_token_ids）、追加 `output_token_ids`、释放请求（day2 §3.5）。**stop string 不在这里判**——那是前端 detokenize 之后的事。

### 5.3 实测印证（`results/day4_runner_test.json`）

| 项 | greedy (`temperature=0`) | `temperature=0.8, top_k=1` |
|---|---|---|
| 输出文本 | 完全一致 | 完全一致 |
| elapsed | 0.317 s | 0.302 s |
| completion_tokens | 24 | 24 |

`same_text=true` 说明 `all_greedy` 分支（`sampler.py:156-163` 直接 argmax 并 return）与 `top_k=1` 的随机分支（先 temperature 缩放再 top-k=1 采样）在数学上等价。混合 batch（52 / 187 / 82 prompt token，各 24 output）三个请求耗时 0.33 / 0.36 / 0.34 s，差距只有 TTFT 级的 prefill 差 —— 说明它们确实被**塞进同一个 step**（step 53 的 `[1, 60, 67]`），而不是串行。

---

## 6. 与此前 Triton Paged Decode、Week 16 Distributed ModelRunner 的差异

三个系统解决的是**同一问题的三个不同层次**：

| | Triton Paged Decode（gpu-kernel-performance-lab） | Week 16 Distributed ModelRunner | Day 4：vLLM V1 |
|---|---|---|---|
| 定位 | 单个融合 kernel 的**正确性原型** | 手写**分布式控制面 + 参考 runtime** | **生产引擎**的单步执行链 |
| 代码 | `triton_paged_decode.py`（155 行） | `week16_day4_continuous_batching.py` 等 | `gpu_model_runner.py` + `flash_attn.py` |
| 调度 | 无（调用方给定 batch） | Rank0 权威 Scheduler，广播 `StepPlan`，全 rank 投票 | `Scheduler` + 增量 `SchedulerOutput` |
| KV 寻址 | kernel 内按 `block_table[r][logical]` 逐页取 | 每 rank 自己的 block id / free list | CPU 侧预算 `slot_mapping` + `block_table` 两张表 |
| Attention | 手写 online-softmax，`one CTA per (req, head)` | PyTorch gather 物化页 → 普通 attention | `flash_attn_varlen_func`（paged，融合 KV 写入） |
| 位置编码/元数据 | 无（`context_lengths` 足够） | `StepPlan.start_positions` | `positions` / `query_start_loc` / `seq_lens` / `slot_mapping` 四件套 |
| 采样 | 不在范围（只算 attention） | 不在范围（decode token 是预定输入） | 引擎内 `Sampler` + `SamplingMetadata` |
| 并行 | 单卡 | TP=2（Gloo/NCCL），两个 AllReduce/层 | TP/PP/DP 可选（本实验 TP=1） |
| 失败模型 | 无 | fail-stop，vote 后崩溃不可恢复 | `EngineDeadError` + 引擎重启（day2 §2.6） |

### 6.1 三个具体跨度

1. **batch 的维度不一样**。Triton kernel 的 batch 维是 `B`（请求），每请求一个 program，`context_lengths` 圈定要扫多少页；vLLM 把 `B` 压平成 **token 维**，用 `query_start_loc` 当 `cu_seqlens`、`seq_lens` 当 `seqused_k`，于是「3 个请求、1 个 decode + 2 段 prefill」能在**一次 launch** 里跑完（step 53）。代价是位置、槽位、seq_len 三套数组必须严格互相对齐——这正是 §3 那些 `np.repeat` / `cumsum` 存在的理由。
2. **KV 写入的位置不一样**。Triton v0 明确声明「不写 KV，调用方先写」；vLLM 在 `FlashAttentionImpl.forward` 里先 `reshape_and_cache_flash`（498）再 `flash_attn_varlen_func`（531），两者共用 `slot_mapping`/`block_table`。Week 16 则是第三种：先把页 gather 成 dense K/V 再算，既不写页也不直读页。
3. **控制面的粒度不一样**。Week 16 每步广播**完整 StepPlan**并做 digest/vote 校验；vLLM 只发**增量**（`new_block_ids`、`num_computed_tokens`），把「已算到哪」缓存进常驻的 `InputBatch`，靠 `resumed_from_preemption` 这一个布尔区分「追加」还是「替换」（§2.1）。前者易于验证，后者省通信量但要求 worker 侧状态机严格可重放。

### 6.2 一句话结论

> Triton Paged Decode 定义的是**kernel 契约**（block_table + context_lengths → out），Week 16 定义的是**分布式一致性契约**（StepPlan + vote → 各 rank 一致提交），Day 4 这条链定义的是**batch 契约**（SchedulerOutput 增量 → InputBatch → 四套 index 数组 → 一次 varlen kernel → SamplerOutput）。
>
> 后者最难的部分不是数学，而是**索引一致性**：`positions`、`slot_mapping`、`query_start_loc`、`logits_indices` 四个数组描述的是同一批 token 的四个不同投影，任何一个偏移一位，输出不会报错、只会静默算错。

---

## 7. 一页结论

1. `Worker.execute_model` 只是 PP 的收发适配层；真正的执行在 `GPUModelRunner.execute_model`，六个阶段里五个有 `record_function_or_nullcontext` 标段（`"Preprocess"/"Forward"/"Postprocess"/"Sample"/"Bookkeep"`，2005/2061/2072/2117/2120），直接对应 profiler 里的五段。
2. `_update_states` 的职责是**把增量 `SchedulerOutput` 折叠进常驻 `InputBatch`**：删已完成、摘未调度、加新请求、按 `resumed_from_preemption` 决定块表 append/replace、`condense()` 压实、`refresh_metadata()` 重建采样元数据。它**不碰物理 KV 块**，也**不生成任何张量**。
3. `_prepare_inputs` 用 5 个索引数组把「按请求的调度信息」翻译成「按 token 的 kernel 参数」：`req_indices`（token→req）、`positions`（RoPE）、`input_ids`（token 内容）、`query_start_loc`（varlen 切分）、`slot_mapping`（token→物理槽）。`logits_indices = query_start_loc[1:] - 1` 是「每请求取最后一个 query」的唯一实现。
4. Qwen3 → FA 的链在 `__init__` 就定死了 backend（`cuda.py:362` 那行日志），运行期每层都是 `Attention.forward → torch.ops.vllm.unified_attention_with_output → FlashAttentionImpl.forward → reshape_and_cache_flash + flash_attn_varlen_func`，KV 写与 attention 读共用同一份 `slot_mapping`/`block_table`。
5. 输出链上只有一个同步点（`_bookkeeping_sync` 的 1909-1913），它之前的一切都留在 GPU 上；同步之后才把 token 写回 `token_ids_cpu`，供下一步的 `input_ids` 使用。

下一步可做（都有明确观测口径）：① 在 `trace_gpu_worker.py` 里补 `slot_mapping[:8]` 与 `logits_indices` 的采样输出，和本文 §3.4 的 `[0,1,61,128]` 对账；② 用 `enforce_eager=False` 跑一次，对比 `_preprocess` 的 padding 行为与 `num_input_tokens`；③ 在 `FlashAttentionImpl.forward` 上打点，量 `reshape_and_cache_flash` vs `flash_attn_varlen_func` 的时间占比（prefill / decode 各一组）。
