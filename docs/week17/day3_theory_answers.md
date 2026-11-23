## 第 1 题

预算 32，先 running 后 waiting：

| 顺序 | 请求        | 本轮 tokens | 原因                                                |
| ---- | ----------- | ----------- | --------------------------------------------------- |
| 1    | A (RUNNING) | 20          | `min(20, 32)`                                       |
| 2    | B (RUNNING) | 1           | `min(1, 12)`                                        |
| 3    | C (WAITING) | 11          | chunked prefill：`min(30, 11)`；running=2<3，可准入 |
| 4    | D (WAITING) | 0           | 预算已 0；且 running=3=`max_num_seqs`               |

合计 32。C 剩 19 个 token 下一轮继续；A/B 已追平，下一步各 decode 1 token。

- **KV 不足的检测时机**：在 `schedule()` 里分配 slot 时检测，不是执行阶段。`allocate_slots()` 判断 `num_blocks_to_allocate > get_num_free_blocks()`。running 分配失败 → 按 FCFS 抢占队尾并重试；waiting 分配失败 → 本轮不准入（不会为它去抢 running）。
- **RUNNING 本轮未被调度 ≠ 回 WAITING**：running 循环按预算推进，预算耗尽后后面的请求只是本轮跳过，`status` 仍 RUNNING、`num_computed_tokens` 不变，下一轮继续。只有被选为抢占 victim 才置 `PREEMPTED` 并放回 waiting 队首。

## 第 2 题

1. 按 16 token 切块，`block_hash = hash(parent_hash, 块内 token_ids, extra_keys)`，extra_keys 含 LoRA、多模态、cache_salt；用 `find_longest_cache_hit` 在 `cached_block_hash_to_block` 中查。前缀链相同、整块 16 token 全等、块仍在 cache 才算命中。
2. `max_cache_hit_length = num_tokens - 1`：最后一个 token 必须重算才有 logits；又要求 `num_computed_tokens` 按 block 对齐，64 正好是整块边界时只命中 48，最后 16 个 token 仍要算。且命中只是不重写 KV，attention 仍要读这些块。
3. 靠 `ref_cnt`：命中/分配时 `touch` +1，释放只 -1；只有 `ref_cnt==0` 的块才进 free queue，可被分配或驱逐。满块只追加、不可变，共享安全，不需要 CoW。
4. `ref_cnt=0` 只代表**可驱逐**：`free_blocks()` 不删 hash、不清 KV，直到该块被 `get_new_blocks()` 取走并 `_maybe_evict_cached_block()` 才失效；在那之前仍可命中并被 `touch` 从 free queue 取回。

## 第 3 题

- **状态/块**：`status=PREEMPTED`，移出 running，`num_computed_tokens=0`，`kv_cache_manager.free()` 释放全部 block（ref_cnt-- 回 free queue；满块 hash 可能还留在 prefix cache），`waiting.prepend_request()` 插到队首，`output_token_ids` 保留。
- **要保留的历史**：prompt + 已生成 output 的 token ids（重算的输入）、sampling_params、max_tokens/stop、已生成计数、request_id/arrival_time 等 Request 元数据；要丢的只是 KV。
- **为什么要 recompute**：V1 0.10.2 只有 RECOMPUTE、没有 SWAP；block 释放后可能已被复用，且最后一个 token、最后一个不满块本来就不在 cache。resume 时从 0 重新 prefill 整段历史。
- **为什么不会重发历史**：历史是输入不是输出；partial prefill 的采样结果会被丢弃，只有追平后才采出“下一个”token，scheduler 也只下发 `new_token_ids`。客户端只多等延迟。
- **Thrashing 判断**：同一请求反复 preempt→resume→再 preempt，每次 resume 重算 token 数≈历史长度，free block 长期贴 0，吞吐低于不抢占时。重算量远大于新增生成量就是 thrashing。

## 第 4 题

- **不保证的原因**：统一调度只取消阶段屏障，不提供配额/公平。running 先扫且每个至少 1 token，预算被吃光后 waiting 的 `token_budget > 0` 不成立；`len(running)==max_num_seqs` 直接 break；本轮有抢占时 waiting 整段跳过（scheduler.py:333）；FCFS 无 aging，队尾可持续等待；KV 抢占只由 running 自身分配失败触发，不会为准入 waiting 而抢占。
- **缓解机制与影响**：
  1. **预留 prefill 配额**（每步留 R 个 token，或每 k 步强制准入 1 个）：TTFT↓、TPOT↑、吞吐略↓。
  2. **Aging/最大等待时间**（等越久优先级越高，超时强制准入或抢占）：尾 TTFT↓、公平性↑、TPOT 抖动↑、吞吐↓（重算代价）。
  3. **限制 running 并发**（低于 `max_num_seqs`，留 admission headroom）：TTFT↓、TPOT↑、吞吐↓。
- **源码已有**：FCFS/Priority 队列与 `--scheduling-policy`、`max_num_seqs`、chunked prefill 默认开、KV 不足抢占并 prepend。**我提出的**：预留配额、aging、低于上限的 running 数、时间片轮转。vLLM 的 priority 是静态优先级，没有 aging。

## 第 5 题

1. free block 只在 16-token 块粒度上变化；这 32 个 token 落在 running 请求已分配块的空槽里（如 96 token 占 7 块=112 槽，有 16 槽富余），没跨块边界；常见情形就是 32 个 running 各 decode 1 token。也可能本轮 free+alloc 相抵。
2. 不能。块分配有粒度、有 lookahead/预分配，还会被 preempt/free、prefix-cache touch/evict 改变，与实际写入量无固定关系。实际写入看 `sum(num_scheduled_tokens)` 或 `slot_mapping` 长度。
3. 开 prefix caching 后 free queue = `ref_cnt==0` 的 LRU 可驱逐链表，块里可能仍有有效 KV 且在 hash 索引中；“free”= 可分配/可驱逐，不是空。命中 `touch` 会把块从队列摘走（free 数下降但没有任何写入）。
4. `schedule()` 是 CPU 纯逻辑；kernel 异步下发，Python 返回时 GPU 可能还没算完；CUDA graph 会合并 kernel；trace 还含 Python/msgpack/ZMQ/日志开销。要看 CUDA event / torch.profiler / nsys 的 kernel 时间。
5. 还需要：每步 preempted/resumed 的请求数与 id、resume 时重算 token 数、preempt→resume 等待时长、同一请求被抢占次数、`vllm:num_preemptions`、KV block alloc/free churn、受害请求的 TTFT/TPOT 尾延迟，以及同负载“有/无抢占”的吞吐 + GPU 时间对比。
