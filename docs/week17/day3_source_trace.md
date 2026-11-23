# Day 3：Scheduler / KVCacheManager / BlockPool 源码追踪（vLLM 0.10.2）

---

## 0. 速查：五个对象各管什么

| 对象 | 位置 | 唯一职责 | 状态 |
|---|---|---|---|
| `Scheduler` | `sched/scheduler.py:41` | 每个 step 决定「谁、算几个 token」 | `waiting` / `running` 两个队列 + `token_budget` |
| `KVCacheManager` | `core/kv_cache_manager.py:84` | 对外门面：查缓存命中、判断能否分配、提交缓存 | 无自有状态（转发给 coordinator） |
| `KVCacheCoordinator` | `kv_cache_coordinator.py:15` | 按 KV cache group 扇出到各单类型 manager | 持有 `block_pool` |
| `SingleTypeKVCacheManager` | `core/single_type_kv_cache_manager.py:17` | `req_id → 该请求的块列表` 映射 | `req_to_blocks`、`num_cached_block` |
| `BlockPool` | `block_pool.py:22` | 全局块的收/发/缓存，**唯一**的物理块来源 | `free_block_queue` + `cached_block_hash_to_block` + 每块的 `ref_cnt` |

一句话：**Scheduler 决定算什么，KVCacheManager 决定能不能算（块够不够），BlockPool 是块账本。**

---

## 1. `Scheduler.schedule()` 的 RUNNING / WAITING 两条路径

`schedule()`（`scheduler.py:177`）结构是**固定两段**，顺序即优先级：

```
① RUNNING 循环  scheduler.py:208   while req_index < len(running) and token_budget > 0
       └─ 每个 running 请求：算 num_new_tokens → allocate_slots → 失败则【抢占】
② WAITING 循环  scheduler.py:333   if not preempted_reqs: while waiting and token_budget > 0
       └─ 每个 waiting 请求：get_computed_blocks → allocate_slots → 失败则【break】
③ 收尾        scheduler.py:546~600 组装 SchedulerOutput → _update_after_schedule() 推进 num_computed_tokens
```

### 1.1 RUNNING 路径（`scheduler.py:208~318`）

```python
num_new_tokens = (request.num_tokens_with_spec        # 已生成 + spec draft
                  + request.num_output_placeholders   # 异步调度占位
                  - request.num_computed_tokens)      # 已经算过的
if 0 < long_prefill_token_threshold < num_new_tokens:  # 长 prompt 限速（防一步吃光预算）
    num_new_tokens = long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)      # ① 全局 token 预算
num_new_tokens = min(num_new_tokens,
                     self.max_model_len - 1 - request.num_computed_tokens)  # ② 位置上限
```

三个细节最容易被忽略：

1. **`num_new_tokens == 0` 时是 `continue` 不是 `break`**（`scheduler.py:236-250`）。注释写得很直白：这样就不严格遵守 FCFS，后面的请求还能被调度。
2. **RUNNING 路径不查前缀缓存**。已经在跑的请求，`num_computed_tokens > 0`，它的块已经通过 `req_to_blocks` 持有；再查 hash 既无意义也会错误地产生「新命中块」。
3. RUNNING 的 `allocate_slots(request, num_new_tokens, num_lookahead_tokens=…)`（`scheduler.py:253`）**只有 2 个有效参数**，`num_new_computed_tokens` 用默认 0 —— 这是它与 WAITING 路径最大的区别。

**失败即抢占**（`scheduler.py:252~290`）：

```python
while True:
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is None:                      # 块不够
        preempted_req = (max(running, key=lambda r: (r.priority, r.arrival_time))
                         if priority else self.running.pop())   # FCFS：踢队尾
        running.remove(preempted_req)
        kv_cache_manager.free(preempted_req)     # 块归还池子
        preempted_req.status = PREEMPTED
        preempted_req.num_computed_tokens = 0    # ← 进度清零（详见 §4）
        self.waiting.prepend_request(preempted_req)   # 插到 waiting 队头
        if preempted_req is request:             # 只剩自己可踢 → 本 step 全体放弃
            can_schedule = False; break
    else:
        break
if not can_schedule: break      # 直接跳出 RUNNING 循环
```

### 1.2 WAITING 路径（`scheduler.py:333~540`）

```python
if not preempted_reqs:                              # ① 本 step 抢过 → 一个都不收
    while waiting and token_budget > 0:
        if len(running) == max_num_running_reqs: break          # ② 并发上限
        request = waiting.peek_request()                        # 只窥视，不摘
        ...
        if request.num_computed_tokens == 0:
            new_computed_blocks, n_hit = kv_cache_manager.get_computed_blocks(request)  # ③ 查缓存
            num_computed_tokens = n_hit + num_external_computed_tokens  #    + KV connector
        else:   # KVTransfer 场景：异步 recv 后 num_computed_tokens 已 > 0
            new_computed_blocks = kv_cache_manager.create_empty_block_list()
            n_hit = 0
            num_computed_tokens = request.num_computed_tokens

        num_new_tokens = request.num_tokens - num_computed_tokens   # ④ 要真正算的
        if not chunked_prefill_enabled and num_new_tokens > token_budget:
            pop + prepend 到 skipped; continue                      # 不切开就不调度（保 FCFS）
        num_new_tokens = min(num_new_tokens, token_budget)

        new_blocks = kv_cache_manager.allocate_slots(
            request,
            num_new_tokens + num_external_computed_tokens,   # 含要为远端 KV 预留的槽
            n_hit, new_computed_blocks,
            num_lookahead_tokens=effective_lookahead_tokens, # num_computed==0 时恒为 0
            delay_cache_blocks=load_kv_async, ...)
        if new_blocks is None: break        # ⑤ 注意：不抢占，直接把剩余预算让给别人
```

三个关键设计点：

1. **`peek` 而不 `pop`**（`scheduler.py:338` vs `501`）。分配失败时请求原封不动留在队头，所以 `schedule()` 里任何 `break`/`continue` 都不会丢请求；`skipped_waiting_requests` 只是临时收容「这一步跳过但状态正常」的请求，最后 `prepend_requests` 放回队头（`scheduler.py:543`）。
2. **`if not preempted_reqs`**（`scheduler.py:333`）是本文件里最反直觉的一行：一旦本 step 发生过抢占，WAITING 段整段跳过。原因是被抢占的请求已经插到 waiting 队头，此时再收新请求会（a）违反 FCFS，（b）立刻把刚释放的块重新抽干。
3. **状态决定去向**（`scheduler.py:514-520`）：`WAITING → scheduled_new_reqs`，`PREEMPTED → scheduled_resumed_reqs`，两者在 `SchedulerOutput` 里走不同字段，并在 `_make_cached_request_data` 里标 `resumed_from_preemption=True`（`scheduler.py:699`）。这个布尔量是**真正决定块表语义**的开关（`gpu_model_runner.py:606-616`）：`False` → 把 `new_block_ids` **追加**到已有块表；`True` → 用完整块表**整体替换** `req_state.block_ids`。这也解释了两条路径在 `req_to_new_blocks` 上的不对称：RUNNING 存的是 `new_blocks`（增量），WAITING 存的是 `get_blocks()`（全量，`scheduler.py:524-525`）。

### 1.3 两条路径对照

| 维度 | RUNNING | WAITING |
|---|---|---|
| 循环条件 | `token_budget > 0`（无并发上限检查） | `token_budget > 0` **且** `len(running) < max_num_seqs` |
| 是否查前缀缓存 | ❌ | ✅ `get_computed_blocks`（仅当 `num_computed_tokens == 0`） |
| `num_new_tokens` 口径 | `num_tokens_with_spec + placeholders - num_computed` | `num_tokens - (本地命中 + 远端命中)` |
| `allocate_slots` 参数 | 2 个（新 token + lookahead） | 5 个（新 token + 外部 token、命中数、命中块、lookahead） |
| 分配失败后果 | **抢占**其他 running 请求，重试 | **break**，本 step 不再收新请求 |
| 成功后状态 | 留在 `running` | `running.append()`，`status = RUNNING` |
| 计入 output | `scheduled_cached_reqs`（只带增量块 id） | `scheduled_new_reqs`（带完整块表） |

### 1.4 一个 step 的收尾：`_update_after_schedule()`（`scheduler.py:630`）

`schedule()` 返回前**不会**推进 `num_computed_tokens`，而是统一在 `_update_after_schedule()` 里加：

```python
for req_id, n in scheduler_output.num_scheduled_tokens.items():
    self.requests[req_id].num_computed_tokens += n
```

为什么要拆成两步？源码注释给了三条理由，核心是**同一份 `SchedulerOutput` 必须携带「调度前」的 `num_computed_tokens`**，worker 靠它算 `input_positions` 和 `input_ids` 的切片；如果提前推进，本 step 的 token 会被重复算一次。同时这一步清空 `finished_req_ids`，让下一个 step 的 output 只报告「上一步之间新结束」的请求。

---

## 2. `get_computed_blocks()` 与 `allocate_slots()` 的关系

一句话概括：**`get_computed_blocks()` 是只读查表，`allocate_slots()` 是唯一的写账动作（预付 + 落账）。**

```
Scheduler.schedule()  WAITING 段
      │
      ├─① KVCacheManager.get_computed_blocks(request)        kv_cache_manager.py:153
      │     ├─ 前缀缓存关闭 / 需要 prompt_logprobs → 返回 ([], 0)      :168
      │     ├─ max_cache_hit_length = request.num_tokens - 1           :179
      │     ├─ coordinator.find_longest_cache_hit(block_hashes, …)     :181
      │     │     └─ FullAttentionManager: 沿 hash 链逐块 get_cached_block，遇 miss 即停
      │     │        （single_type_kv_cache_manager.py:272-281）
      │     └─ 返回 KVCacheBlocks(命中块) + 命中 token 数
      │     ★ 全程不改任何 ref_cnt / 不动 free queue
      │
      └─② KVCacheManager.allocate_slots(request, num_new, n_hit, hit_blocks)  kv_cache_manager.py:192
            ├─ remove_skipped_blocks(num_computed_tokens)            :252（full attn 是 no-op）
            ├─ num_tokens_need_slot = min(num_computed + num_new + lookahead, max_model_len)  :259
            ├─ num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)  :263
            ├─ if num_blocks_to_allocate > block_pool.get_num_free_blocks(): return None  :270 ★唯一的失败出口
            ├─ block_pool.touch(命中块)        → ref_cnt += 1，ref_cnt==0 的从 free queue 摘除  :276
            ├─ coordinator.save_new_computed_blocks(命中块) → 写入 req_to_blocks  :284
            ├─ coordinator.allocate_new_blocks(...) → block_pool.get_new_blocks() 取新块  :287
            └─ coordinator.cache_blocks(request, min(computed+new, num_tokens))  → 写 hash  :301
```

### 2.1 记账公式（读懂这三行就懂了两者的分工）

```python
# allocate_slots 内部
num_computed_tokens   = request.num_computed_tokens + num_new_computed_tokens   # 命中数「加回来」
num_tokens_need_slot  = min(num_computed_tokens + num_new_tokens + lookahead, max_model_len)

# SingleTypeKVCacheManager.get_num_blocks_to_allocate（single_type_kv_cache_manager.py:76-86）
num_required_blocks = cdiv(num_tokens_need_slot, block_size)
num_new_blocks      = num_required_blocks - len(命中块) - len(req_to_blocks[req_id])
return num_new_blocks + num_evictable_computed_blocks      # 每个 ref_cnt==0 的命中块再 +1
```

**方向相反的两处加法是本节唯一的坑**：`allocate_slots(num_new_tokens=…)` 的 `num_new_tokens` **不含**命中块（命中块不用算），但内部算「需要多少槽」时又把 `n_hit` 加回去。所以：

- `num_new_tokens` = 这一步要交给 GPU 算的 token 数（= `SchedulerOutput.num_scheduled_tokens`）；
- `num_tokens_need_slot` = 这块请求的总长度（已算是命中 + 新算 + 投机预分配），决定块表长度。

### 2.2 为什么顺序不能反，以及 4 个易错点

1. **必须先查后分配**。`allocate_slots` 需要 `new_computed_blocks` 才能正确算差量：命中块会被 `touch` 并直接 `save_new_computed_blocks` 写进 `req_to_blocks`，如果先分配再查，同一批块会被算两遍。
2. **命中块 ref_cnt==0 时要额外 +1 个块**（`single_type_kv_cache_manager.py:83-86`）。命中块可能正躺在 free queue 里（缓存块 = 可驱逐块），紧接着的 `block_pool.touch()` 会把它从 free queue 摘出并 `ref_cnt += 1`（`block_pool.py:244-250`），这个「把可驱逐块变成在用块」的动作要占一个配额，所以 `get_num_blocks_to_allocate` 里用 `num_evictable_computed_blocks` 补上 —— 否则会出现「算出来够、真去分配时 `popleft_n` 断言失败」。
3. **`allocate_slots` 失败时几乎无副作用**（`kv_cache_manager.py:270` 是唯一出口，且在任何写操作之前），所以 RUNNING 路径可以放心 `while True` 重试。唯一已执行的 `remove_skipped_blocks` 在 full attention 下是 `pass`。
4. **`max_cache_hit_length = num_tokens - 1`**（`kv_cache_manager.py:179`）。即使 prompt 全部命中，也得重算最后 1 个 token 才能拿到 logits；而 `allocate_slots` 要求 `num_computed_tokens` 按块对齐，所以实际会多算「一个块」而不是一个 token —— 源码注释明确说这是已知的性能损失。

**实测量化（prefix 场景，`results/day3_scheduler_only.log`）**

| step | 事件 | free blocks | 说明 |
|---|---|---|---|
| 1 | 冷请求入场，`num_scheduled_tokens=106` | 63 → 56 | 106 token / block_size 16 = `cdiv` = **7 块** |
| 2 | decode | 56 → 56 | 块已够，不再分配 |
| 26 | 同 prompt 再来，`prefix_hit_tokens=96` | 63 → 56 | 命中 96 token（**6 块**）→ 只需算 `106-96=10` 个 token；但 free 仍掉 7 = **6 个命中块被 touch 摘出队列 + 1 个新块** |

冷 0.276s vs 暖 0.120s（`results/day3_prefix.json`），2.3× —— 省掉的是 96 个 token 的 prefill 计算，而不是块分配。

---

## 3. BlockPool：Free Queue / Hash / Ref Count

`BlockPool`（`block_pool.py:22`）只有三个数据结构，但它们互相耦合得很紧：

```python
self.blocks: list[KVCacheBlock]                    # 全部物理块，idx == block_id
self.free_block_queue = FreeKVCacheBlockQueue(blocks)   # 双向链表，队头 = 最先被驱逐
self.cached_block_hash_to_block: dict[
    BlockHashWithGroupId, dict[int, KVCacheBlock]]      # {hash: {block_id: block}}，允许同 hash 多块
```

### 3.1 Free Queue：为什么是手写双向链表，而不是 deque

`FreeKVCacheBlockQueue`（`kv_cache_utils.py:205`）用 `prev_free_block` / `next_free_block` 串起来，配一对 `fake_free_list_head/tail` 哑节点（`kv_cache_utils.py:243`）以消除边界分支。**不用 `collections.deque` 的唯一原因**：`touch()` 需要一个 O(1) 的「从队列中间摘走某一块」（`remove()`，`kv_cache_utils.py:323`），而 deque 按值删除是 O(n)。类注释还强调「不分配任何 Python 对象」来逼近 C++ deque 的性能。

顺序语义（`kv_cache_utils.py:214-221`）：

1. **队头 = 最先被驱逐**（LRU）；
2. 同一序列释放的块里，**哈希链更长的尾块更靠前**；
3. 实现方式：释放时**逆序**入队 —— `SingleTypeKVCacheManager.free()` 里 `ordered_blocks = reversed(req_blocks)`（`single_type_kv_cache_manager.py:168`），`BlockPool.free_blocks()` 再顺序 `append_n` 到队尾（`block_pool.py:264`）。

接口只有四个原语：`popleft_n(n)`（分配）、`append_n`（释放）、`remove`（touch 时摘除）、`num_free_blocks`（`get_num_free_blocks()` 直接读它）。

### 3.2 Hash：块内容的身份，也是前缀缓存的唯一索引

`KVCacheBlock.block_hash`（`kv_cache_utils.py:165-185`）只在块**写满**时才被赋值，且是 `(block_hash, group_id)` 二元组（`make_block_hash_with_group_id`），支持混合注意力的多 group 场景。

- 写入发生在 `cache_full_blocks()`（`block_pool.py:101`）：`num_cached_blocks → num_full_blocks` 之间新满的块逐个赋 hash 并登记到 `cached_block_hash_to_block`。
- **不去重**：注释（`block_pool.py:58-62`）说明同一份内容可能出现多个块，`get_cached_block()` 用 `next(iter(...values()))` 取第一个（`block_pool.py:97`）。理由是为了让已分配块的 id 不变、块表保持 append-only。
- 驱逐即失效：`_maybe_evict_cached_block()`（`block_pool.py:199`）在**分配**时才检查 —— 从 free queue 拿到的块若还带 hash，先 `reset_hash()` + 从 map 中删除，再给新内容用。所以「缓存命中 → 块被复用」和「块被驱逐」是同一个动作的两面。

### 3.3 Ref Count：三种状态，两个不变式

| `ref_cnt` | 是否在 free queue | 含义 | 谁负责推进 |
|---|---|---|---|
| `0` | ✅ 在 | 空闲；**若带 hash 就是可驱逐的缓存块** | `free_blocks` 释放 / 初始状态 |
| `> 0` | ❌ 不在 | 被 n 个请求（或 1 个请求的命中链）持有 | `get_new_blocks`(=1) / `touch`(+1) |
| 任意 | ❌ | `is_null=True` 的 null block | 永不释放，`ref_cnt` 不维护 |

**不变式 1**：块在 free queue ⟺ `ref_cnt == 0` 且非 null。`free_blocks()` 先全体 `ref_cnt -= 1`，再只把归零的块入队（`block_pool.py:262-267`）—— 所以「多请求共享一个前缀块」时，只有最后一个使用者释放才真正入队。
**不变式 2**：`block_hash is not None ⇒ 块内容等于该 hash 对应的 token 序列`。任何改写块内容的路径（分配复用）必须先 `reset_hash`。

`null_block`（`block_pool.py:69`）是 `block_id=0` 的哨兵，`__init__` 里直接 `popleft` 掉、`is_null=True`，用来在滑动窗口/跨注意力的块表里占位。因为它在池外，所以 `get_usage()` 要用 `num_gpu_blocks - 1` 做分母（`block_pool.py:315`）。

### 3.4 三者协作的两个具体落点

1. **计算可分配量**（§2.1 的 `num_evictable_computed_blocks`）：命中的块 `ref_cnt==0` 意味着它「又像缓存又像空闲块」。Hash 负责找到它，Free queue 负责交出它，Ref count 负责判断要不要为它多算一个配额 —— 三者缺一不可。
2. **级联注意力优化**（`single_type_kv_cache_manager.py:292-301`）：统计「所有 running 请求共享的公共前缀块数」竟是靠 `block.ref_cnt == num_running_requests` 逐块数出来的。这是 ref count 作为「共享度」直接参与调度决策的例子，产物是 `SchedulerOutput.num_common_prefix_blocks`（`scheduler.py:559-565`）。

### 3.5 实测印证（pressure 场景）

step 68：`free_blocks_before=0`，`free_blocks_after=7` —— 抢占释放 7 块（被抢请求 112 token / 16 = 7 块），且**释放后请求立刻继续跑**（`scheduled_tokens` 里其余 7 个请求各 1 个 token）。这说明 free queue 的释放是「同步、立即生效」的，也是抢占能在同一个 step 内救急的原因。

---

## 4. Preemption 如何影响请求进度

### 4.1 触发与动作（只有一处，内联在 RUNNING 循环里）

> 纠正 day2 文档里的说法：0.10.2 V1 引擎**没有** `_preempt_request()` 方法，也**没有** `PreemptionMode.SWAP` / `RequestStatus.SWAPPED`（`request.py:201-213` 的枚举里已无 SWAPPED）。V1 的抢占只有「重算」一种形态，代码就是 `scheduler.py:252-290` 的那个 `while True`。

六个动作，按顺序（`scheduler.py:260-284`）：

| # | 动作 | 对「进度」的影响 |
|---|---|---|
| 1 | 选受害者：FCFS `running.pop()`（**最后进入的**）；PRIORITY `max((priority, arrival_time))` | 保住先到者的进度 |
| 2 | `running.remove(victim)` | 本 step 不再为它算 token |
| 3 | `kv_cache_manager.free(victim)` → 块 `ref_cnt-1`，归零的入 free queue **但保留 hash** | 请求失去自己的 KV；块内容能否存活取决于后续是否被复用 |
| 4 | `victim.status = PREEMPTED` | 状态机切到「待恢复」 |
| 5 | **`victim.num_computed_tokens = 0`** | **计算进度清零** |
| 6 | `waiting.prepend_request(victim)`（→ `appendleft`） | 排到队头，下一个空闲 slot 就是它 |

### 4.2 三种「进度」被区别对待 —— 这是理解抢占的核心

| 状态量 | 抢占后 | 后果 |
|---|---|---|
| `request.num_tokens`（prompt + 已生成 token） | **保留** | 语义进程不丢：不会重复生成、不会改输出 |
| `num_computed_tokens` | **归零** | 下一次被调度时 `num_new_tokens = num_tokens - 0`，等于**全量重 prefill**（含已生成的输出 token） |
| KV 块 | 归还池子 | 若被别的请求复用 → `reset_hash`，缓存彻底失效 |

所以 V1 抢占的本质是：**用「重算」换「不 OOM」**。它保证输出正确性与 FCFS 公平性，但代价是本该已完成的算力被丢掉。

### 4.3 四个连锁效应

1. **本 step 不再收新请求**（`scheduler.py:333` 的 `if not preempted_reqs`）。抢占发生的那一步，WAITING 段整段跳过 —— 因为被抢的请求已插到队头，新请求进来就是插队，并且会立刻再抽干刚释放的块。
2. **抢占可能级联**。释放的块不够时 `while True` 会继续抢下一个受害者；一旦抢到自己（`preempted_req is request`），本 step 直接 `break` 出 RUNNING 循环，剩下的 running 请求也全部放弃 —— 即「整批让路」。
3. **恢复走 WAITING 路径 ⇒ 有前缀缓存兜底的机会**。归零后 `num_computed_tokens == 0`，恢复时必然重新 `get_computed_blocks()`。如果它自己的块还没被驱逐，就能命中自己的前缀，把重算量压到最低；如果已被驱逐，就是纯重算。**这条路径把抢占和前缀缓存绑在了一起**，也是「抢占不至于灾难」的唯一机制。
4. **thrash / 活锁风险**。恢复的请求会被 `running.append()` 到队尾，于是它立刻又成为 FCFS 的**首选受害者**。当工作集 > 池容量时，请求会在「恢复 → 被抢 → 恢复」之间循环，算力全花在重算上。

### 4.4 实测证据（`results/day3_server_trace.log`，8 并发 × 102 prompt + 112 输出，ctx 上限 256）

8 个请求满长后各需 `cdiv(214,16)=14` 块，共 112 块 > 可用 63 块 ⇒ **数学上必然抢占**。

```
step 61~66   running=8, free: 7→6→5→4→3→2   8 个请求各算 1 token，块池被逐格吃光
step 67      running=8, free: 2→0           ← 池空
step 68      free_before=0, free_after=7
             preempted=["d3ab98f5f2-0"]     ← 队尾（最后入场）被抢，释放 7 块
             running 7, waiting 1           其余 7 个请求照常各算 1 token
step 94/113/145 再抢 b10de6bdcd / c14f736b2b / 6dc2cbd724（全部是队尾）
...
step 163     两个被抢请求同一步恢复：6dc2cbd724-0 算 80 token、c14f736b2b-0 算 45 token
             （128 预算被切成两段重算）;  waiting 4→2
step 164     c14f736b2b-0 接着算 114 token；b10de6bdcd-0 恢复，算 11 token
step 165     b10de6bdcd-0 接着算 125 token
step 166     d3ab98f5f2-0 最后恢复：一次算 113 token = 102 prompt + 11 已生成 token，命中 0
             free 31→22（113 token = 8 块，另 1 块是 b10de6bdcd 解码跨块）
step 267     running=0, free=63            全部结束
```

关键读数：**step 68 被抢的 `d3ab98f5f2-0`，直到 step 166 才恢复 —— 空转 98 个 step；恢复时一口气重算 113 个 token，其中 11 个是它已经生成过、纯属白算的输出 token。** 它也是全场最后一个结束的请求，尽管它的 prompt 与别人等长。注意恢复时的「命中 0」：它在 step 68 释放的 7 块全部被其他请求复用（`reset_hash` 驱逐），所以前缀缓存这次没能兜底。

同一 step 的 token 预算账也印证了「重算挤占吞吐」：step 163 的 `total_scheduled_tokens=128` 里，80 个 token 是 `6dc2cbd724-0` 的重算，45 个是 `c14f736b2b-0` 的重算，只有 3 个 token 是真正的 decode 进展。

### 4.5 观测缺口（留给下一轮）

`trace_scheduler.py:49-53` 只统计 `output.scheduled_new_reqs` 的 `num_computed_tokens` 作为前缀命中线索，而**被抢占后恢复的请求走的是 `scheduled_cached_reqs`（`resumed_from_preemption=True`）**，所以在日志里完全看不见 —— 本轮 4 次抢占的「恢复动作」只能靠我事后从 `scheduled_tokens` 的大数值反推。建议在 `TraceScheduler.schedule()` 里补一段：

```python
resumed = [short_id(r) for r, f in zip(output.scheduled_cached_reqs.req_ids,
                                        output.scheduled_cached_reqs.resumed_from_preemption) if f]
# 连同该 req 的 num_scheduled_tokens 一起记，就能直接量出「每次抢占的重算成本」
```

---

## 5. 一页结论

1. `schedule()` 是**两段式**：RUNNING 先吃预算、失败就抢别人；WAITING 后吃剩下的、失败就 `break`。抢占只发生在 RUNNING 段，而且抢过之后这一步就再也不收新请求。
2. `get_computed_blocks()` **只读**（hash 查表，不动 ref_cnt），`allocate_slots()` 是**唯一写账口**（touch 命中块 → 记入块表 → 补新块 → 写回 hash）。二者之间靠 `n_hit` 这个数传递「有多少 token 不用算」。
3. `BlockPool` 三件套的分工是一个闭环：**Hash 定义「这块是谁」→ Ref Count 定义「这块有人用吗」→ Free Queue 定义「这块什么时候被覆盖」**。任何"缓存命中"都是这三者的联合判断结果。
4. 抢占把 `num_computed_tokens` 归零、把块还池、把请求插回队头 —— 输出 token 不丢，但 prefill 算力全部重付；前缀缓存若还在，能把损失压到最小，若已被驱逐，就是本实验里那次 **98 步空转 + 113 token 重算**。

下一步可做（任选其一，都有明确观测口径）：① 把 `--num-gpu-blocks-override` 调小复现更剧烈的 thrash，量 `被抢次数 vs 有效 token 数`；② 在 `TraceScheduler` 里补 `resumed_from_preemption` 统计，把「重算 token 占比」做成一个指标；③ 对比 `--enable-prefix-caching` 开关下同一 pressure 负载的抢占恢复成本，直接量化前缀缓存对抢占的兜底价值。
