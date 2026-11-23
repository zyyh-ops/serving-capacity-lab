## 第 1 题：A/B/C 调度

**1) 本轮 token 数**

- **A = 1**：`min(1, 128)`，即 decode。
- **B = 64**：`192−128=64`，`min(64, 127)=64`，刚好把提示词剩余部分预填完。
- **C = 63**：`min(80, 128−65)=63`，被 chunked prefill 切成第一块，剩 17 token 留到下一轮。
- 合计 128。另有两种分支：若 chunked prefill 关闭，C 需要 80 > 63，因此本轮会被整个跳过 → C=0；若本轮发生抢占，或 running 已满 `max_num_seqs`，则 waiting 段整体让位。B 放在 running 还是 waiting 队列，结果都是 64。

**2) SchedulerOutput 需要携带的关键信息**：`scheduled_new_reqs`（prompt ids、block_ids、num_computed、sampling params）、`scheduled_cached_reqs`（req_ids、`resumed_from_preemption`、`new_token_ids`、`new_block_ids`、`num_computed_tokens`）、`num_scheduled_tokens` 与总量、spec/encoder 调度表、`finished_req_ids`、grammar bitmask、connector metadata。原则是**只传增量**，因为 worker 端已持久缓存请求状态。

**3) 两个函数的职责**
- `_update_states()`对**CPU 侧状态**做校对：删除 finished 请求；把未调度（含被抢占）请求移出持久 batch 但保留 cached state；新建/恢复 `CachedRequestState`；更新 num_computed、追加或替换 block_ids；写入 spec token；最后 add/condense/refresh_metadata。它不参与计算。
- `_prepare_inputs()`把本轮 batch **展平成 GPU 张量**：`num_scheduled_tokens`、`req_indices`/`arange`、positions、gather input_ids、slot_mapping、`query_start_loc`/`seq_lens`、`logits_indices`、attn_metadata（含 block table）。

**4) 两个索引**
- `query_start_loc`：`cu = cumsum(num_scheduled_tokens)`，`[0]=0`，`[1:n+1]=cu`，尾部填 `cu[-1]` 保证非降（FA 要求）。
- `logits_indices = query_start_loc[1:] - 1`，每请求取**切片最后一个 token**；用 spec decode 时换成 `_calc_spec_decode_metadata` 的 indices（`repeat(cu−num_sampled)+arange`，拆 target/bonus，再 `hidden_states[logits_indices]` 进 sampler。

**5) 哪些采样有效**：判据是 `seq_len = num_computed + num_scheduled` 是否追平 `req_state.num_tokens`；`seq_len < num_tokens` 属 partial prefill，**该行采样结果会被清空并回退 generator**。本轮：A 有效；B 的 `128+64=192` 正好追平 → **有效**，产出第 1 个 output token；C 的 `63 < 80` → 采样被丢弃，本轮无输出（scheduler 侧不变式「不返回 partial prefill 输出」。sampler 本身不区分，每行都采，有效性由 runner + scheduler 过滤。

## 第 2 题：抢占与恢复
**1)** 抢占时 `num_computed_tokens` 160 → **0**，状态置 PREEMPTED 并插回 waiting **队首**；逻辑历史完全保留（prompt 160 + 20 output = 180 个 token 都在 `all_token_ids`/`output_token_ids`）；KV 引用全部 `free`，每块 `ref_cnt−1` 回 free queue。即「**逻辑进度不变，KV 归属清零**」。

**2)** `free_blocks()` 只减引用、入 free queue，**不 reset hash、不清 KV**；真正失效延迟到 `get_new_blocks()` → `_maybe_evict_cached_block()`。所以 `ref_cnt=0` 只代表「可驱逐」，之前仍可命中并被 `touch` 取回。

**3)** 恢复走 `get_computed_blocks` 查整块 hash 链，命中受三重限制：16-token 整块对齐、链式 hash（遇 miss 即停）、`num_tokens−1`（最后一个 token 必须算才有 logits）。于是：11 个满块仍在 → 命中 176，只重算 4；只剩 prompt 的 160 → 重算 20；全被 LRU 驱逐或 prefix cache 关闭 → 重算全部 180。

**4)** 重算的历史是**输入**不是输出：`seq_len<num_tokens` 时丢弃采样，追平后才采「下一个」token；scheduler 只 append runner 返回的 `new_token_ids`。用户只多等延迟，不会看到重复 token。

**5)** 指标：`vllm:num_preemptions`、preempt/resume 次数与 id、每次 resume 的重算 token 数、重算占 prefill token 比、preempt→resume 等待、TTFT/ITL 尾延迟、吞吐/goodput、KV churn。区分重算与正常计算靠**位置**：被 `resumed_from_preemption=True` 标记的请求，其 scheduled 区间里落在 `num_tokens_before` 之前的 token 即 recompute；正常新 prefill 是 `num_computed=0` 且无 output。

## 第 3 题：CUDA Graph 归因
**1)** kernel 数（40,468）和累计时长（199→192 ms）几乎不变，说明**没省计算，省的是 kernel 之间的 host 下发**：no_graph 相邻 kernel 间隔 p50 18.11 µs、GPU 忙占比 13.7%；piecewise 把 28 层切成 29 段图、每步 1 次 `cudaGraphLaunch`，间隔 0.06 µs、忙占比 31.5%，图内 kernel 占 79.1%。每 decode step 11 ms 里 kernel 只占 ~2 ms，其余是 launch 间隙。

**2)** attention 被固定切在图外：`splitting_ops` 默认就是 `_attention_ops`；它的 kernel 参数随 `seq_lens`/`block_table`/`num_splits`/`query_start_loc` 每步变化，backend 的 cudagraph 支持也常只允许逐段捕获（FULL 会被降级为 PIECEWISE）。

**3)** 不足够。kernel 时间之和看不到 gap、重叠与测量口径，只能排除「算得更快」。要归因到 host dispatch，还需要：gap 分布 18→0.06 µs、`cudaLaunchKernel` 调用数与总时长、CPU 线程忙、GPU 空闲比例下降，以及「prefill 无图 → TTFT 基本不变（27.4→24.1 ms）」这个对照。

**4)** Nsight 里看三件事：CUDA HW 行大段空白 + CUDA API 行连续 `cudaLaunchKernel`；算 `GPU idle = 1 − Σkernel/wall`；用 `nsys stats --report cuda_api_sum,cuda_gpu_kern_sum` 和 gap SQL，对比图模式每步只有 ≤29 次 `cudaGraphLaunch` 而步长降到 4.1 ms，并用 NVTX 的 Forward 区间确认间隙落在 launch 而非等数据。

**5)** 模型更大 → 每个 kernel 变长、每步计算变多，固定 host 开销占比下降，加速比**向 1 收敛**（Amdahl，上限 ≈ `1/(1−overhead_fraction)`）。不能外推：层数/MoE/TP 会改变每步 kernel 数与形状，batch 变大后 decode 会从 launch-bound 转成 compute/memory-bound，图捕获还受显存与静态形状限制。

## 第 4 题：吞吐与容量
**1)** C=32 时服务端 waiting 峰值 0、KV 峰值 4.2%、preemption 0，吞吐仍近线性（效率 82%），说明**从未积压**；32 只是 README 的 `max_num_seqs` 上限加客户端并发上限。闭环客户端还有背压，所以 5,796 是「本配置实测上限」，不是 GPU 饱和点。

**2)** continuous batching 下每步只做一次 forward 就把 N 条序列一起推进，步长由权重读取 + 固定 launch 开销决定（0.6B，几 ms），与 N 弱相关，所以吞吐≈线性、TPOT 只 +27%（3.93→5.01）。Little's law：λ=L/W，L 1→32、W 只 512.6→697.8 ms（+36%），λ 就涨 23×。

**3)** TTFT 是**每请求串行**的 prefill：要排 FCFS、等 running 有空位、还要和所有 decode 抢同一个 512 token 预算（chunked 后可能分多步）；TPOT 是整 batch 共享一次 step 的摊薄成本。实测 TTFT 13.0→58.9 ms（4.5×），TPOT 3.93→5.01（1.27×）。

**4)** 更严 SLO → 最佳 goodput 点**左移且变低**，先被突破的是 TTFT：TTFT≤50 ms 时 C=32（58.9）出局，≤30 ms 时 C=16（38.8）也出局 → 落到 8。若新 SLO 下无点达标，只能改配置（降 `max_num_seqs`、降 `max_num_batched_tokens`、增大 capture size）。

**5)** 用**开环恒速**扫 `--request-rate` 到过载（不设 max-concurrency），同时把客户端连接/并发开大，记录 offered vs achieved λ、拒绝/超时、waiting/running/KV、队列等待；用 `L=λW` 交叉校验；拐点定义成「吞吐平台 + 队列/TTFT 爆炸」，容量定义成满足 SLO 的最大 λ（goodput）。闭环客户端背压会把到达率自动压到服务能力，正是要避免的掩盖。

## 第 5 题：发布审核
**1)** **Blocker**：E（真实 key 被 Git 跟踪，硬阻塞，必须轮换并清历史）；D（「GPU 最大吞吐」夸大，属结论 blocker，改文可解）。**Known Limitations**：A（缺 Day6 启动日志 → 标证据等级 C，并在 README 给启动命令/配置）、C（缺 `.nsys-rep` → 标「仅由现有汇总支持」）。B 不是问题（hash 一致正是结论）。注意：若某条结论只能靠 A/C 支撑，则升级为 blocker。

**2)** 仅改文档：D、A、C。必须改仓库/补证据：E（删 key、rotate、`.gitignore`、secret scan、必要时 BFG 清历史）；A 若要「可复现」则补真实启动日志；C 若要「可复算」则补 raw `.nsys-rep`/SQL。

**3)** 以原始 JSON 为准复算（audit 已从 JSON 算出 2.573/2.323，统一指标定义（p50、单请求 vs 8 并发），改 README 并加自动一致性检查；不手改 JSON，发布前必须一致。

**4)** 把 commit SHA + tag 写进结果与 audit manifest（`sha256_manifest` 已有），在 CI 对同一 commit 重跑 audit + benchmark，commit message 引用证据 id，tag 与产物哈希一起归档；结果 JSON 记录 vLLM commit 与配置。

**5)** 最低标准：环境可复现（pin 版本 + 启动命令 + 配置）、证据与结论一致（JSON↔README、输出 hash）、无密钥/PII、范围与限制写清、正确性 + 自动 audit 通过、许可与归属（vLLM Apache-2.0）、README 一屏说明定位与边界。