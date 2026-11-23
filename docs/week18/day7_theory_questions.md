# Week 18 Day 7 — 五道综合理论验收题（每题 10 分）

**要求**：按 AI Infra 技术面试标准作答，给出计算、源码/数据证据、替代解释、仍未知的条件。单纯列结论不满分。每题满分10，共50分。

## 第 1 题：SLO Capacity 的跨实验复核（10 分）

实验 A：P128/O128，vLLM V1，RTX 4090，KV auto，prefix cache on。80 req/s、240s 时全部符合 TTFT≤200ms 且 TPOT≤15ms；90 req/s 时约3.1%的请求 TPOT 违约。

实验 B：相同模型/GPU，KV override=4096、prefix cache off。单 seed、40s 下 90 req/s，3588/3588 最终成功且全部符合联合 SLO；105 req/s 25s 窗口 goodput 19.04 req/s，completed 95.08 req/s。

请回答：
1. 能否据此证明“生产稳定 SLO Capacity = 90 req/s”？为什么？
2. 解释窗口完成率、最终 arrival-cohort SLO 和含 Drain 批次吞吐的差异。
3. 设计最小可接受的容量认证方案（配置锁定、95/100点、多seed、长窗口、置信区间、恢复、余量），给出你会采用的上线决策规则。

## 第 2 题：KV Preemption 与冗余计算（10 分）

在关闭 Prefix Cache、无 Spec Decode、单 Engine 的实验中：`num_preemptions_total` 增加70；Scheduler Trace 记录 153,265 个被调度逻辑 Token 位置次数、107,845 个唯一逻辑位置、45,420 个重复位置次数；119个请求的名义 Prompt+Output 总量为107,964 Tokens。

请回答：
1. 重复调度率是多少？调度放大倍数是多少？
2. 为什么唯一位置 = 107,964 − 119？该等式成立有哪些前提？
3. 为什么不能使用 `prompt_tokens_total` 或 `preemptions * prompt_len` 估计真实冗余 GPU 时间？
4. 设计一种能避免 trace 重复、丢事件和跨异步 Step 误计的验证方法。

## 第 3 题：Nsight 因果归因与时间线陷阱（10 分）

某次真实 Serving Profiler 捕获 22.71s，报告 GPU Busy 16.24s（71.5%）、>1ms 的 Kernel Gap 合计4.805s；共推断1,619个 Step，`cudaEventSynchronize`累计5.78s，Attention/GEMM Kernel 活动份额约60.8%/30.9%。仅有1条非vLLM NVTX标记，CPU线程态未记录。

请回答：
1. 这些数据能证明“CPU Scheduler 是主瓶颈”吗？
2. 为什么同步 API 耗时不能从 GPU Idle 或 Step 时长中直接相加/相减？
3. 给出至少三个竞争性假设（Host/Launch、Attention KV 带宽、计算、请求饥饿等）与最小反事实实验。
4. 指出 `--cuda-graph-trace=node` 以及不完整采集窗口会怎样影响结论。

## 第 4 题：Admission Control 的双分母决策（10 分）

同一条25s过载 Arrival Trace：Offered=2,601。Open：全部送出，最终有441个该 Cohort 的请求满足联合 SLO，窗口内 Goodput=19.04 req/s；Gate：客户端主动拒绝634，接收1967，该 Cohort 最终1967全部达标，窗口内Goodput=77.52 req/s。混合负载60 req/s下 Open Cohort 100%达标，而 heuristic Cost Gate 额外拒绝472个（Long拒绝27.8%，Short拒绝8.5%）。

请回答：
1. 计算 Open/Gate 过载下 Offered Cohort 达标率、Gate 拒绝率及最终达标数净增长。
2. 为什么过载 Gate 有用，但混合负载 Cost Gate 是负收益？
3. 为什么客户端 local_reject 不等于服务端 HTTP 429？
4. 提出同时包含效率、可用性、公平性以及重试放大风险的业务决策标准。

## 第 5 题：Release Blocker 与技术面试叙事（10 分）

假设你负责对外发布“Qwen3-0.6B 单卡 RTX 4090 Serving 容量与性能诊断”开源项目。团队有四个候选结论：

A. “单卡 GPU 可稳定支持90 req/s，且满足生产 SLA。”
B. “在本实验配置、短窗过载点，Rate Gate 使完成窗口的 SLO Goodput 从19.04提高到77.52 req/s，另有24.4%的客户端主动拒绝。”
C. “70次Preemption对应45,420个重复**调度逻辑位置次数**，约占总调度次数29.6%。”
D. “Nsight显示GPU Busy 71.5%，因此Host Scheduler导致28.5%的GPU空转。”

请逐项判定：`可以原样发布` / `须加限定后发布` / `不能以该措辞发布`；提供修订后的精确表述与证据路径。最后给出一段约90秒的 AI Infra 面试口头介绍结构（问题→方法→证据→决策→局限）。
