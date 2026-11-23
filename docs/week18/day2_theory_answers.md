## 第 1 题：容量与观察窗口
- 注入总量 80×60 = **4,800** 条；窗口内完成 3,900 → **窗口吞吐 65 req/s**（分母是 60s，只统计窗口内完成的请求，尾部被截断）。
- **最终成功比例** (3,900+900)/4,800 = **100%**；无失败只说明 vLLM 用排队吸收过载，不靠报错。
- **含 Drain 的批次吞吐** 4,800/(60+35) = **50.5 req/s**。
- 三个数不能互替：65 是窗口内的离开率（下界）；100% 是结果统计方式；50.5 把 35s 排空摊进分母，是长程下界而非稳态。
- **不能判定饱和**：尾部 900 条在 35s 内排空（≈26/s，远低于窗口内 65/s），队列最终清空，属于暂态积压。要判饱和得看：队列是否单调上升不回落、末尾窗口吞吐是否走平、把 offered 提到 90/100 后 completed 是否封顶而 backlog 继续涨、TTFT/TPOT 是否持续恶化。

## 第 2 题：排队增长与稳定性
- **B 组更可能接近过载**：A 是 2s 内的突发/冷启动，峰值高但立刻回落，是瞬态；B 在 60s 内从 2 单调爬到 12，符合 λ>μ 的持续积压（Little's law 下队列斜率≈λ−μ）。
- 只看 Waiting Peak 无意义，需联合：到达率 vs 完成率时间序列、Waiting/TTFT 分位数趋势、KV Cache 与 prefill 队列、停止注入后的回落时间、多 seed/长窗口重复。
- **排除客户端瓶颈**：查 `max_concurrency`/信号量、连接池上限、客户端 CPU/GC、单进程事件循环；用多客户端/多进程、放宽并发上限复测；把客户端 dispatch 时间戳与服务端 `request_success`、`num_requests_waiting` 对比，确认客户端没有先行限流。

## 第 3 题：SLO Goodput 容量
- 完成率封顶在 ~77 req/s，但 goodput 在 60 之后开始大幅下降（57→45→20）。可报告的 **SLO 容量 ≈ 57–60 req/s**，即 goodput 拐点（仍满足 TTFT/TPOT SLO、队列稳定）的最大 offered，而非 77 completed 或 100 offered。
- 最高 token throughput 靠大 batch/长输出堆出来，batch 越大 TPOT/TTFT 越差；goodput 只统计满足 SLO 的请求，所以最优点在更小 batch，两者不是同一点。
- 还缺：拐点附近细扫（65/70/75）、长跑看队列漂移、多次重复与置信区间、P99 而非均值、preemption/KV 压力、失败与超时、客户端注入校验。

## 第 4 题：vLLM 负载发生器的统计方式
- `get_request()` 预生成 Poisson 间隔并 sleep，2,400/80 = **30s 发完**；`benchmark()` 的计时器在发请求前启动、在 `asyncio.gather` 全部返回后停止 → **55s = 30s 注入 + 25s 尾部**（最后一批请求的完整延迟）。
- `max_concurrency` 是客户端信号量 + `TCPConnector` 上限：C 太小时请求先在客户端排队，真实注入率 ≈ C/延迟 < 80，服务端根本没收到 80/s。
- `calculate_metrics()` 用 `completed / dur_s`，把 drain 摊进分母且只算成功 → 43.6 是“整批 cohort 平均”，不是服务端容量。
- 验证：客户端 dispatch 时间戳（应约 30s、~80/s）、in-flight 数、信号量/连接池；服务端 `request_success_total`、`num_requests_waiting/running`、token 计数按秒求导；去掉或放大 `max_concurrency`；只取稳态窗口算吞吐。

## 第 5 题：瓶颈归因设计
- 三个竞争解释：**H1 前端/CPU**（tokenization、API server、SSE 序列化）饱和导致 GPU 一直处于等待状态；**H2 prefill 受限**（长 prompt/chunked prefill），排队为 0 但 TTFT 高、KV 低；**H3 engine core 每步 Python/调度开销大、batch 小**，GPU 步间空转。可再加 **H4 客户端在 100/s 先饱和**。
- 最小实验：① 50 vs 100 的 rate ladder（是否随速率变化）；② 同速率换短 prompt/预 tokenize（区分 prefill vs 前端）；③ py-spy + nsys/DCGM（CPU 热点、GPU 空泡、SM active）；④ 多客户端、去掉并发上限（排除 H4）；⑤ 查 GPU 时钟/功耗/是否共享。
- 判据：接受“GPU 算力饱和”需 SM active≈100%、kernel 时间≈墙钟、无步间空泡、减小 batch 不改善 TTFT、加 GPU 吞吐线性上升。本例 **GPU 低 + Waiting=0 + KV 6%** 基本否定该结论，更像 CPU/前端瓶颈或 prefill 空泡。