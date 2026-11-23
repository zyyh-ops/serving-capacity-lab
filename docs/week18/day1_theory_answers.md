## 第 1 题：Load 与吞吐口径

- **Offered Rate** = 1200 / 60 = **20 req/s**
- **Achieved Throughput** = 1000 / 60 ≈ **16.7 req/s**（只统计成功完成的请求；80 个失败和 120 个仍在执行的请求不计入）
- **Request Goodput** = 900 / 60 = **15 req/s**（满足 SLO 的请求数 / 窗口时长）

不能据此断言 GPU 的最大稳定容量：16.7 < 20，说明系统已经处理不过来；120 个仍在执行的请求，其结果尚未确定；失败请求又被剔除，因此 60 秒的完成数天然偏乐观。吞吐量只是一个窗口的平均值，容量是“持续 offered λ 下仍满足 SLO”的最大值，需要把速率逐步提高到饱和，并看 p99 和排队情况。而且 GPU 容量本质上是 token/s，会随输入、输出长度变化。

## 第 2 题：TPOT 与 ITL

按仓库中的 TPOT 定义 `ΣITL/(N−1)`，等价于 `(E2E−TTFT)/(N−1)`：

| 请求         | TPOT                    | 非首 token 间隔数 |
| ------------ | ----------------------- | ----------------- |
| A（101 tok） | (440−40)/100 = **4 ms** | 100               |
| B（3 tok）   | (240−40)/2 = **100 ms** | 2                 |

- 请求等权平均 TPOT = (4+100)/2 = **52 ms**
- 整体平均 ITL = (400+200)/(100+2) = 600/102 ≈ **5.88 ms**

两者相差明显，是因为**权重不同**：等权平均先算每个请求的“比值”，再让请求各占 50%；B 只有 2 个间隔，却拿走一半权重。整体 ITL 是所有间隔合并后算“总时间/总间隔”，B 只占 2/102≈2%，所以被 A 的 100 个 4ms 间隔主导。B 的单个慢间隔也会被 TPOT 均摊成 100ms。

## 第 3 题：客户端背压

- 闭环上限由 Little's law 给出：λ = C / E2E = 4 / 0.5 = **8 req/s**。目标 40 req/s 根本发不出去，多出的请求堵在**客户端**信号量/队列里。
- 所以即使服务端 Waiting 长期为 0，也很正常：服务端从未被压到排队；这只能说明“4 并发下没排队”，不能说明能稳定处理 40 req/s。0.5s 采样还会漏掉短峰值。
- 要补的测量：客户端实际发送速率、在途数、本地排队时间；开环 `--request-rate` 且**不设** `max-concurrency`，扫 10/20/40/80 req/s，长跑多轮；看 achieved λ、p99 TTFT/TPOT、joint goodput、waiting、running、KV、preemption（0.1–0.2s + Prometheus）。40 req/s、500ms 至少需要 C≈20。

## 第 4 题：SLO Goodput

- 完成请求吞吐 = 100 / 10 = **10 req/s**
- 联合 SLO Goodput = 50 / 10 = **5 req/s**
- 成功完成率 = 100/120 ≈ **83.3%**；成功请求中双 SLO 通过率 = 50/100 = **50%**

不能分别算通过率再相乘：65% × 60% = 39%，与实际的 50% 不符。原因是 **TTFT 与 TPOT 不独立**（同一请求、同一排队/批次，长 prompt 会同时推高两者，相关性可正可负），而且 20 个失败请求不进任何 SLO 分子。Goodput 必须逐请求做 `TTFT≤200 && TPOT≤15` 再计数，分母统一用窗口时长（或统一的 offered 数），不能把两个边际概率相乘。

## 第 5 题：性能证据可信性

至少四处不足：

1. **样本太小**：128 个请求的 p99 实际由最差 1–2 个样本决定，置信区间很宽；[audit_bench.py](scripts/audit_bench.py#L71) 也把 <100 标为“p99 unstable”，128 只是勉强够，180ms 距 200ms 仅 10% 余量，谈不上“stable”。
2. **Waiting Peak 不可靠**：来自 0.5s 间隔的采样，短排队会被漏掉；且 `waiting=0` 只代表采样瞬间服务端没排队，不含客户端排队、prefill/token-budget 排队和抢占。
3. **负载未定义**：没有 offered vs achieved rate、到达过程和持续时长；128 个请求可能是闭环/低负载，服务端本来就压不满，“没有排队”是低负载下的必然结果。
4. **结论缺联合 SLO 与重复**：只报 TTFT p99，没有 TPOT、错误率、goodput、多轮重复与置信区间，也没有配置、预热、cache/graph 状态，既不可复现也不可外推。

补充实验：

- 固定模型/GPU/dtype/输入输出长度/seed，prefix cache 关闭、完成预热和 graph capture；
- **开环**恒速或 Poisson 扫 5/10/20/40/80 req/s，每点 ≥5 分钟或 ≥2000 请求，独立重复 3–5 次，rate 模式不设 `max-concurrency`；另做闭环 concurrency 1–64 对照；
- 采集每请求 TTFT/TPOT/E2E、失败与超时、客户端实际 λ、joint goodput，以及服务端 waiting/running/KV/preemption（0.1–0.2s 采样 + Prometheus 直方图）；
- 画 latency/goodput–offered λ 曲线，取“p99 TTFT≤200ms（bootstrap CI）且 TPOT 达标”的最大 λ 作为容量，再继续加压确认 waiting 增长/吞吐平台；用 `audit_bench.py` 校验 `completed==num_prompts`、`p50≤p95≤p99` 和 JSON 自洽。