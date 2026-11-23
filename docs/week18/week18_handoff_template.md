# Week 18 → Week 19 学习交接文档（待 Day7 验收完成后填写）

## 本周研究目标
真实 vLLM 0.10.2 V1 / RTX 4090 Serving：测量、容量、Workload Shape、Scheduler/KV、GPU/Host Profiling、SLO Admission。

## 已完成（有证据的数据）
- Day 1：Benchmark 方法论校准，理论 45/50。
- Day 2：Open-loop 容量曲线与 Overload Knee，理论 42/50。
- Day 3：Short/Long 干扰实验，理论 45/50。
- Day 4：Admission/KV 分离与 Preemption，理论 43/50。
- Day 5：Scheduler Trace 与 Nsight，理论 45/50。
- Day 6：SLO Gate、Recovery 和 Cost Gate 反例，理论 46/50。
- Day 7：理论__/50；Audit结果__；Release SHA__。

## 核心可复现结果（填写证据路径）

1. Workload P128/O128、vLLM 0.10.2 V1、RTX 4090、KV 4096、Prefix Cache 关：单 seed/40s、90 req/s 全达标；**非生产容量认证**。Evidence: ____
2. KV 512 blocks：70次抢占、45,420重复调度逻辑位置、29.6%调度量。Evidence: ____
3. Nsight 22.71s Kernel活动，71.5% GPU Busy、显著Step间隙；**Host因果未闭合**。Evidence: ____
4. 105 req/s短时过载：客户端RPS Gate将窗口Goodput由19.04提高至77.52 req/s，拒绝634/2601（24.4%）。Evidence: ____
5. Mixed 60 req/s: Cost Gate以不必要的拒绝降低Goodput，Long拒绝率显著更高。Evidence: ____

## 尚未解决的问题

- Capacity 95/100 req/s 多 Seed × 300s 确认；独立客户端；置信区间。
- GPU HBM / compute vs Host 准确归因：NVTX/CPU采样/硬件性能指标；同负载无Profiler对照。
- Production API Gateway admission/429、retry budget、公平性；GPU成本模型拟合。
- Prefix caching/length distribution/更多模型及跨GPU的泛化。

## 下一阶段建议（Week 19，建议并非已决定）

**SLO-aware Serving Optimization**：以可复现基线为前提，选一个调度/推理优化手段做有控制的A/B，严禁同时修改多项关键参数。

## Release 状态与可公开范围

- Release gate: ___
- Commit SHA: ___
- Public-facing claims verified: ___
- Artifacts remaining local due to size/sensitivity: ___
