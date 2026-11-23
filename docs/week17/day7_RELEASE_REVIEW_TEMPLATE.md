# Week 17 · Real vLLM Inference — Release Review

> 状态：**DRAFT / 待人工审核**。本文件是模板，不代表已经通过 Release Closure。

## 1. Scope / 版本边界

- 对象：vLLM **v0.10.2 V1**，`Qwen/Qwen3-0.6B`，单卡 RTX 4090，BF16，TP=1。
- 项目性质：**源码分析 + 插桩追踪 + 可复现实验**；不是从零实现或改造生产级 vLLM Runtime。
- 证据基准：固定 Git tag/commit、启动命令、软件依赖与 GPU 信息，以仓库证据为准。
- 未验证内容：多卡 vLLM TP/PP 生产部署、生产级高负载稳定性及故障恢复；不得借用 Week 16 的教学 Reference 结果替代 Week 17 的真实 vLLM 证据。

## 2. Architecture / 控制流

`HTTP → AsyncLLM → EngineCoreClient → EngineCore → Scheduler → Worker/GPUModelRunner → Qwen3/FlashAttention → Sampler → ModelRunnerOutput → OutputProcessor → SSE`

按源码文件、版本和函数签名补足：

- Day 2：Request Lifecycle / IPC 与进程边界；见 `docs/day2_source_trace.md`
- Day 3：RUNNING/WAITING、KVCacheManager、BlockPool、Prefix Caching、Preemption；见 `docs/day3_source_trace.md`
- Day 4：InputBatch、Query Start、Positions、Slot Mapping、Attention Backend、Sampler；见 `docs/day4_source_trace.md`
- Day 5：Compilation Level 3、Graph NONE/PIECEWISE、Capture & Replay；见 `docs/day5_source_trace.md`
- Day 6：客户端 TTFT/TPOT/ITL 与服务端 Prometheus；见 `docs/day6_benchmark_analysis.md`

## 3. Evidence Matrix / 关键结果

| 实验 | 需保留的原始证据 | 预期可支持的结论 | 人工复核 |
|---|---|---|---|
| Day 1 环境/Smoke | 环境、启动日志、Smoke JSON | 模型在指定环境可正常部署 | [ ] |
| Day 2 请求生命周期 | Request trace / source trace | 控制流、IPC、Scheduler、输出回传链路 | [ ] |
| Day 3 Prefix/Pressure | `day3_prefix.json`、`day3_pressure.json`、Scheduler 原始 Trace（若有） | 相同请求输出、Cache 命中和抢占机制 | [ ] |
| Day 4 Mixed Batch | `day4_runner_test.json`、`day4_runner_trace.log` | 真实 Batch Query 累加与 Metadata 不变量 | [ ] |
| Day 5 Graph A/B | `day5_no_graph.json`、`day5_piecewise.json`、Nsight Summary | 保持编译级别一致时，Graph 路径的相对收益 | [ ] |
| Day 6 Serving | 12 个 Benchmark JSON 与对应 Metrics JSONL | 1–32 并发、2–20 req/s 和三类工作负载的实测行为 | [ ] |

### 可报告的数据（必须注明实验范围）

- Day 3：Cold 275.87 ms、Warm 119.75 ms（**单次顺序请求观察值，不能直接当成严谨加速比基准**）。
- Day 4：77 条 ModelRunner Trace 检查 Query 索引不变量；Step 53 为 `[1,60,67]` 的 Mixed Batch。
- Day 5：Compiled No Graph vs Compiled PIECEWISE，单请求 E2E p50 **1066.34→414.39 ms（2.57×）**；8 并发 **1183.28→509.44 ms（2.32×）**。
- Day 6：Short Decode C=1→32，**248.83→5796.28 output tok/s**；E2E p50 **512.6→697.8 ms**；按实验设定的 200/15 ms SLO，12 组均达标。
- Day 6：Long Prefill C=8，客户端 ITL p95 **12.31 ms**；服务器采样 Waiting 峰值为 **3**；不能仅从客户端 ITL 尖峰断言某一个特定调度步骤是根因。

## 4. Benchmark Contract / 复现契约

必须记载：

- 模型与 tokenizer 的 revision/版本、vLLM commit/tag、CUDA/Driver、GPU、PyTorch、dtype、TP。
- 服务器启动命令、compilation config、Graph 模式、KV 大小、max-model-len、max-num-batched-tokens、max-num-seqs、Prefix Cache 开关。
- 生成负载（input/output token、seed、ignore_eos、request-rate、max-concurrency）、Warmup、重复数、实际返回 token 数。
- TTFT/TPOT/ITL 的统计对象和百分位数计算方式；记录失败、丢失请求、预热混入 Counter 的情况。
- 若报告 Nsight 细粒度数据，应归档可复算 SQL 和原始 `.nsys-rep` / SQLite 或标注“仅由现有汇总支持”。

## 5. Known Limitations / 不应隐藏的边界

1. **不是生产级 Runtime 实现**：这是一项 vLLM V1 源码研究和实验复现项目。
2. **模型/硬件范围有限**：单 RTX 4090、Qwen3-0.6B；无证据支持跨模型、跨 GPU 或多卡外推。
3. **未测出最大饱和吞吐**：C=32 是所测并发范围的上限；未做长期超负载与稳定性验证。
4. **SLO 200/15 ms 不具区分力**：现有 12 组全部达标；更严格的 SLO 是事后敏感性分析。
5. **Profiling 与测量边界**：Nsight 插桩影响性能；API 延迟与 GPU Kernel 耗时不是同一指标。
6. **Day 6 缺独立服务启动日志**：环境/配置部分依靠 README 或前期产物，需标注证据级别。
7. **Day 3 抢占与 Day 5 Nsight 部分缺本次归档原始 Trace**：保留报告中的观察结论，但标注可复核性限制。
8. **服务端 Metrics 为 0.5 秒采样**：Waiting=0 不等于从未排队；ITL 尖峰不能独立完成因果归因。

## 6. Documentation / Regression / Git Release Checklist

- [ ] `README.md`：一屏解释项目定位、环境、架构、运行方式、验证结果与限制
- [ ] 所有源码行号按 **v0.10.2** 校核（文档不得引用与本 tag 不符的新 API）
- [ ] Day 1–6 结果 JSON 可解析，指标重新计算与报告一致
- [ ] Day 5 同请求输出哈希一致，Day 4 Query 不变量仍通过
- [ ] Day 6 12 个负载点及 Metrics 均齐全，原始文件不作事后改写
- [ ] Nsight 时间/统计口径明确，未将 `CUDA Graph` 说成 `Kernel Fusion`
- [ ] 清除/轮换真实 API Key、Token、环境私密地址、绝对个人路径；仅提交可公开的脱敏示例
- [ ] `.gitignore` 排除虚拟环境、模型权重、敏感 `.env`、大体积临时文件
- [ ] `git diff --check`、脚本 Syntax/Regression PASS
- [ ] `git status --short` 复核，Commit ID 与验收结果对应
- [ ] Release Tag、日期和 SHA256 Manifest 均记录；不做未经验证的生产宣称

## 7. Release Decision

- 自动证据审计：`PENDING` / `PASS` / `REVIEW_REQUIRED`。
- 人工 Documentation Review：`PENDING`。
- 回归/Correctness：`PENDING`。
- 安全/敏感信息检查：`PENDING`。
- 最终结论：**PENDING — 未完成前不得写 RELEASED**。

## 8. Day 7 综合验收题答案

另建 `docs/day7_theory_answers.md`，逐题回答，未回答题保持空白，不要填充猜测。
