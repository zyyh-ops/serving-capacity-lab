# Week 18 — 发布收口审计

> 审计日期：2026-11-23；Git 收口复审：2026-11-23T21:05+08:00。范围：`results/week18_*`、`docs/week18/*`、`results/week18_day7/evidence_audit.json`。
> `通过` = 证据已核实；`警告` = 证据缺失或非阻塞；`失败` = 发布阻塞项。

| 检查项 | 状态 | 证据 / 说明 |
|---|---|---|
| 1. 实验契约按版本/配置固定 | **通过** | vLLM 0.10.2 + V1 和完整 CLI 见 `results/week18_day5/servers/kv_512.log`、`results/week18_day6/servers/admission_256.log` 以及 Day 4 的 `.cmd` 文件。 |
| 2. CSV 与原始 JSON 账目一致 | **通过** | `results/week18_day7/evidence_audit.json`：`status=PASS`，审计 8 个 CSV、7 个 Day 6 `summary.json` 和 7 个 `request_records.json`，0 个问题。`scripts/test_audit.py` 的变异测试能检测出人为制造的 offered 计数不一致。范围仅限已实现的校验；Day 2/3 的逐请求原始记录没有完整重算。 |
| 3. 性能结论带单位与有效窗口 | **通过** | `docs/week18/day2_analysis.md` §1/§5 区分到达窗口、完成窗口和排空；`docs/week18/day6_analysis.md` §3/§4 统一使用 req/s、ms、tok/s，并区分 cohort 与窗口；`results/week18_day6/day6_comparison.md` 表头也写明了这一区别。目前还没有对外 README 结论（见检查项 8）。 |
| 4. 结果按相同实验配置分组 | **通过** | `docs/week18/day5_analysis.md` 表头把 BASELINE / SCHED_TRACE / NSYS_PROFILE 分开；`docs/week18/day6_analysis.md` §1 逐臂列出策略和阶段；已知限制第 1 条明确禁止把 Day 1–6 合并。Day 4 admission 的混杂（8,254 对比 7,448 blocks）已在 `docs/week18/day4_analysis.md` §0/§B 披露。 |
| 5. vLLM 源码追踪路径与 0.10.2 实现一致 | **通过** | 服务日志有 `Initializing a V1 LLM engine (v0.10.2)`（`results/week18_day5/servers/kv_512.log`、`results/week18_day6/servers/admission_256.log`）。源码追踪文档引用了本地 `vllm/...` 行号（`docs/week18/day2_source_trace.md`、`day4_source_trace.md`、`day5_source_trace.md`）。Scheduler 未被插桩（`vllm/v1/core/sched/scheduler.py` 中没有 `W18D5_SCHED_TRACE_V1`）。注意：当前审计 Shell 无法导入运行中的 vLLM。 |
| 6. Profiling 与 Scheduler 插桩限制已披露 | **通过** | `docs/week18/day5_analysis.md` §C/§E 披露了 22.71 s 窗口对比 41.26 s probe、`--cuda-graph-trace=node`、无 CPU 线程态、仅 1 条 NVTX、无 DRAM 计数器；`results/week18_day7/evidence_audit.md` 的解释边界；已知限制第 6/7 条。 |
| 7. 客户端拒绝/失败请求未隐藏 | **通过** | `results/week18_day2/summary.csv` 的 rate_120：`client_drops=403`；`results/week18_day6/day6_comparison.csv`：`local_reject=472`（mix_cost）、634（gate overload）；`results/week18_day6/recovery_open/summary.json` 总计中包含 `client_local_reject`；已知限制第 4 条。 |
| 8. README + 架构 + 运行说明正确 | **失败** | 根目录和 `docs/week18` 都没有 `README.md`；`docs/week17/README.md` 是 Week 17 运行手册，参数已过时（`max-model-len 1024`、`max-num-seqs 32`）；`docs/week18/week18_handoff_template.md` 仍有 `____` 占位符；未找到架构图。外部用户无法按仓库复现 Week 18。 |
| 9. 脚本/测试可运行；回归已保存 | **通过** | `python -m compileall -q scripts` 通过；`python scripts/test_contract.py` 通过；`python scripts/test_audit.py` 通过（因 DSH 沙箱拒绝 `tempfile.mkdtemp` 的 ACL，使用工作区 `TEMP` 绕过）；`python scripts/audit_week18.py --root . --out results/week18_day7` 通过，扫描 366 个文件，已保存 `results/week18_day7/evidence_audit.json`。干净 GPU 环境不属于本检查项，见证据工作表 F31。 |
| 10. 公开 Git 已清理密钥/私有路径/大文件 | **通过** | Git 已初始化（`main`，100 个 tracked files），`results/**` 被 `.gitignore` 排除；没有 tracked 的 `.sqlite`/`.nsys-rep`/`.env`/`.pem`/`.key`，也没有 >1 MB 的 tracked 文件。 |
| 11. Git SHA + 干净工作区已记录 | **通过** | `git rev-parse HEAD` = `caca754fddc3048c5111bd73eb1ac1b6e6f9a270`（`main`）；审计基线（提交 `caca754`）时 `git status --short` 为空；`git diff --check` 退出码 0；remote = `https://github.com/zyyh-ops/serving-capacity-lab.git`；`results/week18_day7/environment_fingerprint.md` 已记录 `git_commit`、`git_branch`、`git_dirty=false`。本次修改的两篇审计文档尚未提交，提交后需更新 fingerprint 中的 SHA。 |
| 12. 已知限制准确、非生产声明明确 | **通过** | 见下方已知限制第 1–8 条；`docs/week18/week18_handoff_template.md` 把 90 req/s 称为“非生产容量认证”；`docs/week18/day2_analysis.md` §5 和 `day6_analysis.md` §7 区分了观测 / 推断 / 未知。 |

## 发布决策

- 发布标签：`week18-serving-capacity-lab`（仅建议，不创建 Git tag）
- 最终审计结果： Git 基线已通过，仅剩 README/架构/运行说明阻塞。
- 阻塞项：
  1. **检查项 8**：缺 Week 18 README、架构图和可复现运行说明；唯一的 README 是过时的 Week 17 手册。
- 非阻塞项：
  - 容量结论仍是单 seed/有限窗口；95/100 req/s 的多 seed × 300 s 确认仍缺失。
- Git 基线：**通过** — `main`、HEAD `caca754fddc3048c5111bd73eb1ac1b6e6f9a270`、审计基线时 `git status --short` 为空、`git diff --check` 退出码 0。
- 最终 Git SHA（审计基线）：**caca754fddc3048c5111bd73eb1ac1b6e6f9a270**（`main`，审计时干净；审计文档提交后以新 SHA 为准）。
- 回归命令与结果：
  - `python -m compileall -q scripts` → 通过
  - `python scripts/test_contract.py` → 通过
  - `python scripts/test_audit.py` → 通过（工作区 `TEMP` 绕过）
  - `python scripts/audit_week18.py --root . --out results/week18_day7` → 通过（366 个文件；见 `results/week18_day7/evidence_audit.json`）
- 复审日期：2026-11-23（Git 收口复审 21:05+08:00）。

## 已知限制（即使检查通过也要保留）

1. Day 1–6 的历史实验使用了不同的 KV/Prefix/Graph 配置，不能合并成一条通用 SLO 容量曲线。
2. Day6 中观测到的最高 90 req/s 通过点只有单 seed、40 s；没有验证过的最大稳定生产吞吐/SLA。
3. 自定义客户端时间戳和 SSE 指标可能与服务端指标及 vLLM 官方 benchmark 口径不同。
4. 客户端本地合成拒绝不是真实网关/HTTP 429；未评估重试风暴或业务收益。
5. `P+2O` 工作负载成本模型是启发式的，未用 GPU 时间校准；在测试场景中对 Long 请求的拒绝分布不公平。
6. Scheduler 重复逻辑位置计数不是 FLOPs、GPU 时间或应用层正确性证明。
7. Nsight GPU gap 数据缺少足够的 vLLM NVTX 和 CPU 上下文，无法证明所有空闲时段的根因。
8. 只有一种 GPU/模型/负载族；性能不能泛化到其他硬件、batch 策略、prompt 或模型版本。

## 复审前需要补的发布项

- 编写 `README.md`、`docs/week18/architecture.md` 和 Week 18 运行手册，写清 `vllm serve` 命令、客户端命令、SLO 定义和证据路径；替换或明确标注 Week 17 README 为历史文档。完成后重跑检查项 8。
