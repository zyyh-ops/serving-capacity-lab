# Day 7 — 证据审计工作表

> 审计日期：2026-11-23。范围：`results/week18_*`、`docs/week18/*`、`results/week18_day7/evidence_audit.json`。
> `[x]` = 已核实通过；`[ ]` = 警告 / 失败 / 未检查。路径均为工作区相对路径。

## A. 运行时身份
- [x] vLLM==0.10.2 + V1 已在保存日志中确认 — **通过** 
- [x] GPU 型号、驱动、CUDA、torch、精度、模型和 revision 已记录 — **警告** — `results/week18_day7/environment_fingerprint.md` 记录了 RTX 4090 / 570.124.06 / 24,564 MiB / torch 2.8.0+cu128 / Python 3.12.16；模型 revision 在 server config 日志中是 `None`，没有固定。
- [x] 每个结果的有效 `max_num_seqs`、`max_num_batched_tokens`、模型最大长度已记录 — **通过** —  `.cmd` 和 server log 记录了 `max_num_seqs=256/512`、`max_num_batched_tokens=2048`、`max_model_len=2048`。
- [ ] `num_gpu_blocks` 有效值、block size 和 prefix cache 设置已记录 — **警告** — Day 4–6 日志记录了 `num_gpu_blocks=512/1024/4096`，block size 16（`docs/week18/day4_analysis.md` §A），prefix cache 关闭。Day 2/3 没有 server log；Day 2 使用自动 KV（推断 8,254 blocks）且 prefix cache 开启（`docs/week18/day2_source_trace.md` §1）。
- [ ] CUDAGraph 模式、capture sizes、`enforce_eager`、profiler/patch 模式已记录 — **警告** — Day 4–6 日志显示 `cudagraph_mode=1`（PIECEWISE）、67 个 capture sizes、`enforce_eager=false`；Day 5 的 scheduler 插桩记录在 `scripts/patch_scheduler.py` 和 `docs/week18/day5_source_trace.md`；nsys 模式见 `docs/week18/day5_analysis.md` §C。Day 2/3 没有 server log。

## B. 数据集与到达过程
- [x] 请求负载形状和 tokenizer/chat-template 长度已验证 — **通过** — `docs/week18/day3_analysis.md` §3/§4：实际 P50 为 136/520，而不是名义的 128/512（chat template +8），输出长度 mismatch 为 0；`docs/week18/day4_analysis.md` §A；`docs/week18/day6_analysis.md` §1。
- [x] 到达窗口、Poisson seed、计划 vs 实际派发速率已记录 — **通过** — `results/week18_day2/confirm_90_seed2027/summary.json` 的 meta（`arrival_duration_s=240`、`seed=2027`、`offered_rate_realized`、`injected_rate_dispatch_proxy`）；`results/week18_day6/recovery_gate/summary.json` 的 meta/phases；`docs/week18/day2_analysis.md` §1/§3.1。
- [x] 客户端 guard drop、并发/连接池、事件循环延迟已计入 — **通过** — `results/week18_day2/summary.csv` 的 rate_120：`client_drops=403`、`dispatch_lag_p99_ms=13.13`；`results/week18_day6/day6_comparison.csv`：`guard_drop=0`；`docs/week18/day3_analysis.md` §3.5 记录了约 200 ms 的客户端事件循环停顿。
- [x] Warmup、运行顺序、prefix cache 影响已披露 — **通过** — `docs/week18/day2_source_trace.md` §1（warmup、prefix cache 开启、无 server log）；`docs/week18/day3_analysis.md` §1（每类 1 条 warmup，prefix cache 跨 run 累积）；`docs/week18/day6_analysis.md` §1（同一 server 进程、固定顺序、无 AB/BA）。
- [x] 混合负载中短/长请求分类报告 — **通过** — `results/week18_day6/mix_open/summary.json` 的 `categories`；`docs/week18/day6_analysis.md` §5（short/long 的 offered/good/reject）；`docs/week18/day3_analysis.md` §3。

## C. 延迟、goodput 与完成
- [x] TTFT 取首个 SSE choices 事件；E2E 取最后一个 data/usage 事件 — **通过** — `docs/week18/day2_source_trace.md` §2.3（首个 `choices` 事件；`usage` 尾包更新 `most_recent_timestamp`）；`scripts/openloop_probe.py` 的 `first_choice`/`last_data`。
- [x] TPOT 分母为 `(output_tokens-1)`，token 数取自 usage — **通过** — `docs/week18/day2_source_trace.md` §2.3；`docs/week18/day3_analysis.md` §4（`(e2e−ttft)/(usage_tokens−1)`）。
- [x] 联合 SLO 为 TTFT <=200 ms 且 TPOT <=15 ms（已核对实际配置） — **通过** — `results/week18_day2/confirm_90_seed2027/summary.json` 的 meta（`ttft_slo_ms=200`、`tpot_slo_ms=15`）；`results/week18_day6/recovery_gate/summary.json` 的 meta；`docs/week18/day5_analysis.md` §B。
- [x] Offered cohort、accepted-only cohort、completion window 分开 — **通过** — `docs/week18/day6_analysis.md` §3/§4；`results/week18_day6/day6_comparison.md` 表头；`results/week18_day7/evidence_audit.md` 的 解释边界。
- [x] 本地拒绝、错误/超时、drain 后结果未隐藏 — **通过** — `results/week18_day6/day6_comparison.csv` 的 `local_reject`；`results/week18_day2/summary.csv` 的 `failed_timeout=0` 和 `client_drops`；`results/week18_day6/recovery_open/summary.json` 总计中的 `client_local_reject`、`http_error_or_timeout`。
- [x] 窗口完成数可包含前序 phase 的到达；没有错误不等式 — **通过** — `scripts/test_audit.py` 的合成测试允许 `window_completed=2` 而 recovery 只有 1 条到达，且 `window_good > cohort_good`；`docs/week18/day6_analysis.md` §4 解释了 `completed 68.93 > injected 65.58`。
- [ ] 置信区间/多 seed 状态如实报告 — **警告** — 没有 CI 或多 seed 重复。该限制已在 `docs/week18/day2_analysis.md` §4.3/§5.3、`docs/week18/day6_analysis.md` §6、已知限制 第 2 条中披露。

## D. 资源与抢占
- [x] Waiting/Running gauge 和采样间隔已报告；采样峰值是下界 — **通过** — `results/week18_day2/confirm_90_seed2027/summary.json` 的 meta 为 0.25 s 采样；`docs/week18/day2_analysis.md` §1；`docs/week18/day5_analysis.md` §C（78 个样本、0.25 s）。
- [x] KV 百分比按各臂真实 block 数解释 — **通过** — `docs/week18/day4_analysis.md` §A/§B：8,254 vs 7,448 blocks，`kv%` 分母不同，不能跨臂直接比较；`docs/week18/day5_analysis.md` §A 两臂都使用 4,096 blocks。
- [x] `num_preemptions_total` 重置/缺样本已检查 — **通过** — `docs/week18/day4_analysis.md` §C（0→356/360，单调）；`docs/week18/day5_analysis.md` §B（70 次计数器 = 70 个 trace 事件；无重置）。
- [x] 抢占 trace 记录 request ID、step、抢占/恢复状态、逻辑 token 区间 — **通过** — `results/week18_day5/kv_512_trace.jsonl`；`scripts/patch_scheduler.py` 输出 `request_id`、`step`、`previously_computed`、`start`、`n`；`results/week18_day5/kv_512_trace_summary.json`。
- [x] 重复逻辑位置没有被描述为直接 FLOPs/时间测量 — **通过** — `docs/week18/day5_analysis.md` §B；已知限制 第 6 条。
- [x] 源码机制与观测事件分开引用 — **通过** — `docs/week18/day5_source_trace.md`；`docs/week18/day4_source_trace.md`；`docs/week18/day5_analysis.md` §E。

## E. GPU/Host Profiling
- [ ] Nsight 采集窗口确实覆盖目标负载 — **警告** — `docs/week18/day5_analysis.md` §C：kernel 活动窗口 22.71 s，而 probe 为 41.26 s；窗口从 probe 第 18.4 s 开始，只覆盖后约 55%，且包含 drain；不是完整的 60 s 请求窗口。
- [x] 原始 `.nsys-rep`/SQLite 或导出摘要可用 — **通过** — `results/week18_day5/nsys/admission_256_delay90_dur60.sqlite`（125.3 MB）、`kernel_summary.csv`、`kernel_exec_summary.csv`、`cuda_api_summary.csv`；原始 `.nsys-rep` 在 `docs/week18/day5_analysis.md` §C 中有引用。
- [x] CUDA Graph node tracing/profiling 开销已披露 — **通过** — `docs/week18/day5_analysis.md` §C：`--cuda-graph-trace=node`、37,526 次 `cudaGraphLaunch`、约 196 次 launch/step；TPOT p99 +15%、E2E p99 +14%，但 rate/seed 不同，报告将其定为弱证据。
- [ ] GPU kernel busy 计算未重复计入重叠 stream — **警告** — `docs/week18/day5_analysis.md` §C 在单卡上把 kernel 活动相加得到 16.24 s busy，但没有显式的 per-stream 重叠审计；多 stream 重叠可能重复计时。
- [x] Host 同步等待没有被算作 GPU 空闲时间 — **通过** — `docs/week18/day5_analysis.md` §C/§D 把 `cudaEventSynchronize` 5.78 s（1,619 次）与 4.805 s gap 分开，并明确两者重叠；`docs/week18/day7_theory_answers.md` 第 3 题定位到 `_to_list()`（`gpu_model_runner.py:3730-3742`）。
- [ ] 在给出确定性 host 瓶颈结论前，已有 CPU NVTX/range 和线程状态 — **警告** — 只有 1 条 NVTX 标记（“CCCL”），且 `--sample=none --cpuctxsw=none`（`docs/week18/day5_analysis.md` §C），没有 CPU 线程态采样。报告正确地没有给出确定性的 host 瓶颈结论。

## F. 发布控制
- [x] Python/Bash 单元回归测试通过 — **通过（CPU）** — `python -m compileall -q scripts` 通过；`python scripts/test_contract.py` 通过；`python scripts/test_audit.py` 通过（因 DSH 沙箱拒绝 `tempfile.mkdtemp` 的 ACL，使用工作区 `TEMP` 绕过）；`python scripts/audit_week18.py --root . --out results/week18_day7` 通过（366 个文件）。
- [ ] 所有 benchmark 脚本和依赖在干净环境可运行 — **警告** — 当前 Python 没有 `vllm`（`importlib.metadata.PackageNotFoundError`）；本次审计没有做干净 GPU/服务端启动。只执行了 CPU 侧的 audit/contract/compile 检查。
- [ ] README、架构图、trace 路径与代码一致 — **失败** — 根目录和 `docs/week18` 都没有 README；`docs/week17/README.md` 是 Week 17 的运行手册，参数已过时（`max-model-len 1024`、`max-num-seqs 32`）；`docs/week18/week18_handoff_template.md` 仍有 `____` 占位符；没有架构图。源码追踪路径本身与本地 `vllm/` 子集一致。
- [x] 已知限制和证据强度已标注 — **通过** — 已知限制第 1–8 条；`docs/week18/day2_analysis.md` §5；`docs/week18/day5_analysis.md` §E；`docs/week18/day6_analysis.md` §7。
- [ ] 未提交密钥/私有 URL/机器 ID/UUID/绝对私有路径/大 profiling 文件 — **警告** — 不是 Git 仓库，无法做受控文件扫描；`.gitignore` 已排除 `results/**`，但工作区仍有 `results/week18_day5/nsys/admission_256_delay90_dur60.sqlite`（125.3 MB）和至少一个绝对工作区路径（已脱敏）。源码 grep 未发现明显密钥，但这不是完整的安全审计。
- [ ] `git diff --check`、干净状态、最终 SHA、回归日志已保存 — **失败** — `results/week18_day7/environment_fingerprint.md` 写明 `git: not a repository`；没有 `git diff --check`、没有 SHA、没有干净状态记录。唯一保存的回归产物是 `results/week18_day7/evidence_audit.json`。

## 审计结论
- 通过（所有必需证据均已核实）：**否** — A/B/C/D 核心测量证据通过了已实现的检查；发布收口 F32/F35 失败。
- 警告（缺失/非阻塞证据）：A2、A3、A4、A5、C17、E24、E27、E29、F31、F34。
- 失败（计算/正确性/安全阻塞）：F32（缺 Week 18 README/架构/运行手册）、F35（无 Git SHA/干净工作区）。
- 剩余阻塞项：（1）编写 Week 18 README、架构图和运行说明，并消除 Week 17 README 的歧义；（2）把项目纳入 Git（或提供源码控制导出），只提交代码/文档，记录最终 SHA 和干净状态；（3）对外发布前移走/忽略 125.3 MB 的 Nsight SQLite，并移除绝对私有路径。
- 审计人/日期/Git SHA：**AI 审计 / 2026-11-23 / 不适用（非 Git 仓库）**。
