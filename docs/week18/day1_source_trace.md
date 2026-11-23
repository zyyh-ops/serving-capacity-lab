# Week 18 Day 1 — Source Trace / Measurement Contract

## Environment & Evidence
- vLLM commit/tag (must be 0.10.2): **0.10.2**
- Python / torch / CUDA / driver / GPU: Python **3.12.16**（`python_info`）；torch **2.8.0+cu128**、CUDA **12.8**；driver **570.124.06**；**RTX 4090 24 GB**。
- Model / tokenizer revision: `Qwen/Qwen3-0.6B` + 同模型 tokenizer；架构 `Qwen3ForCausalLM`；权重加载 1.1201 GiB。
- Startup CLI (copy verbatim, incl. Graph/Compile config):
  ```bash
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve Qwen/Qwen3-0.6B \
    --dtype auto --max-model-len 2048 --gpu-memory-utilization 0.7
  ```
- Engine/runtime config: V1；`dtype=torch.bfloat16`；`max_seq_len=2048`；TP=PP=DP=1；`speculative_config=None`；`quantization=None`；`enforce_eager=False`；`kv_cache_dtype=auto`；engine `seed=0`；`chunked_prefill_enabled=True`；`max_num_batched_tokens=2048`；`use_async_output_proc=True`；`enable_prefix_caching=True`。
- Compilation / CUDA Graph: `compilation_config.level=3`；`cudagraph_mode=1`（PIECEWISE）；`use_inductor=true`；`full_cuda_graph=false`；`max_capture_size=512`；捕获 67 个 size（1…512）；从 `/root/.cache/vllm/torch_compile_cache/a0a5381945/...` **直接加载已编译图**（torch.compile 5.24 s），CUDA graph 捕获 2 s / 0.49 GiB，无 cold compile。
- Attention/sampling backend: Flash Attention on V1；**FlashInfer 不可用**，top-p/top-k 回退 PyTorch-native；本轮 bench 请求 `temperature=0.0`。
- Generation config: 未加 `--generation-config vllm`，server 默认 chat sampling 被 HF generation config 覆盖为 `temperature=0.6, top_k=20, top_p=0.95`；bench 请求显式传 `temperature=0.0`，主测按 greedy 发送。
- KV cache / capacity: 可用 **14.11 GiB / 132,064 tokens / 8,254 blocks**；2,048-token 请求最大并发 **64.48×**；本轮 128×(128+128)≈32k tokens，仅约 KV 容量 1/4.
- Warmup: engine init（profile + create KV cache + warmup model）12.23 s；每次 bench 先跑 1 个 initial single prompt test；另有 warmup。
- Prefix-cache status and between-run isolation method: `enable_prefix_caching=True`；cap1→cap128 **未重启 server、未显式清缓存**→ 第二轮可能复用 chat template 公共前缀，顺序/缓存效应未隔离。
- Server & bench host/network path: server 监听 `0.0.0.0:8000`；bench 用 `--host 127.0.0.1 --port 8000`，同容器/主机走 loopback，无跨机网络；server 12:33:33 ready，bench 12:58 运行。

## 1. Benchmark path (vLLM 0.10.2)
1. `vllm bench serve` -> selected entrypoint: CLI `bench serve` → `vllm.benchmarks.serve.main()` → `asyncio.run(main_async())` → `benchmark()`。
2. `vllm/benchmarks/serve.py`: `main_async` -> `get_request` -> `limited_request_func` -> `calculate_metrics`:
   - `main_async`：设 seed、加载 tokenizer/dataset、解析 `--goodput`，调用 `benchmark()`；
   - `get_request`：预计算 gamma/指数到达间隔并归一化到 `num_prompts/request_rate`，按计划 `await asyncio.sleep` 后 yield；
   - `benchmark`：每个 yield 建 task → `limited_request_func` →（可选）信号量 → `request_func`；最后 `asyncio.gather`；
   - `calculate_metrics`：只统计 `success`，算 `TPOT=(E2E−TTFT)/(output_len−1)`、池化 ITL、联合 goodput。
3. `vllm/benchmarks/backend_request_func.py` streaming path (`async_request_openai_chat_completions`):
   v0.10.2 实际路径是 `vllm/benchmarks/lib/endpoint_request_func.py::async_request_openai_chat_completions`。POST `stream=True` + `stream_options.include_usage`；`StreamedResponseHandler` 按空行切 SSE，去掉 `data: `，跳过 `:` 注释/ping 和 `[DONE]`；每个非 `[DONE]` data 事件先打 `timestamp`：首个带 `choices` 的事件置 TTFT（即使 `delta.content` 为空，role/空 delta 也会触发），之后每个带 `choices` 的事件都追加 `timestamp − most_recent_timestamp` 为 ITL（finish_reason/空 content 也算一个间隔）；`usage` 事件只取 `completion_tokens`，同时更新最后时间戳，故 `latency = 最后 data 事件 − st`。
4. Expected request arrival schedule vs actual request initiation (where does semaphore live?):
   计划是 20 req/s 的 Poisson 到达（`burstiness=1`）；`get_request` 只控制 yield 时刻。实际 HTTP 发起还要在 `benchmark()` 内的 `asyncio.Semaphore(max_concurrency)` 上排队，信号量在**客户端**，不在 server。cap1 时实际发起 ≈ 上一个请求完成，即 `1/E2E`；cap128 时才接近计划。
5. Definitions of client TTFT, ITL, E2E, TPOT; are SSE events always 1 token?:
   - TTFT = 首个带 `choices` 的 SSE 事件时间 − POST 前 `perf_counter()`（不要求 content 非空）；
   - ITL = 相邻两个带 `choices` 事件的时间差（空 delta/finish 事件也计入）；
   - E2E = 最后一个非 `[DONE]` data 事件时间 − 起点（含 usage 尾包，不是 response close）；
   - TPOT = `(E2E−TTFT)/(output_tokens−1)`，`output_tokens` 优先 `usage.completion_tokens`，否则 tokenizer 回退计数。
   **SSE 事件不保证 1 token**：一个事件可含多 token，也可能只有 role/空 delta/usage/finish，或被 UTF-8 或网络拆并；因此 `len(itl)` 可能多于或少于 `output_tokens−1`。本数据每请求 `len(itl)=128`，而 `output_len−1=127`，说明确有一个非 token 事件（role/空 delta/finish 之一）被计入了 ITL。
6. `--goodput` criteria and denominators; how are failed requests treated?:
   `--goodput ttft:200 tpot:15`（ms）；只对 `success` 请求逐条判 `actual ≤ SLO`（源码 `all(s >= r)`），分子 `good_completed`，分母是 benchmark `duration`，即 `request_goodput = good_completed / duration`。失败请求不计入 `completed` 和 goodput 分子，但仍占 `num_prompts`/offered；`output_len ≤ 1` 时 TPOT 记 0。
7. Which fields are preserved by `--save-detailed` in v0.10.2?:
   不加时删除 `input_lens`、`output_lens`、`ttfts`、`itls`、`generated_texts`、`errors`；加了则全部保留。本目录两个 JSON 都含这些逐请求字段，以及 `date/model_id/tokenizer_id/num_prompts/request_rate/burstiness/max_concurrency/duration/completed/tokens/throughput/percentiles`。

## 2. Server Metrics path
- `vllm/v1/metrics/loggers.py` histogram/gauge/counter definitions:
  - Gauge：`vllm:num_requests_running`（“Number of requests in model execution batches”，`mostrecent`）、`vllm:num_requests_waiting`（“Number of requests waiting to be processed”）、`vllm:kv_cache_usage_perc`（1=100%；旧名 `vllm:gpu_cache_usage_perc` 已 deprecated）。
  - Counter：`vllm:num_preemptions_total`、`vllm:prompt_tokens_total`、`vllm:generation_tokens_total`、`vllm:request_success_total{finished_reason=...}`。
  - Histogram：`vllm:time_to_first_token_seconds`（0.001…2560 s 桶）、`vllm:inter_token_latency_seconds`（0.01…80 s；旧名 `time_per_output_token_seconds`）、`vllm:e2e_request_latency_seconds`（0.3…7680 s）。
- `/metrics` sample names observed (copy exact names from endpoint, do not assume):
  `vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`、`vllm:gpu_cache_usage_perc`、`vllm:num_preemptions_total`、`vllm:prompt_tokens_total`、`vllm:generation_tokens_total`、`vllm:request_success_total`、`vllm:time_to_first_token_seconds_{bucket,count,sum}`、`vllm:time_per_output_token_seconds_*`、`vllm:inter_token_latency_seconds_*`、`vllm:e2e_request_latency_seconds_*`，以及 `python_info`/`process_*`。
- Why sample Waiting Peak=0 cannot prove no queue ever existed:
  只有 `metrics_before`、`after_cap1`、`after_cap128` 三个**瞬时快照**，不是连续抓取；跑完后 waiting=0 是必然的，运行中的亚秒级峰值会被完全漏掉。`num_requests_waiting` 还只覆盖 server 调度队列，不含客户端信号量排队、HTTP/frontend 排队、chunked-prefill/token-budget 等待或抢占。故快照里的 0 只是“采样时刻为 0”的下界，不能证明从未排队。
- Client-time vs server-time timing boundary differences:
  client 从 POST 前开始计时，含连接/HTTP/frontend/tokenization/排队/prefill；server 直方图从 engine 收到请求（`arrival_time`）到首 token，不含网络和客户端排队；client E2E 到最后一个非 `[DONE]` SSE 事件（含 usage 尾包），server e2e 到 engine 完成，尾部 detokenize/usage 包可能不同。实测 client mean TTFT 15.52/17.21 ms，而 server 累计均值约 6.98/7.23 ms（`sum/count`，且含 warmup/探针），差约 8–10 ms，正是边界差。

## 3. Calibration results
| Run | Offered Rate target | Client cap | Completed | Duration (s) | Request tok/s? | Request req/s | Goodput req/s | P99 TTFT (ms) | P99 TPOT (ms) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cap1 | 20 | 1 | 64 | 33.96 | 241.2 out（总 482.3） | 1.884 | 1.884 | 20.44 | 4.30 | 闭环：1 在途，实际≈1/0.530s；0 失败 |
| cap128 | 20 | 128 | 64 | 3.79 | 2163.2 out（总 4325.3） | 16.900 | 16.900 | 25.60 | 4.73 | 64/3.79s；目标 20，含尾部 drain；0 失败 |

> “Request tok/s?”按输出 token 吞吐填，括号为总 token 吞吐。两轮 completed=`num_prompts`=64，errors=0，output_lens 全 128；goodput=throughput 说明 64 条全部同时满足 TTFT≤200 ms 且 TPOT≤15 ms（P99 仍远低于 SLO）。

## 4. Correctness / Measurement audit
- JSON self-consistency check output: [day1_audit.json](results/week18_day1/day1_audit.json) `all_pass=true`，`checks_failed=[]`；两条 warning 均为 “Small completed sample: p99 unstable; calibration only”。`completed=64`、`failed_request_records=0`。
- Failed requests / token lengths / SLO: 失败 0；input_lens 62×128 + 1×127 + 1×125（tokenizer 边界），output_lens 64×128；`ttft:200 tpot:15` 全部通过。
- Client backpressure hypothesis, evidence and alternative explanations:
  cap1 的 1.884 req/s ≈ `1/0.530 s`，duration≈`64×0.530`≈33.9 s，说明 `max_concurrency=1` 让下一条必须等上一条完成；信号量在客户端，所以 server 的到达率被压到 ~1.9 req/s。cap128 为 16.9 req/s：64 条按 20/s 注入需 3.2 s，而 3.79 s 的分母含尾部 drain，故低于 20 不能直接归因于 server 饱和。替代解释：64 条样本太短、Poisson 抖动、缓存/顺序效应；server 快照 waiting/running/KV 均为 0、preemptions=0，不能支持“已饱和”，但快照太稀疏也不能排除瞬态排队。
- Dataset + seed + repeated prefix / cache warming confounders:
  `random`、input/output 128/128、`random_range_ratio=0`、seed 2026、temperature 0、`ignore_eos`；两轮同一 server 连续跑（cap1→cap128），未重启、未清 prefix cache（启动日志确认 `enable_prefix_caching=True`），第二轮可能命中 chat template 公共前缀，顺序与缓存效应未量化；`metrics_before` 还含 warmup/探针的累计 token。
- Re-run requirements before capacity claims:
  交换顺序并重启 server / 关 prefix cache 各跑一遍；每点至少数千请求或 5–10 min；开环 `--request-rate` 且不设 `max-concurrency`，同时记录客户端实际发送时刻；多 seed/多轮重复；连续抓 `/metrics`（0.1–0.2 s）或 Prometheus；报告 p99 置信区间与 joint goodput；固定并归档启动命令、模型 revision、graph/compile 状态。

## 5. Evidence-strength classification
**Observed**:
- cap1：64/64 成功，33.96 s，1.884 req/s，P99 TTFT 20.44 ms，P99 TPOT 4.30 ms；
- cap128：64/64 成功，3.79 s，16.900 req/s，P99 TTFT 25.60 ms，P99 TPOT 4.73 ms；
- 两轮 goodput=throughput（全部满足双 SLO）、0 失败、output 全 128；
- 三个 `/metrics` 快照的 waiting/running/KV 均为 0、preemptions=0；server 累计 TTFT 均值约 7 ms，低于 client 的 15–17 ms。

**Inferred, not established**:
- cap1 的低吞吐由客户端 `max_concurrency=1` 背压导致（数值高度吻合，但未做无 cap 对照）；
- cap128 的 16.9 < 20 主要来自 64 条样本的尾部 drain，而非 server 上限；
- server 在该负载下大概率未饱和（TTFT/TPOT 很低、无 preemption；128×256 tokens 仅约 KV 容量 1/4），但缺少连续队列数据；
- prefix cache/运行顺序可能影响第二轮 TTFT，方向与幅度未测。

**Unknown / cannot conclude**:
- 最大稳定容量、可持续 offered rate、20 req/s 是否能长期满足 SLO；
- 运行期间的 Waiting/Running/KV 峰值与真实排队时长；
- GPU 利用率/MFU、模型 revision（未 pin）；
- p99 的统计不确定性（64 条样本不足）。

## 6. Day 2 readiness
- Working reproducible command + raw JSON and logs:
  `vllm bench serve --backend openai-chat --host 127.0.0.1 --port 8000 --endpoint /v1/chat/completions --model Qwen/Qwen3-0.6B --dataset-name random --random-input-len 128 --random-output-len 128 --random-range-ratio 0.0 --num-prompts 64 --request-rate 20 --burstiness 1.0 --max-concurrency {1|128} --seed 2026 --ignore-eos --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 --goodput ttft:200 tpot:15 --save-result --save-detailed --result-dir results/week18_day1 --result-filename client_cap{N}.json --disable-tqdm`
  原始文件：[client_cap1.json](results/week18_day1/client_cap1.json)、[client_cap1.log](results/week18_day1/client_cap1.log)、[client_cap128.json](results/week18_day1/client_cap128.json)、[client_cap128.log](results/week18_day1/client_cap128.log)、[day1_audit.json](results/week18_day1/day1_audit.json)。
- Identified sampling and concurrency caveats:
  64 条短样本，p99 不稳；`--request-rate=20` 不等于实际注入率；`max-concurrency` 会把背压放在客户端；`/metrics` 只有 3 个快照，不能测峰值排队；两轮同 server、未清 cache，顺序混淆。
- Proposed longer open-loop sweep protocol:
  固定模型/GPU/dtype/长度/seed/启动参数，warmup 与 graph capture 完成后，以 `--request-rate 5/10/20/40/80`、**不设 max-concurrency**、每点 ≥5 min 或 ≥2000 请求、3–5 次重复；每点重启 server 或关 prefix cache；连续采样 `/metrics` 0.1–0.2 s（或 Prometheus），记录客户端实际 send 时刻、achieved λ、p99 TTFT/TPOT、joint goodput、waiting/running/KV/preemption；画 latency/goodput–λ 曲线，把“p99 TTFT≤200 ms 且 TPOT≤15 ms 的最大 λ”作为容量，再继续加压确认 waiting 增长与吞吐平台。
