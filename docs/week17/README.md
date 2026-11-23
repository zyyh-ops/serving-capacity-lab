# Week 17 Day 6 — Serving Benchmark

适用于 vLLM 0.10.2，Qwen/Qwen3-0.6B，RTX 4090，TP=1。

## 1. 启动服务

在 `/data/vllm_learning` 激活 `.venv`，将 `experiments/` 中两个脚本复制到相同相对位置：

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --dtype auto --max-model-len 1024 \
  --max-num-batched-tokens 512 --max-num-seqs 32 \
  --gpu-memory-utilization 0.7 --no-enable-prefix-caching \
  --generation-config vllm \
  --compilation-config '{"level":3,"cudagraph_mode":1}'
```

不要打开 Day 3/4 的 Python Trace，避免影响性能。

## 2. 测试负载

另一个终端：

```bash
source .venv/bin/activate
mkdir -p results
# 可选：先用 12 个请求预热，不计入正式结果
DAY6_NUM_PROMPTS=12 bash experiments/day6_suite.sh concurrency 4 balanced
# A: 无限速发起、在客户端限制同时在途数（不是严格闭环）
for c in 1 2 4 8 16 32; do
  bash experiments/day6_suite.sh concurrency "$c" short_decode
done
# B: Poisson 到达，无 max-concurrency 限流，检查排队和 SLO
for rate in 2 5 10 20; do
  bash experiments/day6_suite.sh rate "$rate" short_decode
done
# C: 两类负载的比较
bash experiments/day6_suite.sh concurrency 8 long_prefill
bash experiments/day6_suite.sh concurrency 8 balanced
```

默认 128 请求/组；快速小样本验证设置 `DAY6_NUM_PROMPTS=24`，但其 p99 不应作正式结论。

## 3. 结果

- `results/*.json`：官方 Benchmark 统计和每请求明细（`--save-detailed`）。
- `results/*.log`：官方 CLI 输出，含指标与 Goodput。
- `results/*_metrics.jsonl`：采样 `/metrics`，记录请求排队、Running、GPU KV Usage、累计抢占次数。可取峰值；计数器差量需要前后相减。
- `num_requests_waiting > 0` 表示观察到服务端排队；`num_preemptions_total` 增长表示发生过抢占。
- 当前 SLO 示例：TTFT <= 200ms 且 TPOT <= 15ms；不是生产级标准。

## 4. 注意

1. `--request-rate inf --max-concurrency C` 是突发/客户端限并发负载，不是严格的闭环用户思考时间模型。
2. 有限 `--request-rate` + `--burstiness 1` 为 Poisson 到达；本脚本 **没有** 再设置 `--max-concurrency`，防止客户端限流悄悄改变到达率。
3. 保持同一模型、同一 server 配置、相同随机种子和固定 token 长度。Prefix Cache 关闭是为了减少缓存干扰。
4. 结果代表客户端口径；Prometheus 直方图是服务端口径，并受桶宽影响。不能混成同一指标。
5. `/metrics` 每 0.5s 采样可能漏掉很短的峰值，负载较重时建议 0.2s 或结合 Prometheus。
6. 单轮 128 请求的 p99 很不稳定；正式 SLO 评估应增加样本、持续时间和独立重复。
7. 不要把原始 Benchmark 负载推到公网服务；本实验限 `127.0.0.1`。

## 5. 五道理论题

1. TTFT、ITL、TPOT、E2E 的定义是什么？为什么 `content_chunks` 不等于生成 token 数？
2. 对比 `--request-rate inf --max-concurrency 8` 和 `--request-rate 8`：两者负载模型、背压位置、延迟含义有何不同？
3. 随着到达率提高，服务端 Running、Waiting、KV Usage、Preemption、TTFT 与 TPOT 如何变化？哪些现象不能直接归因于 KV 不足？
4. 若 SLO 是 TTFT <= 200ms、TPOT <= 15ms，如何计算 Goodput？为什么最大 Throughput 与最大 Goodput 可能出现在不同负载点？
5. 长 Prefill/短 Decode 和短 Prefill/长 Decode 对 TTFT、TPOT、GPU/Host 瓶颈分别有什么影响？设计一套排除 Prefix Cache、Graph Warmup、过载客户端三类混杂因素的对照实验。
