# 04 — Experiments

Runnable experiments for Topic 01.

| File | What it covers |
|---|---|
| [`exp-01-prefill-vs-decode-profiling/`](./exp-01-prefill-vs-decode-profiling/README.md) | Prefill vs decode profiling — TTFT and per-token decode latency (HF + vLLM), and whether GPU utilization matches compute- vs memory-bound expectations. |
| [`exp-02-kv-cache-memory-measurement/`](./exp-02-kv-cache-memory-measurement/README.md) | KV cache memory — compare measured GPU memory growth during generation against the closed-form KV size formula and find the crossover with weight memory. |
