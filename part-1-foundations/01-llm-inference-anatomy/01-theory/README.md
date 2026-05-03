# 01 — Theory

Conceptual notes for Topic 01.

| File | What it covers |
|---|---|
| [`01-overview.md`](./01-overview.md) | End-to-end request lifecycle — what happens from prompt to streamed response, and how serving pieces fit together. |
| [`02-compute-vs-memory.md`](./02-compute-vs-memory.md) | Where inference is compute-bound vs memory-bound, and what that means for GPU utilization. |
| [`03-prefill-and-decode.md`](./03-prefill-and-decode.md) | Prefill vs decode — different bottlenecks, batching behavior, and why this split drives most serving optimizations. |
| [`04-kv-cache-explained.md`](./04-kv-cache-explained.md) | KV cache mechanics, memory footprint, and why the cache dominates decode cost and capacity planning. |
