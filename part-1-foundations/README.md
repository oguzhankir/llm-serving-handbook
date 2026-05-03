# Part 1 — Foundations

Core concepts you need before optimizing or operating LLM serving: how inference works end to end, GPU behavior, memory versus compute, precision, attention, and tokenization.

## Layout

- **Topics** live in numbered folders (`01-…`, `02-…`, …). Each topic mirrors the [handbook structure](../README.md#structure): theory, papers, implementations, experiments, benchmarks, decision guides, and references.
- **Assets** for this part live under [`assets/`](./assets/README.md) (diagrams and figures), grouped by topic folder name so media stays easy to find as more topics are added.

## Topics in this part

| Folder | Topic |
|--------|--------|
| [`01-llm-inference-anatomy/`](./01-llm-inference-anatomy/README.md) | The anatomy of LLM inference (lifecycle, prefill/decode, KV cache, compute vs memory) |

Additional foundation topics from the [main roadmap](../README.md#part-1--foundations) will get their own folders here as they are written.
