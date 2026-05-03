# 01 — The Anatomy of LLM Inference

End-to-end view of what happens when a prompt is served: tokenization, GPU work, KV cache, scheduling, streaming, and where prefill vs decode differ. This topic builds the mental model used in later parts of the handbook.

## Where to start

Read theory notes in order, beginning with [`01-theory/01-overview.md`](./01-theory/README.md).

## Folder map

| Directory | Purpose |
|-----------|---------|
| [`01-theory/`](./01-theory/README.md) | How and why inference behaves the way it does (numbered notes + reading order). |
| [`02-papers/`](./02-papers/README.md) | Paper summaries and reading notes. |
| [`03-implementations/`](./03-implementations/README.md) | How real engines implement these ideas (code reading). |
| [`04-experiments/`](./04-experiments/README.md) | Runnable experiments and measurements. |
| [`05-benchmarks/`](./05-benchmarks/README.md) | Raw data, methodology, reproducibility. |
| [`06-decision-guides/`](./06-decision-guides/README.md) | Trade-offs, when to use what, failure modes. |
| [`07-references/`](./07-references/README.md) | Further reading and links. |

Figures for this topic sit under [`../assets/01-llm-inference-anatomy/`](../assets/01-llm-inference-anatomy/README.md).
