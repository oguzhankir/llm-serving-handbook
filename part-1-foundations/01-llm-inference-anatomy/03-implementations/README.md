# 03 — Implementations

Code reading notes for Topic 01.

| File | What it covers |
|---|---|
| [`hf-transformers-generate.md`](./hf-transformers-generate.md) | How `model.generate()` works — prefill, KV cache allocation, and why it's unsuitable for production serving. |
| [`vllm-overview.md`](./vllm-overview.md) | How vLLM structures the pipeline differently — scheduler, block pool, and how a request flows from arrival to completion. |