# LLM Serving Handbook

**Theory · Experiments · Benchmarks · Decision Guides**

---

A hands-on, opinionated guide to serving large language models in production.

This repo is where I document everything I learn about LLM serving infrastructure — the theory behind each technique, paper reading notes, code-level analysis of real engines, and most importantly, actual experiments with real measurements on real hardware.

Not an awesome list. Not a link dump. Every section is written from scratch, backed by runnable code and reproducible benchmarks where applicable.

**This is a work in progress.** Topics are added as I study them, one at a time, each to the level of depth I'd want if I were learning it from scratch. If a folder doesn't exist yet, I haven't gotten to it.

---

## Structure

Each topic follows a consistent format:

```
topic-name/
├── README.md              — summary and key takeaways
├── 01-theory/             — how and why it works
├── 02-papers/             — paper summaries and reading notes
├── 03-implementations/    — how real engines do it (code reading)
├── 04-experiments/        — runnable benchmarks, real measurements
├── 05-benchmarks/         — raw data, methodology, reproducibility
├── 06-decision-guides/    — when to use, trade-offs, failure modes
└── 07-references/         — further reading and resources
```

**Part-level assets:** In a part folder (e.g. `part-1-foundations/`), diagrams and other binaries live under `assets/<topic-folder>/…`. Mirror numbered section and doc names under `01-theory/` (e.g. `01-theory/01-overview/` for `01-overview.md`). From `topic-name/01-theory/01-overview.md`, reference e.g. `../../assets/01-llm-inference-anatomy/01-theory/01-overview/figure.png`.

---

## Roadmap

### Part 1 — Foundations

| # | Topic | Status |
|---|-------|--------|
| 01 | The Anatomy of LLM Inference | ✅ Done |
| 02 | GPU Architecture for LLM Engineers | 🚧 In Progress |
| 03 | Memory-Bound vs Compute-Bound | 📋 Planned |
| 04 | Numerical Precision | 📋 Planned |
| 05 | Attention Mechanisms for Serving | 📋 Planned |
| 06 | Tokenization and Its Impact on Throughput | 📋 Planned |

### Part 2 — Single-GPU Optimization

| # | Topic | Status |
|---|-------|--------|
| 07 | KV Cache Management (PagedAttention) | 📋 Planned |
| 08 | Continuous Batching | 📋 Planned |
| 09 | Prefix Caching | 📋 Planned |
| 10 | Chunked Prefill | 📋 Planned |
| 11 | Flash Attention | 📋 Planned |
| 12 | Weight Quantization (GPTQ, AWQ) | 📋 Planned |
| 13 | KV Cache Quantization | 📋 Planned |
| 14 | Activation Quantization | 📋 Planned |
| 15 | Speculative Decoding | 📋 Planned |
| 16 | Structured Output Optimization | 📋 Planned |
| 17 | CUDA Graphs | 📋 Planned |
| 18 | torch.compile for Inference | 📋 Planned |

### Part 3 — Distributed Inference

| # | Topic | Status |
|---|-------|--------|
| 19 | Tensor Parallelism | 📋 Planned |
| 20 | Pipeline Parallelism | 📋 Planned |
| 21 | Expert Parallelism (MoE) | 📋 Planned |
| 22 | Sequence Parallelism | 📋 Planned |
| 23 | Context Parallelism | 📋 Planned |
| 24 | Disaggregated Serving | 📋 Planned |
| 25 | Collective Communications (NCCL) | 📋 Planned |
| 26 | Multi-Node Deployment | 📋 Planned |

### Part 4 — Inference Engines

| # | Topic | Status |
|---|-------|--------|
| 27 | vLLM Internals | 📋 Planned |
| 28 | SGLang Internals | 📋 Planned |
| 29 | TensorRT-LLM Internals | 📋 Planned |
| 30 | llama.cpp Internals | 📋 Planned |
| 31 | MLC-LLM | 📋 Planned |
| 32 | LMDeploy | 📋 Planned |
| 33 | DeepSpeed-MII | 📋 Planned |
| 34 | Engine Comparison Matrix | 📋 Planned |
| 35 | When to Use Which Engine | 📋 Planned |

### Part 5 — Serving Infrastructure

| # | Topic | Status |
|---|-------|--------|
| 36 | Triton Inference Server | 📋 Planned |
| 37 | Ray Serve | 📋 Planned |
| 38 | KServe | 📋 Planned |
| 39 | BentoML | 📋 Planned |
| 40 | Serverless GPU Platforms | 📋 Planned |
| 41 | Autoscaling Strategies | 📋 Planned |
| 42 | Cold Start Mitigation | 📋 Planned |
| 43 | Model Loading Strategies | 📋 Planned |
| 44 | Multi-Model / Multi-Tenant Serving | 📋 Planned |

### Part 6 — Routing & Gateway

| # | Topic | Status |
|---|-------|--------|
| 45 | LLM Gateway Design | 📋 Planned |
| 46 | Semantic Caching | 📋 Planned |
| 47 | Prompt Caching | 📋 Planned |
| 48 | Model Routing | 📋 Planned |
| 49 | Token-Aware Rate Limiting | 📋 Planned |
| 50 | Cost Tracking | 📋 Planned |
| 51 | PII Masking & Guardrails | 📋 Planned |
| 52 | Prompt Injection Defense | 📋 Planned |
| 53 | A/B Testing for LLMs | 📋 Planned |

### Part 7 — Multi-LoRA & Adapters

| # | Topic | Status |
|---|-------|--------|
| 54 | LoRA Fundamentals for Serving | 📋 Planned |
| 55 | Multi-LoRA Serving (S-LoRA, Punica) | 📋 Planned |
| 56 | LoRA Hot-Swapping | 📋 Planned |
| 57 | Adapter Routing | 📋 Planned |

### Part 8 — Observability

| # | Topic | Status |
|---|-------|--------|
| 58 | Metrics That Matter (TTFT, TPOT, ITL) | 📋 Planned |
| 59 | Prometheus + Grafana Stack | 📋 Planned |
| 60 | OpenTelemetry GenAI Conventions | 📋 Planned |
| 61 | Distributed Tracing | 📋 Planned |
| 62 | Production Evaluation | 📋 Planned |
| 63 | Debugging Production Issues | 📋 Planned |

### Part 9 — Evaluation & Quality

| # | Topic | Status |
|---|-------|--------|
| 64 | Benchmarking Methodology | 📋 Planned |
| 65 | Quality-Speed Pareto Analysis | 📋 Planned |
| 66 | Quantization Quality Loss | 📋 Planned |
| 67 | Long-Context Evaluation | 📋 Planned |
| 68 | Regression Detection | 📋 Planned |

### Part 10 — Cost & Economics

| # | Topic | Status |
|---|-------|--------|
| 69 | Cost Modeling ($/1M tokens) | 📋 Planned |
| 70 | Self-Host vs API (TCO) | 📋 Planned |
| 71 | GPU Economics | 📋 Planned |
| 72 | Spot vs On-Demand | 📋 Planned |
| 73 | Cost Optimization Playbook | 📋 Planned |

### Part 11 — Hardware

| # | Topic | Status |
|---|-------|--------|
| 74 | NVIDIA Datacenter GPUs (A100 → B200) | 📋 Planned |
| 75 | AMD Instinct | 📋 Planned |
| 76 | Google TPU | 📋 Planned |
| 77 | AWS Inferentia & Trainium | 📋 Planned |
| 78 | Specialized Accelerators (Groq, Cerebras) | 📋 Planned |
| 79 | Edge & Mobile Inference | 📋 Planned |

### Part 12 — Emerging Topics

| # | Topic | Status |
|---|-------|--------|
| 80 | Multimodal Serving | 📋 Planned |
| 81 | Agentic Workloads | 📋 Planned |
| 82 | Reasoning Models & Test-Time Compute | 📋 Planned |
| 83 | Long Context (1M+ tokens) | 📋 Planned |
| 84 | On-Device LLMs | 📋 Planned |
| 85 | Energy & Sustainability | 📋 Planned |