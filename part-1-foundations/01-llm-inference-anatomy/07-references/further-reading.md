# Further Reading — The Anatomy of LLM Inference

Curated external resources that complement the theory notes in this topic. Each entry has a one-line annotation explaining why it is worth reading.

---

## Blog posts and articles

**Transformer Inference Arithmetic — Kipply's Blog (2022)**  
https://kipp.ly/transformer-inference-arithmetic/  
The best single article on the math behind LLM inference. Covers FLOPs, memory bandwidth, KV cache size, and batch size effects with concrete numbers. Read this alongside `03-prefill-and-decode.md`.

**The Illustrated Transformer — Jay Alammar (2018)**  
https://jalammar.github.io/illustrated-transformer/  
Visual walkthrough of the Transformer architecture. Best introduction to attention, Q/K/V, and multi-head attention if you find the original paper dense.

**LLM Inference Performance Engineering — Databricks Blog (2023)**  
https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices  
Practical breakdown of throughput vs latency trade-offs, batch size effects, and GPU memory management from a production perspective.

**Dissecting Batching Effects in GPT Inference — Anyscale Blog (2023)**  
https://www.anyscale.com/blog/continuous-batching-llm-inference  
Explains why static batching wastes GPU capacity and how continuous batching solves it. Good complement to the Orca paper in `02-papers/reading-list.md`.

**Making Deep Learning Go Brrrr From First Principles — Horace He (2022)**  
https://horace.io/brrr_intro.html  
First-principles explanation of compute-bound vs memory-bound workloads, roofline model, and kernel fusion. One of the best resources on GPU performance intuition. Read this alongside `02-compute-vs-memory.md`.

**The KV Cache: Memory Usage in Transformers — Hugging Face Blog (2023)**  
https://huggingface.co/blog/kv-cache  
Short, practical explanation of the KV cache with diagrams. Good entry point before reading `04-kv-cache-explained.md`.

---

## Talks and videos

**Andrej Karpathy — Intro to Large Language Models (2023)**  
https://www.youtube.com/watch?v=zjkBMFhNj_g  
1-hour talk covering LLM internals from first principles. The inference section (around 30 min in) is a good intuition builder for the prefill/decode distinction.

**Tim Dettmers — LLM Hardware Requirements (2023)**  
https://www.youtube.com/watch?v=y9PHWGOa8HA  
Practical breakdown of GPU memory requirements, quantization trade-offs, and hardware selection for LLM serving.

**vLLM Talk — OSDI 2023**  
https://www.youtube.com/watch?v=80bIUggRJf4  
The original PagedAttention presentation. Covers the memory fragmentation problem and how paged KV management solves it. Watch this after reading `04-kv-cache-explained.md`.

---

## Books

**AI Engineering — Chip Huyen (2025)**  
Chapter on inference optimization covers batching, quantization, and serving trade-offs from a practitioner's perspective. Good high-level complement to the more detailed topics in this handbook.

**Designing Machine Learning Systems — Chip Huyen (2022)**  
Chapter 7 (Model Deployment and Prediction Service) covers latency vs throughput trade-offs, batching strategies, and hardware selection for ML serving broadly.

---

## Interactive tools

**Transformer Explainer**  
https://poloclub.github.io/transformer-explainer/  
Interactive visualization of a GPT-2 style Transformer. Lets you trace how a token flows through attention layers. Useful for building intuition about what the model actually computes.

**LLM Visualization — Brendan Bycroft**  
https://bbycroft.net/llm  
Step-by-step 3D visualization of a small GPT model processing a prompt. Shows matrix multiplications, attention scores, and residual connections visually.

---

## Related topics in this handbook

Once the foundational anatomy is clear, these topics are the natural next steps:

| Topic | Why it follows from this one |
|---|---|
| Part 2, Topic 07 — PagedAttention | Solves the KV cache fragmentation problem from `04-kv-cache-explained.md` |
| Part 2, Topic 08 — Continuous Batching | Solves the static batching problem from `01-overview.md` |
| Part 2, Topic 09 — Prefix Caching | Reuses KV cache across requests with shared prefixes |
| Part 2, Topic 10 — Chunked Prefill | Solves prefill blocking decode from `03-prefill-and-decode.md` |
| Part 2, Topic 11 — FlashAttention | Reduces memory bandwidth in the attention kernel |
| Part 2, Topic 13 — KV Cache Quantization | Reduces KV cache memory footprint |