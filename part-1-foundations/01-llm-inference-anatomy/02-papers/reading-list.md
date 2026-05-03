# Reading List — The Anatomy of LLM Inference

This file collects the papers most relevant to understanding LLM inference from a systems perspective. The goal is not exhaustive coverage — it is to give you the minimum set of papers that build the mental model behind Part 1 and motivate the optimizations in Part 2.

Papers are grouped by theme. Within each group, they are ordered by reading priority.

**Priority guide:**
- 🔴 Must-read — foundational, directly shapes how you think about serving
- 🟡 Useful — deepens understanding, worth reading after the must-reads
- 🟢 Optional — specialized or historical context, read when the topic comes up

---

## 1. Transformer Architecture

These papers establish what the model actually is. You cannot reason about serving bottlenecks without understanding the architecture you are serving.

---

### Attention Is All You Need
**Authors:** Vaswani et al. (Google Brain, 2017)  
**Link:** https://arxiv.org/abs/1706.03762  
**Priority:** 🔴 Must-read

**What it claims:** The Transformer architecture, built entirely on self-attention and feed-forward layers without recurrence or convolution, outperforms prior sequence models on translation tasks.

**Why it matters for serving:** Every LLM you will ever serve is built on this architecture. The paper introduces the Q, K, V attention formulation, multi-head attention, and the encoder-decoder structure. Understanding what the model is doing — matrix projections, attention score computation, value aggregation — is the prerequisite for understanding why prefill is compute-bound, why decode reads the KV cache, and why attention cost grows quadratically with sequence length.

**What the paper does not cover:** KV caching, autoregressive generation at serving scale, memory bandwidth constraints, batching strategies, or any of the serving problems this handbook addresses. It is a training paper. The serving implications have to be derived.

**Key serving takeaways:**
- Attention is `O(n²)` in sequence length — directly motivates FlashAttention and prefix caching
- Multi-head attention projects Q, K, V separately — the K and V projections are what the KV cache stores
- The feed-forward sublayer is 2× the hidden size — this dominates FLOPs in large models

**Connected theory files:** `01-overview.md`, `03-prefill-and-decode.md`, `04-kv-cache-explained.md`

---

### Language Models are Unsupervised Multitask Learners (GPT-2)
**Authors:** Radford et al. (OpenAI, 2019)  
**Link:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf  
**Priority:** 🔴 Must-read

**What it claims:** A decoder-only Transformer trained on diverse web text learns to perform many NLP tasks without task-specific fine-tuning.

**Why it matters for serving:** GPT-2 established the decoder-only architecture that almost all modern LLMs (GPT-3/4, Llama, Mistral, Gemma, Qwen, etc.) follow. The decoder-only design means there is no separate encoder — every token attends to all previous tokens via causal (masked) self-attention. This is why inference is autoregressive: generation must happen one token at a time, each attending to everything before it. Understanding this architecture makes the prefill/decode distinction intuitive rather than arbitrary.

**Key serving takeaways:**
- Causal (left-to-right) attention is why generation is sequential
- No encoder means no separate "encode the input" step — the prompt is processed alongside generation
- The decoder-only design is why the KV cache grows during generation rather than being fixed

**Connected theory files:** `01-overview.md`, `03-prefill-and-decode.md`

---

### LLaMA: Open and Efficient Foundation Language Models
**Authors:** Touvron et al. (Meta AI, 2023)  
**Link:** https://arxiv.org/abs/2302.13971  
**Priority:** 🟡 Useful

**What it claims:** A family of open foundation models trained on publicly available data that match or outperform proprietary models at smaller parameter counts.

**Why it matters for serving:** Llama and its successors (Llama 2, Llama 3, Llama 3.1) are the reference models used throughout this handbook for all numerical examples. The architecture introduces RoPE positional embeddings, RMSNorm instead of LayerNorm, and SwiGLU activations — all of which are now standard in open LLMs. Llama 3.1 specifically introduces grouped-query attention (GQA), which directly affects KV cache size calculations.

**Key serving takeaways:**
- RoPE allows extending context length beyond training — relevant for long-context serving
- GQA reduces KV heads from 32 to 8 in the 8B model — 4× smaller KV cache compared to full MHA
- Architecture config (hidden size 4096, 32 layers, 8 KV heads, head dim 128) used in all numerical examples in theory files

**Connected theory files:** `03-prefill-and-decode.md`, `04-kv-cache-explained.md`

---

## 2. Scaling Laws and Model Size

These papers explain why models are large and why they keep growing. Serving cost is a direct consequence of scale.

---

### Scaling Laws for Neural Language Models
**Authors:** Kaplan et al. (OpenAI, 2020)  
**Link:** https://arxiv.org/abs/2001.08361  
**Priority:** 🔴 Must-read

**What it claims:** Language model performance (loss) follows smooth power laws with model size, dataset size, and compute budget. Each factor contributes independently, and there are optimal allocations of a fixed compute budget.

**Why it matters for serving:** This paper explains why the models you are serving are as large as they are. The empirical finding that loss scales predictably with parameter count created the incentive to train progressively larger models. For serving, larger models mean more GPU memory for weights, more compute per token, larger KV caches, and higher serving cost per request. Understanding scaling laws helps you reason about the trade-off between model quality and serving infrastructure cost — a tension that runs through every topic in this handbook.

**Key serving takeaways:**
- Bigger models are more sample-efficient but more expensive to serve
- The parameter/compute/data trade-off at training time determines what you get to serve at inference time
- Motivates the entire quantization and distillation ecosystem — large models are expensive, so we approximate them

**Connected theory files:** `01-overview.md`, `02-compute-vs-memory.md`

---

### Training Compute-Optimal Large Language Models (Chinchilla)
**Authors:** Hoffmann et al. (DeepMind, 2022)  
**Link:** https://arxiv.org/abs/2203.15556  
**Priority:** 🟡 Useful

**What it claims:** Prior large models were significantly undertrained. Given a fixed compute budget, the optimal strategy is to train a smaller model on more data — roughly equal scaling of parameters and tokens.

**Why it matters for serving:** Chinchilla shifted the industry toward smaller, better-trained models. A 70B model trained on 2 trillion tokens (Llama 2 style) serves cheaper than a 175B undertrained model while achieving comparable quality. This paper is why you now see emphasis on training data quality and compute-optimal recipes — and why serving a well-trained 7B model is a legitimate production strategy. It reframes the serving cost problem: quality is not just a function of model size.

**Key serving takeaways:**
- Chinchilla-optimal models tend to be smaller and faster to serve than their predecessors at equivalent quality
- Motivates the 7B-13B serving tier that dominates production deployments
- Helps reason about when to serve a smaller model vs a larger one

**Connected theory files:** `01-overview.md`

---

## 3. Attention Efficiency and Memory

These papers address the computational and memory cost of attention — the most expensive operation in both prefill and decode.

---

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
**Authors:** Dao et al. (Stanford, 2022)  
**Link:** https://arxiv.org/abs/2205.14135  
**Priority:** 🔴 Must-read

**What it claims:** Standard attention implementations are memory-bandwidth-bound due to repeated reads and writes to HBM. By tiling the computation and fusing operations, attention can be computed with far fewer HBM accesses while producing identical results.

**Why it matters for serving:** FlashAttention is the foundational paper for understanding why attention is memory-bandwidth-bound and not compute-bound, even during prefill. It formalizes the IO-complexity argument for GPU kernels — the idea that the number of bytes moved between HBM and SRAM determines performance, not just FLOPs. This is the same argument that underlies `02-compute-vs-memory.md`. FlashAttention is now standard in virtually every serious inference engine (vLLM, SGLang, TensorRT-LLM). Reading the paper gives you the mental model for kernel-level optimization.

**Key serving takeaways:**
- Standard attention requires O(n²) HBM reads/writes — FlashAttention reduces this to O(n)
- The IO-complexity framing applies beyond attention — it is a general lens for GPU kernel optimization
- Prefill benefits from faster attention for long prompts; decode benefits from reduced KV cache read overhead
- FlashAttention 2 and 3 extend this with better work partitioning and async memory pipelining

**Connected theory files:** `02-compute-vs-memory.md`, `04-kv-cache-explained.md`

---

### GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
**Authors:** Ainslie et al. (Google, 2023)  
**Link:** https://arxiv.org/abs/2305.13245  
**Priority:** 🟡 Useful

**What it claims:** Grouped-query attention (GQA) interpolates between multi-head attention (MHA) and multi-query attention (MQA) by sharing key and value heads across groups of query heads. Models converted from MHA checkpoints using GQA retain most quality while significantly reducing KV cache size.

**Why it matters for serving:** GQA is why Llama 3 has 8 KV heads instead of 32, reducing KV cache memory by 4× at no meaningful quality cost. This is one of the most impactful architectural changes for serving in recent years. The paper directly motivates the GQA/MQA row in the architecture comparison table in `04-kv-cache-explained.md`.

**Key serving takeaways:**
- MQA (1 KV head) is most memory-efficient but shows quality degradation on some tasks
- GQA (H/G KV heads, where G is the group size) provides a better quality-memory trade-off
- Directly reduces KV cache memory by a factor of `H / H_kv`
- Most modern open LLMs (Llama 3, Mistral, Qwen, Gemma 2) use GQA

**Connected theory files:** `04-kv-cache-explained.md`

---

## 4. Inference Serving Systems

These papers address LLM serving as a systems problem — scheduling, memory management, and throughput optimization.

---

### Efficiently Scaling Transformer Inference
**Authors:** Pope et al. (Google, 2022)  
**Link:** https://arxiv.org/abs/2211.05102  
**Priority:** 🔴 Must-read

**What it claims:** A systematic analysis of the compute and memory requirements of Transformer inference, with partitioning strategies for serving large models across multiple accelerators efficiently.

**Why it matters for serving:** This is the closest thing to a "serving systems" companion to the Transformer paper. Pope et al. formalize the prefill vs decode distinction, the memory-bandwidth-bound nature of decode, the roofline model applied to LLM inference, and the impact of model parallelism strategies. The paper's arithmetic intensity analysis is the basis for the calculations in `02-compute-vs-memory.md` and `03-prefill-and-decode.md`. It is the paper you would cite if someone asked "where does the '2× parameters = FLOPs per token' rule come from?"

**Key serving takeaways:**
- Formalizes the FLOPs-per-token ≈ 2 × parameters approximation
- Analyzes how batch size affects compute utilization during decode
- Shows that decode is memory-bandwidth-bound and prefill is compute-bound across model sizes
- Discusses tensor parallelism and pipeline parallelism trade-offs (preview for Part 3)

**Connected theory files:** `02-compute-vs-memory.md`, `03-prefill-and-decode.md`

---

### Orca: A Distributed Serving System for Transformer-Based Generative Models
**Authors:** Yu et al. (Seoul National University, 2022)  
**Link:** https://www.usenix.org/conference/osdi22/presentation/yu  
**Priority:** 🔴 Must-read

**What it claims:** Static batching wastes GPU capacity because requests finish at different times, leaving idle slots. Continuous batching — scheduling at the iteration level rather than the request level — dramatically improves throughput by adding new requests as soon as slots free up.

**Why it matters for serving:** Orca introduced continuous batching, which is now standard in every serious LLM serving engine. The paper is the direct predecessor of vLLM's scheduler. Understanding why static batching is inefficient and how iteration-level scheduling works is essential for understanding throughput optimization. This paper is covered in depth in Part 2 (Topic 08).

**Key serving takeaways:**
- Static batching: batch is fixed at request start, slot wasted when request finishes early
- Continuous batching: new requests join the batch at token boundaries without stopping other requests
- Iteration-level scheduling requires the engine to be able to add/remove sequences mid-flight
- Throughput improvements of 10-36× over static batching in their evaluation

**Connected theory files:** `01-overview.md`, `03-prefill-and-decode.md`

---

### Efficient Memory Management for Large Language Model Serving with PagedAttention
**Authors:** Kwon et al. (UC Berkeley, 2023)  
**Link:** https://arxiv.org/abs/2309.06180  
**Priority:** 🔴 Must-read

**What it claims:** KV cache memory fragmentation and over-reservation waste 60-80% of GPU memory in existing serving systems. PagedAttention manages KV cache like virtual memory pages, enabling near-zero waste and allowing copy-on-write sharing of cached KV blocks across requests.

**Why it matters for serving:** PagedAttention is the foundational contribution of vLLM, now the most widely used open-source LLM serving engine. The paper motivates and solves the memory fragmentation problem described in `04-kv-cache-explained.md`. Reading it makes the connection between operating system virtual memory management and LLM serving concrete. This paper is covered in depth in Part 2 (Topic 07).

**Key serving takeaways:**
- Pre-PagedAttention engines waste 60-80% of KV memory due to fragmentation and over-reservation
- Fixed-size KV blocks (pages) decouple logical sequence layout from physical memory layout
- Copy-on-write enables efficient prefix sharing without duplicating KV tensors
- Combined with continuous batching, achieves 2-4× throughput improvement over previous systems

**Connected theory files:** `04-kv-cache-explained.md`

---

## 5. Further Reading

Papers worth reading as you move into later parts of the handbook. Not required for Part 1.

| Paper | Authors | Year | Why it is relevant |
|---|---|---|---|
| FlashAttention-2 | Dao (Stanford) | 2023 | Better work partitioning, 2× faster than FlashAttention 1 |
| FlashAttention-3 | Shah et al. | 2024 | Asynchronous pipelining, FP8 support, H100-specific optimizations |
| S-LoRA | Sheng et al. | 2023 | Multi-LoRA serving at scale, relevant for Part 7 |
| Splitwise | Patel et al. | 2023 | Disaggregated prefill/decode serving, relevant for Part 3 |
| DejaVu | Liu et al. | 2023 | Contextual sparsity in LLMs, relevant for Part 2 |
| Sarathi-Serve | Agrawal et al. | 2024 | Chunked prefill to reduce TTFT variance, relevant for Part 2 |
| Medusa | Cai et al. | 2024 | Speculative decoding variant, relevant for Part 2 |
| Mooncake | Qin et al. | 2024 | KV cache disaggregation across heterogeneous memory, relevant for Part 3 |
| vAttention | Prabhu et al. | 2024 | System-level KV memory management alternative to PagedAttention |

---

## Reading order recommendation

If you are reading through this topic for the first time:

1. **Vaswani et al. 2017** — understand what you are serving
2. **Pope et al. 2022** — understand the compute and memory constraints formally
3. **Dao et al. 2022 (FlashAttention)** — understand why attention is memory-bound
4. **Kwon et al. 2023 (PagedAttention / vLLM)** — understand the memory management problem
5. **Yu et al. 2022 (Orca)** — understand continuous batching
6. **Radford et al. 2019 (GPT-2)** — fill in the decoder-only architecture intuition
7. **Kaplan et al. 2020 (Scaling Laws)** — understand why models are this large

Papers 1-5 give you the foundation for 90% of the serving discussions in this handbook.