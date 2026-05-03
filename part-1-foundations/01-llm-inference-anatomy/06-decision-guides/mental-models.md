# Mental Models for LLM Inference

This file distills the four theory documents in Topic 01 into a set of reusable mental models. The goal is practical: when you encounter a new serving optimization, benchmark result, or production problem, these frameworks help you place it immediately.

---

## Mental Model 1 — Every request has two personalities

The single most important mental model in this handbook.

A request is not one workload. It is two fundamentally different workloads back to back.

```
[Prefill]  ──────────────────────────────►  [Decode loop]
Many tokens, parallel, compute-bound         One token/step, sequential, memory-bound
TTFT is the user-facing metric               TPOT / ITL is the user-facing metric
```

**How to use this:**

When you read about any optimization, ask first:

> Which half of this request does it target?

If the answer is unclear, the optimization is not well understood yet. Almost every technique in this handbook falls into one of three categories:

| Category | What it targets | Examples |
|---|---|---|
| Prefill optimization | Reduce TTFT, speed up the compute-bound phase | Prefix caching, chunked prefill, FlashAttention |
| Decode optimization | Improve TPOT, increase memory bandwidth utilization | Continuous batching, KV quantization, speculative decoding |
| Both | Usually memory management or scheduling | PagedAttention, GQA/MQA, weight quantization |

**Red flag:** If someone claims a technique "speeds up inference" without specifying which metric — TTFT, TPOT, or throughput — they are being imprecise. Ask which phase it targets.

---

## Mental Model 2 — The two ceilings

Every GPU workload runs into one of two limits:

```
Ceiling 1: Compute (FLOPs)        → you are compute-bound
Ceiling 2: Memory bandwidth        → you are memory-bound

Performance = min(what compute allows, what memory bandwidth allows)
```

The ratio that determines which ceiling you hit:

<p align="center">
  <strong>Arithmetic intensity = FLOPs performed / bytes moved</strong>
</p>

**LLM inference in practice:**

| Phase | Operation shape | Arithmetic intensity | Regime |
|---|---|---|---|
| Prefill (long prompt) | Matrix × Matrix | High (~N × 2 FLOPs/byte) | Compute-bound |
| Decode (single request) | Matrix × Vector | Very low (~1-2 FLOPs/byte) | Memory-bound |
| Decode (large batch) | Matrix × Matrix | Higher | Approaching compute-bound |

**How to use this:**

When evaluating a hardware or software change, ask:

> What is the bottleneck right now — compute or memory?

Then ask:

> Does this change address that bottleneck?

A GPU with 2× the FLOPs but identical HBM bandwidth will not improve single-request decode. A GPU with 2× the HBM bandwidth will. Getting these confused leads to expensive wrong decisions.

**The four-quadrant test:**

| Change | Bottleneck is compute | Bottleneck is memory |
|---|---|---|
| More FLOPs (better GPU) | ✅ Helps | ❌ Does not help |
| More HBM bandwidth | ❌ Does not help | ✅ Helps |
| Larger batch size | Moves toward compute ceiling | ✅ Helps (amortizes weight loads) |
| Quantization | Marginal (fewer FLOPs) | ✅ Helps (fewer bytes moved) |

---

## Mental Model 3 — KV cache is the dynamic memory constraint

Model weights are fixed once loaded. The KV cache is dynamic — it grows with every token generated for every active request.

```
GPU memory = model weights (fixed) + KV cache (dynamic) + activations + runtime
```

The KV cache growth formula:

<p align="center">
  <strong>KV bytes = 2 × L × H_kv × d_head × T × b</strong>
</p>

**The key scaling rules:**

- Double the sequence length → double the KV cache per request
- Double the concurrent requests → double total KV memory
- Half the KV heads (GQA vs MHA) → half the KV cache
- Half the precision (INT8 vs FP16) → half the KV cache

**How to use this:**

When a serving system runs out of memory or throughput drops at high concurrency, the first question is:

> Is this a model weight problem or a KV cache problem?

Model weight memory is fixed and predictable. KV cache memory is variable and load-dependent.

**Decision tree for memory pressure:**

```
GPU OOM or throughput cliff?
│
├── Happens at model load → model weights problem
│   └── Fix: quantize weights, use smaller model, add GPUs
│
└── Happens as requests accumulate → KV cache problem
    ├── At low concurrency with long contexts → per-request KV too large
    │   └── Fix: KV quantization, GQA/MQA, shorter max sequence length
    └── At high concurrency with normal contexts → total KV too large
        └── Fix: PagedAttention, KV quantization, reduce batch size
```

**The crossover point:**

For Llama 3.1 8B in BF16:
- Model weights: ~16 GB (fixed)
- KV cache per request at 4096 tokens: ~0.5 GB
- KV cache equals model weights at: ~32 concurrent requests

Beyond ~32 concurrent requests at 4096 tokens, the KV cache dominates GPU memory. Every architectural decision that reduces `H_kv` or `d_head` has compounding effects here.

---

## Mental Model 4 — Latency and throughput are different objectives

They are often in tension. Optimizing for one can hurt the other.

```
Latency objective:    minimize time for one request  (TTFT + M × TPOT)
Throughput objective: maximize tokens/sec across all requests
```

**The fundamental trade-off:**

| Strategy | Latency | Throughput | Why |
|---|---|---|---|
| Run requests immediately, no batching | ✅ Low | ❌ Low | GPU underutilized, memory bandwidth wasted |
| Wait to fill large batches | ❌ High (queuing) | ✅ High | Better GPU utilization, amortized weight loads |
| Continuous batching | ✅ Moderate | ✅ High | Best of both: no waiting, high utilization |

**How to use this:**

Before evaluating a serving system, ask:

> What is the primary objective — latency (user-facing product) or throughput (batch processing)?

Most production serving systems have SLOs for both. A typical constraint looks like:

- P95 TTFT < 500ms
- P95 TPOT < 50ms
- Maximize throughput within those constraints

Optimizations that improve throughput at the cost of tail latency may violate SLOs even if average latency looks acceptable. Always check P95 and P99, not just mean.

---

## Mental Model 5 — Batching is arithmetic intensity recovery

Decode is memory-bound because a single token generates too little arithmetic relative to the data loaded. Batching fixes this by multiplying the arithmetic without proportionally multiplying the data load.

```
Batch size 1:   load 16 GB weights → produce 1 token  → 16 GB/token
Batch size 32:  load 16 GB weights → produce 32 tokens → 0.5 GB/token
```

**The batching ceiling:**

Batching helps until one of these limits is hit:

1. **Compute ceiling** — arithmetic intensity becomes high enough that FLOPs become the bottleneck
2. **KV memory ceiling** — total KV cache for all requests exhausts GPU memory
3. **Latency constraint** — requests wait too long to join the batch

In practice for decoder-only serving, limit 2 (KV memory) is usually hit before limit 1 (compute). This is why PagedAttention, GQA, and KV quantization matter: they push limit 2 higher, allowing larger effective batch sizes.

**How to use this:**

If throughput is below expectations, ask:

> Why can I not batch more requests?

| Blocker | Symptom | Fix |
|---|---|---|
| KV memory exhaustion | OOM at N concurrent requests | KV quantization, GQA, PagedAttention |
| Memory fragmentation | Low utilization even with free memory | PagedAttention |
| Prefill blocking decode | High TPOT variance | Chunked prefill |
| Requests arrive too slowly | GPU idle between bursts | Request queuing, autoscaling |

---

## Mental Model 6 — Long context multiplies every cost

Increasing sequence length is not free in any dimension.

| What grows | Why |
|---|---|
| Prefill compute | More tokens to process → more FLOPs → higher TTFT |
| KV cache memory | More tokens cached → less concurrency on same hardware |
| Attention compute | O(n²) naive, O(n) IO with FlashAttention — still grows |
| Decode memory bandwidth | More KV cache to read per step → slower TPOT |

**The 10× context rule of thumb:**

Going from 4K to 40K context length does not make serving 10× harder in one dimension. It compounds:

- Prefill time: ~10× (linear in tokens)
- KV cache per request: ~10× (linear in tokens)
- Max concurrency on same hardware: ~10× fewer requests
- Per-step attention cost: grows with context regardless of FlashAttention

This is why long-context serving is a distinct infrastructure problem, not just "same thing but bigger."

---

## Quick reference — Optimization map

Use this table to place any optimization you encounter:

| Optimization | Targets | Improves | Mechanism |
|---|---|---|---|
| Continuous batching | Decode | Throughput | Fills GPU with more concurrent decode steps |
| PagedAttention | KV cache | Concurrency + throughput | Eliminates fragmentation, enables prefix sharing |
| Prefix caching | Prefill | TTFT | Reuses KV cache for shared prompt prefixes |
| Chunked prefill | Scheduling | TPOT variance | Prevents long prefills from blocking decode |
| FlashAttention | Attention (both phases) | Speed + memory | Reduces HBM reads/writes in attention kernel |
| KV cache quantization | KV cache + decode | Concurrency + TPOT | Fewer bytes per KV element → smaller cache, less bandwidth |
| Weight quantization | Decode | TPOT + memory | Fewer bytes to load per decode step |
| GQA / MQA | KV cache | Concurrency + TPOT | Fewer KV heads → smaller cache per request |
| Speculative decoding | Decode | TPOT (latency) | Draft tokens increase compute utilization |
| Tensor parallelism | Both | Throughput at scale | Splits model across GPUs, enables larger batches |
| Disaggregated serving | Scheduling | TTFT + TPOT | Separate hardware optimized per phase |

---

## Quick reference — Metrics map

| Metric | Full name | What it measures | Primarily affected by |
|---|---|---|---|
| TTFT | Time to first token | Responsiveness | Queue time + prefill time |
| TPOT | Time per output token | Generation speed | Decode time per step |
| ITL | Inter-token latency | Same as TPOT | Decode time per step |
| Throughput | Tokens/sec or requests/sec | System capacity | Batching + scheduling efficiency |
| MFU | Model FLOPs utilization | GPU efficiency | Batch size, arithmetic intensity |
| HBM utilization | Memory bandwidth used / peak | Memory efficiency | KV cache access patterns |

---

## The five questions to ask about any optimization

When you encounter a new technique — in a paper, a blog post, or a conference talk — run through this checklist:

**1. Which phase does it target?**
Prefill, decode, or both? If a paper claims both without explanation, be skeptical.

**2. What bottleneck does it address?**
Compute, memory bandwidth, memory capacity, or scheduling? An optimization cannot address a bottleneck that does not exist in your workload.

**3. Which metric improves?**
TTFT, TPOT, throughput, or concurrency? A technique that improves throughput by 2× may have zero effect on P95 TTFT.

**4. What are the trade-offs?**
Quality degradation? Implementation complexity? Compatibility restrictions? Every technique has a cost column that papers tend to underreport.

**5. At what scale does it matter?**
Some optimizations only matter at high batch sizes. Some only matter at long context. Some only matter on specific hardware. Make sure the regime in the paper matches your production regime.

---

## Summary

The mental models on this page reduce to a single framing:

> LLM serving is a resource allocation problem with two bottlenecks (compute and memory), two phases (prefill and decode), two objectives (latency and throughput), and one central dynamic constraint (KV cache memory) that links all of them.

Every optimization in this handbook pushes against one or more of these limits without catastrophically worsening the others. Once that framing is internalized, no serving technique should feel arbitrary.