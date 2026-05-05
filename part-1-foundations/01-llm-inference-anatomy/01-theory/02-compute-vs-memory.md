# Compute vs Memory: The Real Bottleneck in LLM Inference

> 📖 Recommended order: You should read this after `01-overview.md`.

This chapter answers one of the most important hardware questions in LLM serving:

> Why is a GPU — one of the most powerful compute devices ever built — often sitting mostly idle during inference?

The short answer is that LLM inference, especially during decode, is not limited by how fast the GPU can compute. It is limited by how fast data can move between memory and the compute units. Understanding this distinction is the foundation for understanding why almost every serving optimization exists.

---

## 1. What a GPU actually does

A GPU is not one thing. It is two distinct subsystems that must work together:

| Subsystem | Role | Example capacity (A100 80GB) |
|---|---|---|
| Compute units (SMs) | Perform arithmetic (multiply, add, activate) | ~312 TFLOPS (BF16) |
| High-bandwidth memory (HBM) | Store and retrieve data (weights, activations, KV cache) | ~2 TB/s bandwidth |

These two subsystems operate in parallel, but they are not equally fast relative to the work that needs to be done.

The critical asymmetry:

> The GPU can do arithmetic roughly 150× faster than it can move data from memory.

Specifically, for an A100 80GB:

- Compute: 312 TFLOPS = 312 × 10¹² FLOPs per second
- Memory bandwidth: 2 TB/s = 2 × 10¹² bytes per second

If a kernel performs 312 FLOPs for every 2 bytes it moves, both subsystems are perfectly balanced — neither is waiting on the other. But most LLM inference kernels do not achieve this ratio.

---

## 2. The two regimes: compute-bound and memory-bound

Every GPU workload falls into one of two regimes.

### Compute-bound

The GPU's arithmetic units are the bottleneck. The memory system can feed data fast enough, but the compute units cannot process it all immediately.

Characteristics:
- Compute units near 100% utilization
- Memory bandwidth has headroom
- Performance is limited by FLOPs

This is where GPUs shine. Large matrix multiplications during training or long-prompt prefill are often compute-bound.

### Memory-bound

The memory system is the bottleneck. Data cannot arrive at the compute units fast enough, so compute units sit idle waiting for their next operands.

Characteristics:
- Compute units underutilized (sometimes <10%)
- Memory bandwidth near saturation
- Performance is limited by bytes/second, not FLOPs

This is the regime that dominates LLM decode. The GPU has enormous arithmetic capability that goes unused because the memory cannot feed it fast enough.

The fundamental model:

<p align="center">
  <strong>Performance = min(compute throughput limit, memory bandwidth limit)</strong>
</p>

A workload runs at the slower of the two. If memory is the bottleneck, adding more FLOPs accomplishes nothing.

---

## 3. Arithmetic intensity: the ratio that determines everything

The quantity that tells you which regime you are in is called **arithmetic intensity**.

<p align="center">
  <strong>Arithmetic intensity = FLOPs performed / bytes moved from memory</strong>
</p>

The unit is FLOPs per byte, sometimes written as FLOP/B.

For a given GPU, there is a critical threshold called the **ridge point**:

<p align="center">
  <strong>Ridge point = peak compute (FLOP/s) / peak memory bandwidth (byte/s)</strong>
</p>

For an A100 80GB in BF16:

<p align="center">
  <strong>Ridge point = 312 × 10¹² / 2 × 10¹² ≈ 156 FLOP/B</strong>
</p>

The rule:

| Arithmetic intensity of a kernel | Regime |
|---|---|
| Above 156 FLOP/B | Compute-bound |
| Below 156 FLOP/B | Memory-bound |
| Near 156 FLOP/B | Balanced (roofline peak) |

Any kernel with arithmetic intensity below 156 FLOP/B on an A100 cannot be compute-bound — it will always be memory-bandwidth-limited regardless of how many FLOPs it is asked to do.

---

## 4. The roofline model

The roofline model makes this visual. Performance is bounded from above by two lines:

```
Performance (FLOP/s)
        │
        │         ┌─────────────────────────── Compute ceiling
        │        /
        │       /
        │      /  ← memory bandwidth ceiling (slope = bandwidth)
        │     /
        │    /
        └───────────────────────────────────────
                  Ridge point          Arithmetic intensity (FLOP/B)
```

Below the ridge point, the performance ceiling is set by memory bandwidth: doubling arithmetic intensity doubles performance. Above the ridge point, the performance ceiling is set by compute: more intensity no longer helps.

**For LLM inference:**

- Prefill with long prompts: arithmetic intensity is high → lives to the right of the ridge point → compute-bound
- Single-request decode: arithmetic intensity is very low → lives far to the left → deeply memory-bound
- Large-batch decode: moves rightward as more tokens share the same weight load → eventually approaches compute-bound

---

## 5. Why arithmetic intensity is low during decode

At each decode step, the model processes exactly one new token per sequence.

Consider what has to happen for a single linear layer (a matrix-vector multiply at decode time):

- **Weight matrix:** for a hidden size of 4096, a weight matrix might be `4096 × 4096` = 16.7 million parameters × 2 bytes (BF16) = **33.5 MB** loaded from HBM
- **Compute performed:** for one token (a vector of size 4096), the multiply is `4096 × 4096 × 2` = **33.5 million FLOPs**

So for this layer:

<p align="center">
  <strong>Arithmetic intensity ≈ 33.5 × 10⁶ FLOPs / 33.5 × 10⁶ bytes = 1 FLOP/B</strong>
</p>

That is 1 FLOP/B against a ridge point of 156 FLOP/B. The kernel operates at less than 1% of the ridge point intensity.

The weight matrix must be fully loaded from HBM. The GPU does a tiny amount of arithmetic with it. Then it moves on to the next layer.

This is the core problem:

> Decode loads enormous amounts of data (model weights) to do a tiny amount of arithmetic (one token's worth).

---

## 6. A concrete decode memory calculation: Llama 3.1 8B

To make this concrete, consider a single decode step for Llama 3.1 8B.

**Model size:** ~8 billion parameters at BF16 = ~16 GB

At each decode step, the model must load essentially all the model weights to compute one token. The attention layers also load KV cache, but weights dominate at typical context lengths.

Using the approximate A100 HBM bandwidth of 2 TB/s:

<p align="center">
  <strong>Time to load 16 GB = 16 × 10⁹ / 2 × 10¹² = 8 ms</strong>
</p>

So the minimum time for a single decode step is approximately **8 ms** — not because the arithmetic takes 8 ms, but because it takes 8 ms just to move the weights from HBM to the compute units.

The arithmetic itself takes far less time:

<p align="center">
  <strong>FLOPs per token ≈ 2 × 8 × 10⁹ = 16 × 10⁹ FLOPs</strong>
</p>

<p align="center">
  <strong>Time to compute = 16 × 10⁹ / 312 × 10¹² ≈ 0.05 ms</strong>
</p>

The GPU is doing 0.05 ms of arithmetic but waiting 8 ms for data. It is sitting idle for over 99% of the time.

This is what "memory-bound" means in practice — not a slight imbalance, but a 160× gap between what the GPU could do and what the memory system allows it to do.

---

## 7. Why prefill is different: the matrix multiply advantage

During prefill, the model processes `N` prompt tokens together as a batch.

For the same `4096 × 4096` weight matrix, but now multiplied against a matrix of `N` query vectors:

- **Weight matrix loaded:** still **33.5 MB** — loaded once regardless of N
- **Compute performed:** `4096 × 4096 × N × 2 FLOPs` — scales linearly with N

For N = 512 (a 512-token prompt):

<p align="center">
  <strong>Arithmetic intensity ≈ (33.5 × 10⁶ × 512) FLOPs / 33.5 × 10⁶ bytes = 512 FLOP/B</strong>
</p>

That is 512 FLOP/B against a ridge point of 156 FLOP/B. Prefill is comfortably compute-bound at this prompt length.

The key insight:

> Prefill loads each weight once but uses it for N token computations. Decode loads each weight once but uses it for only 1 token computation. Longer prompts make prefill more compute-bound; single-token decode is always memory-bound.

More precisely, arithmetic intensity during a forward pass scales roughly as:

<p align="center">
  <strong>Intensity ≈ 2 × batch_size × sequence_length / bytes_moved</strong>
</p>

For decode, `sequence_length = 1`, so intensity is proportional only to `batch_size`. For prefill, `sequence_length = N`, so intensity is proportional to `N × batch_size`.

---

## 8. The transition point: when does decode become compute-bound?

Decode is not permanently condemned to be memory-bound. As batch size increases, the arithmetic intensity improves because the same weight load is amortized across more token computations.

For a weight matrix of `d × d` at BF16:

<p align="center">
  <strong>Arithmetic intensity = (2 × d² × B) / (2 × d²) = B</strong>
</p>

Where `B` is the batch size (tokens decoded in parallel).

This simplifies to:

> **Arithmetic intensity ≈ batch size (in FLOP/B for a square weight matrix)**

For an A100 with a ridge point of 156, decode becomes compute-bound only when batch size exceeds approximately 156 tokens per forward pass.

In practice, for non-square weight matrices (the feed-forward layers are typically 4× wider than the hidden size), the threshold is somewhat different — but the principle holds: a large enough batch size can push decode into compute-bound territory.

This is why batching is not just a throughput strategy — it is a mechanism for recovering GPU utilization.

| Batch size (tokens) | Approximate regime for Llama 3.1 8B on A100 |
|---:|---|
| 1 | Deeply memory-bound (<1% of compute ceiling) |
| 8 | Memory-bound |
| 32 | Memory-bound, improving |
| 64 | Approaching the ridge point |
| 128 | Near the transition |
| 256+ | Potentially compute-bound |

---

## 9. The KV cache adds another memory pressure

The weight loading problem is only half the story. As context length grows during decode, the model must also read the KV cache from HBM at every step.

For Llama 3.1 8B (32 layers, 8 KV heads, head dim 128, BF16):

<p align="center">
  <strong>KV bytes per token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB</strong>
</p>

At a context length of `T` tokens, the attention step reads approximately `T × 128 KB` of KV data per decode step.

For T = 4096 tokens:

<p align="center">
  <strong>KV read per step ≈ 4096 × 128 KB = 512 MB</strong>
</p>

Compare to the weight load of ~16 GB: at 4096 tokens, KV cache reads are about 3% of the weight bandwidth. At 16,384 tokens, they become ~12%. At 131,072 tokens (Llama 3.1's max context), KV reads would theoretically rival the weight load in bandwidth demands.

This means:

> At short-to-medium context lengths, decode bandwidth is dominated by weight loading. At very long context lengths, KV cache reads become a significant second source of memory pressure.

Both contribute to decode being memory-bound — and both scale in ways that compound with concurrency.

---

## 10. Operational intensity of attention specifically

Attention deserves separate analysis because it has a different structure than linear layers.

The attention computation for one token attending to a context of length `T`:

<p align="center">
  <strong>FLOPs: roughly 4 × H × d_head × T (for QK^T and weighted V sum)</strong>
</p>

<p align="center">
  <strong>Bytes: KV cache for T tokens = 2 × L × H_kv × d_head × T × b</strong>
</p>

For a single layer with H_kv heads, the attention intensity at decode time is roughly:

<p align="center">
  <strong>Attention intensity ≈ (4 × H × d_head × T) / (2 × H_kv × d_head × T × b)</strong>
</p>

<p align="center">
  <strong>≈ 2H / (H_kv × b)</strong>
</p>

For Llama 3.1 8B (H=32, H_kv=8, b=2 bytes):

<p align="center">
  <strong>≈ 2 × 32 / (8 × 2) = 4 FLOP/B</strong>
</p>

Attention has even lower arithmetic intensity than the linear layers during decode — only 4 FLOP/B versus ~1 FLOP/B for weights. Both are far below the ridge point.

This is why FlashAttention focuses on minimizing HBM reads rather than reducing FLOPs: in the memory-bound regime, bytes moved matters more than FLOPs performed.

---

## 11. Implications for hardware selection

The compute-vs-memory framing has direct implications for hardware choices.

When comparing two GPUs, the question to ask is:

> For my workload (prefill-heavy vs decode-heavy), which resource is actually limiting throughput?

**For decode-heavy workloads:**

The relevant metric is HBM bandwidth, not peak TFLOPS. A GPU with higher bandwidth will improve decode TPOT even if its TFLOP count is lower.

| GPU | HBM Bandwidth | Peak BF16 TFLOPS | Ridge point |
|---|---|---|---|
| A100 80GB | 2.0 TB/s | 312 | ~156 FLOP/B |
| H100 SXM5 | 3.35 TB/s | 989 | ~295 FLOP/B |
| H200 SXM | 4.8 TB/s | 989 | ~206 FLOP/B |

Note that the H100 has a higher ridge point than the H100/H200 — its compute grew faster than its bandwidth. For memory-bound decode, the H200's bandwidth improvement matters more than the H100 TFLOPS improvement.

**For prefill-heavy workloads:**

Peak TFLOPS matters more, since long prompts are compute-bound. Here the H100's dramatic compute improvement is directly useful.

**The practical implication:**

A serving system that is decode-heavy (many users, short prompts, long outputs) benefits more from memory bandwidth than raw compute. A system that is prefill-heavy (document processing, RAG with long retrieved contexts) benefits more from compute throughput.

---

## 12. What optimizations target which bottleneck

The compute-vs-memory framing directly organizes the optimization space:

| Optimization | Addresses | How |
|---|---|---|
| Continuous batching | Memory-bound decode | Amortizes weight loads across more concurrent tokens |
| Weight quantization (INT8/INT4) | Memory-bound decode | Reduces bytes loaded per weight; directly reduces bandwidth demand |
| KV cache quantization | Memory-bound decode | Reduces KV cache bytes read per attention step |
| GQA / MQA | Memory-bound decode | Fewer KV heads = fewer KV bytes = less attention bandwidth |
| FlashAttention | Memory-bound attention | Fuses kernels to minimize HBM roundtrips |
| Tensor parallelism | Both | More GPUs = more combined bandwidth + compute |
| Speculative decoding | Memory-bound decode | Larger effective batch of candidate tokens per weight load |
| Chunked prefill | Compute-bound prefill | Interleaves prefill chunks with decode to smooth utilization |

A key observation: most single-GPU optimizations target memory bandwidth, not compute. This is because decode dominates the latency of most serving workloads, and decode is memory-bound.

---

## 13. GPU utilization: a misleading metric

GPU utilization as reported by `nvidia-smi` (the `SM activity` percentage) measures whether the compute units are doing work — not whether the work is useful.

During memory-bound decode:

- SM utilization may be reported as 80-90%
- But 90% of that time is spent waiting for HBM data, not computing

The more meaningful metric is **MFU — Model FLOPs Utilization**:

<p align="center">
  <strong>MFU = actual FLOPs achieved / theoretical peak FLOPs</strong>
</p>

For single-request decode, MFU is often below 1% — the GPU is using less than 1% of its arithmetic potential. This is not a bug or a misconfiguration; it is the direct consequence of the memory-bound nature of decode.

High SM utilization during decode is not a sign of efficiency. It is a sign that the GPU is busy waiting. The right metric to optimize is bytes/second of HBM bandwidth utilization relative to peak bandwidth.

---

## 14. The bandwidth-compute interplay during a full request

A single request moves through several regimes as it is served:

**Phase 1 — Tokenization and queueing:** CPU-only, negligible GPU activity.

**Phase 2 — Prefill (short prompt, <128 tokens):** Still somewhat memory-bound. The prompt is too short to amortize the weight loads into compute-bound territory.

**Phase 3 — Prefill (long prompt, 1K+ tokens):** Compute-bound. Large matrix multiplications. High SM and tensor core utilization. TTFT grows linearly with prompt length.

**Phase 4 — First decode step:** Immediately drops to memory-bound territory. One token per step. Weights must be loaded from HBM again for each step.

**Phase 5 — Decode loop (single request):** Persistently memory-bound. TPOT is roughly constant at `model_size_bytes / bandwidth`, ~8 ms for Llama 3.1 8B on A100 at batch=1.

**Phase 6 — Decode loop (batched):** As more requests share the same forward pass, TPOT improves (toward compute-bound) but memory pressure increases due to accumulated KV caches.

The implication for system design:

> Prefill and decode should often be optimized differently because they live in different hardware regimes. An optimization that helps prefill (more FLOPs) may do nothing for decode (more bandwidth needed), and vice versa.

---

## 15. Why this matters for disaggregated serving

The different bottleneck regimes for prefill and decode motivate **disaggregated serving** — running prefill and decode on separate hardware.

If decode is memory-bound, the ideal hardware for decode has high HBM bandwidth and modest compute. If prefill is compute-bound, the ideal hardware for prefill has high TFLOPS and large enough memory for the model.

A serving system that mixes both phases on the same GPU is making neither hardware choice optimally.

Disaggregated serving (Splitwise, Mooncake, etc.) exploits this by routing prefill requests to compute-optimized hardware and decode steps to bandwidth-optimized hardware. This is covered in depth in Part 3.

---

## 16. Common misconceptions

### Misconception 1: "More FLOPs means faster inference."

Only for compute-bound workloads. For memory-bound decode, adding FLOPs does not improve TPOT. What matters is bandwidth.

### Misconception 2: "GPU utilization below 50% means the system is misconfigured."

Not for single-request decode. Low GPU utilization during decode is expected and normal. The fix is batching, not configuration.

### Misconception 3: "Quantization hurts quality and is only for resource-constrained environments."

Weight quantization reduces the bytes that must be loaded from HBM per decode step, directly improving TPOT in the memory-bound regime. For INT8, bandwidth is halved with minimal quality loss. Quality-neutral quantization is one of the highest-leverage optimizations for decode throughput.

### Misconception 4: "FlashAttention speeds up inference because it does fewer FLOPs."

FlashAttention does approximately the same number of FLOPs as standard attention. Its speedup comes from reducing HBM reads and writes — a memory bandwidth optimization, not a compute optimization.

### Misconception 5: "Increasing context length makes decode slower because the model has more to think about."

The correct framing: increasing context length grows the KV cache, which increases the bytes that must be read per attention step. Decode slows because of memory bandwidth pressure, not because of any change in the model's "reasoning."

---

## 17. Key takeaways

- A GPU has two subsystems: compute units and HBM. Whichever is slower determines performance.
- Arithmetic intensity (FLOPs/byte) determines which subsystem is the bottleneck.
- The ridge point — compute / bandwidth — is the threshold between the two regimes.
- For an A100, the ridge point is approximately 156 FLOP/B.
- During prefill with long prompts, arithmetic intensity is high and the workload is compute-bound.
- During single-request decode, arithmetic intensity is approximately 1 FLOP/B — far below the ridge point.
- Decode is memory-bound: the GPU is waiting for weights and KV cache data, not for arithmetic.
- Decode TPOT for a single request is approximately `model_size_bytes / HBM_bandwidth`.
- For Llama 3.1 8B on A100: decode TPOT ≈ 16 GB / 2 TB/s ≈ 8 ms at batch size 1.
- Batching improves arithmetic intensity linearly — doubling batch size roughly doubles intensity.
- Decode becomes compute-bound only around batch size 150+ for Llama 3.1 8B on A100.
- At long context lengths, KV cache reads add significant bandwidth pressure on top of weight loading.
- Most single-GPU decode optimizations target memory bandwidth (quantization, GQA, KV compression) rather than compute.
- High SM utilization during decode does not mean efficient GPU use — MFU is the right metric.

---

## Next

The next file translates this hardware framing into the two inference phases and makes the numbers concrete with Llama 3.1 8B:

[`03-prefill-and-decode.md`](./03-prefill-and-decode.md)