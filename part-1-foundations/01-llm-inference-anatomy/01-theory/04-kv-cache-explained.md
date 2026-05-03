# KV Cache Explained

> 📖 Recommended order: You should read this after `03-prefill-and-decode.md`.

This chapter explains one of the most important mechanisms in LLM serving:

> The KV cache.

The KV cache is the reason autoregressive generation is fast enough to be practical. It avoids recomputing attention information for previous tokens at every decode step.

But it also creates one of the main scaling problems in serving: memory pressure.

This chapter builds the idea from attention basics to serving-level memory calculations.

---

## 1. Why the KV cache exists

A decoder-only LLM generates text one token at a time.

At generation step `t`, the model predicts:

<p align="center">
  <strong>P(y<sub>t</sub> | x, y<sub>&lt;t</sub>)</strong>
</p>

This means the new token depends on everything before it:

- the original prompt,
- all previously generated tokens.

So every new token must attend to previous tokens. The question is:

> Do we recompute information for all previous tokens every time, or do we store it?

The KV cache is the answer:

> Store the reusable attention information from previous tokens.

---

## 2. Quick attention recap

Inside each Transformer layer, the model computes queries, keys, and values.

Given hidden states `X`:

<p align="center">
  <strong>Q = XW<sub>Q</sub></strong><br>
  <strong>K = XW<sub>K</sub></strong><br>
  <strong>V = XW<sub>V</sub></strong>
</p>

Attention is then computed as:

<p align="center">
  <strong>Attention(Q, K, V) = softmax((QK<sup>T</sup>) / √d) V</strong>
</p>

A useful intuition:

| Tensor | Role |
|---|---|
| `Q` | What this token is looking for |
| `K` | What each token offers as an address |
| `V` | The information carried by each token |

The query of the current token is compared against keys from previous tokens. The result decides how much value information to read from each previous token.

---

## 3. What changes during decode

During prefill, all prompt tokens are processed together.

During decode, only the newest token is processed at each step.

Suppose the model has already seen tokens:

<p align="center">
  <strong>(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>t-1</sub>)</strong>
</p>

Now a new token arrives at position `t`.

The model needs attention over the full history:

<p align="center">
  <strong>current query q<sub>t</sub> attends to previous keys K<sub>&lt;t</sub></strong>
</p>

The key observation:

> Keys and values for previous tokens do not change.

Once the model has computed `K_1`, `V_1`, `K_2`, `V_2`, and so on, those tensors can be reused.

---

## 4. Without KV cache

Without a KV cache, every decode step would recompute keys and values for all previous tokens.

For step `t`, the model would recompute:

<p align="center">
  <strong>K<sub>1</sub>, K<sub>2</sub>, ..., K<sub>t</sub></strong><br>
  <strong>V<sub>1</sub>, V<sub>2</sub>, ..., V<sub>t</sub></strong>
</p>

Then at step `t+1`, it would recompute almost the same tensors again.

That creates repeated work.

A simple cost intuition:

| Step | Tokens recomputed |
|---|---:|
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| ... | ... |
| T | T |

Total work grows like:

<p align="center">
  <strong>1 + 2 + 3 + ... + T = T(T + 1) / 2</strong>
</p>

So the total recomputation cost grows approximately as:

<p align="center">
  <strong>O(T<sup>2</sup>)</strong>
</p>

This is too expensive for long outputs.

---

## 5. With KV cache

With a KV cache, the model stores previous keys and values.

At decode step `t`, it only computes the new tensors:

<p align="center">
  <strong>q<sub>t</sub>, K<sub>t</sub>, V<sub>t</sub></strong>
</p>

Then it attends using:

<p align="center">
  <strong>q<sub>t</sub> with cached K<sub>≤t</sub> and V<sub>≤t</sub></strong>
</p>

Finally, it appends the new key and value:

<p align="center">
  <strong>K<sub>≤t</sub> = [K<sub>&lt;t</sub> ; K<sub>t</sub>]</strong><br>
  <strong>V<sub>≤t</sub> = [V<sub>&lt;t</sub> ; V<sub>t</sub>]</strong>
</p>

Now the model avoids recomputing old keys and values.

The total cached state grows linearly with sequence length:

<p align="center">
  <strong>KV cache size ∝ T</strong>
</p>

The trade-off:

| Without KV cache | With KV cache |
|---|---|
| Less memory | More memory |
| Recomputes past K/V | Reuses past K/V |
| Very slow decode | Practical decode |
| Compute-heavy repeated work | Memory-heavy cached state |

The core idea:

> KV cache trades compute for memory.

---

## 6. Decode step with KV cache

At each decode step:

1. Take the newest token representation `x_t`.
2. Compute query, key, and value:

<p align="center">
  <strong>q<sub>t</sub> = x<sub>t</sub>W<sub>Q</sub></strong><br>
  <strong>K<sub>t</sub> = x<sub>t</sub>W<sub>K</sub></strong><br>
  <strong>V<sub>t</sub> = x<sub>t</sub>W<sub>V</sub></strong>
</p>

3. Attend over cached history:

<p align="center">
  <strong>Attention(q<sub>t</sub>, K<sub>≤t</sub>, V<sub>≤t</sub>) = softmax((q<sub>t</sub>K<sub>≤t</sub><sup>T</sup>) / √d) V<sub>≤t</sub></strong>
</p>

4. Produce logits for the next token.
5. Append `K_t` and `V_t` to the cache.

This process repeats until generation stops.

---

## 7. What exactly is stored?

For each layer, the serving system stores:

- key vectors for previous tokens,
- value vectors for previous tokens.

For a single layer, the cache shape is conceptually:

<p align="center">
  <strong>K cache: sequence length × KV heads × head dimension</strong><br>
  <strong>V cache: sequence length × KV heads × head dimension</strong>
</p>

Across all layers:

<p align="center">
  <strong>KV cache = all layers × both K and V × all cached tokens</strong>
</p>

This is why the cache can become large. It is not just storing token IDs. It is storing high-dimensional tensors for every cached token in every layer.

---

## 8. KV cache size formula

The standard formula is:

<p align="center">
  <strong>KV cache bytes = 2 × L × H<sub>kv</sub> × d<sub>head</sub> × T × b</strong>
</p>

Where:

| Symbol | Meaning |
|---|---|
| `2` | K and V |
| `L` | number of Transformer layers |
| `H_kv` | number of KV heads |
| `d_head` | dimension of each head |
| `T` | sequence length |
| `b` | bytes per element |

The most important scaling rule is:

<p align="center">
  <strong>KV cache size grows linearly with sequence length.</strong>
</p>

If the context doubles, KV cache memory roughly doubles.

If the number of active requests doubles, total KV memory roughly doubles.

---

## 9. Concrete calculation: Llama 3.1 8B at 4096 tokens

Use approximate Llama 3.1 8B assumptions:

| Quantity | Symbol | Value |
|---|---:|---:|
| Layers | `L` | 32 |
| KV heads | `H_kv` | 8 |
| Head dimension | `d_head` | 128 |
| Sequence length | `T` | 4096 |
| Precision | `b` | 2 bytes |

Plug into the formula:

<p align="center">
  <strong>KV bytes = 2 × 32 × 8 × 128 × 4096 × 2</strong>
</p>

Step by step:

<p align="center">
  <strong>2 × 32 = 64</strong><br>
  <strong>64 × 8 = 512</strong><br>
  <strong>512 × 128 = 65,536</strong><br>
  <strong>65,536 × 4096 = 268,435,456</strong><br>
  <strong>268,435,456 × 2 = 536,870,912 bytes</strong>
</p>

So the KV cache is approximately:

<p align="center">
  <strong>536,870,912 bytes ≈ 512 MiB ≈ 0.5 GiB</strong>
</p>

For one active request at 4096 tokens, the KV cache is roughly half a GiB under these assumptions.

Important note:

> If the model uses more KV heads, higher precision, longer context, or larger hidden dimensions, the KV cache grows quickly.

---

## 10. Why some estimates differ

You may see different KV cache estimates for “8B models” online. This usually happens because of different assumptions.

The most common difference is the number of KV heads.

Older multi-head attention stores K and V for every attention head:

<p align="center">
  <strong>H<sub>kv</sub> = H</strong>
</p>

Grouped-query attention stores fewer KV heads:

<p align="center">
  <strong>H<sub>kv</sub> &lt; H</strong>
</p>

For example, if `H_kv = 32` instead of `8`, the KV cache becomes 4× larger.

Using `H_kv = 32` at 4096 tokens:

<p align="center">
  <strong>KV bytes = 2 × 32 × 32 × 128 × 4096 × 2 ≈ 2 GiB</strong>
</p>

So both numbers can be correct depending on architecture.

The safe rule:

> Always check the model’s number of KV heads before estimating KV cache memory.

---

## 11. KV cache and concurrent requests

Serving systems rarely handle only one request.

If there are `R` active requests, total KV memory is approximately:

<p align="center">
  <strong>Total KV memory ≈ Σ<sub>r=1</sub><sup>R</sup> KV memory of request r</strong>
</p>

If all requests have similar sequence length, a rough estimate is:

<p align="center">
  <strong>Total KV memory ≈ R × KV memory per request</strong>
</p>

Example:

| Active requests | KV per request | Total KV memory |
|---:|---:|---:|
| 1 | 0.5 GiB | 0.5 GiB |
| 8 | 0.5 GiB | 4 GiB |
| 32 | 0.5 GiB | 16 GiB |
| 64 | 0.5 GiB | 32 GiB |

This is why KV cache memory directly limits concurrency.

---

## 12. Why KV cache can dominate serving memory

Model weights are fixed once the model is loaded.

KV cache is dynamic. It grows with:

- sequence length,
- output length,
- number of active requests,
- number of layers,
- number of KV heads,
- precision.

The model weights may fit on the GPU, but that does not mean the server can handle many long requests.

A useful mental model:

<p align="center">
  <strong>GPU memory = model weights + KV cache + activations + runtime overhead</strong>
</p>

During serving, the model weights are mostly constant, while the KV cache expands and contracts as requests arrive and finish.

This makes KV cache management a central scheduling problem.

---

## 13. Memory fragmentation problem

A naive serving system may allocate a large contiguous KV cache region for each request.

That creates fragmentation.

Example:

- request A uses 700 tokens,
- request B uses 2048 tokens,
- request C finishes early,
- request D arrives and needs 1500 tokens.

If memory was allocated in large contiguous chunks, there may be enough total free memory but not enough contiguous free memory.

This is the problem PagedAttention addresses.

PagedAttention treats KV cache memory more like virtual memory pages:

> Instead of requiring one contiguous block per sequence, it stores KV blocks in fixed-size pages.

This improves memory utilization and allows more active requests.

---

## 14. Prefix reuse problem

Many real workloads reuse the same prompt prefix.

Examples:

- system prompt,
- policy instructions,
- retrieval template,
- few-shot examples,
- shared chat history prefix.

Without prefix caching, the model recomputes the same prompt prefix many times.

With prefix caching, the engine can reuse previously computed KV cache entries for shared prefixes.

This reduces prefill cost and improves TTFT for repeated prompts.

---

## 15. KV cache quantization

KV cache is usually stored in FP16 or BF16.

But if memory is the bottleneck, the cache can be quantized.

For example:

| Format | Bytes per element | Memory impact |
|---|---:|---|
| FP16 / BF16 | 2 | Baseline |
| INT8 | 1 | ~2× smaller |
| INT4 | 0.5 | ~4× smaller |

The trade-off is quality and implementation complexity.

KV cache quantization is attractive because decode is memory-heavy. Reducing KV bytes can directly improve serving capacity and sometimes speed.

---

## 16. Why the KV cache matters for scheduling

Schedulers need to know whether a request can fit in memory.

A request is not just “one request.” It has a memory footprint that grows every token.

The scheduler must consider:

- current sequence length,
- expected output length,
- available KV blocks,
- other active requests,
- prefill vs decode priority,
- eviction or cancellation policies.

This is why serving systems are not simple queues. They are memory-aware schedulers.

---

## 17. Common misconceptions

### Misconception 1: “The KV cache stores tokens.”

It does not store token IDs. It stores key and value tensors for each layer.

### Misconception 2: “The KV cache is optional.”

For practical long-form autoregressive generation, it is essential. Without it, decode would be far too slow.

### Misconception 3: “KV cache only matters for long context.”

It matters for all serving, but long context and high concurrency make it dominant.

### Misconception 4: “If model weights fit, serving will fit.”

Not necessarily. KV cache can consume a large amount of additional memory.

---

## 18. Connection to later topics

The KV cache explains why many later topics exist.

| Topic | Why it exists |
|---|---|
| PagedAttention | KV cache memory becomes fragmented |
| Continuous batching | Decode steps need to be batched efficiently |
| Prefix caching | Shared prompt prefixes should not be recomputed |
| KV cache quantization | KV memory can dominate GPU memory |
| Chunked prefill | Long prefills can interfere with decode |
| Disaggregated serving | Prefill and decode stress hardware differently |

Once KV cache behavior is clear, these techniques become much easier to understand.

---

## 19. Key takeaways

- Attention uses queries, keys, and values.
- During decode, keys and values for previous tokens do not change.
- The KV cache stores those previous keys and values.
- Without KV cache, decode repeatedly recomputes past K/V tensors.
- With KV cache, decode computes only the new token’s K/V and reuses the past.
- KV cache trades compute for memory.
- KV cache size grows linearly with sequence length.
- Total KV memory grows with the number of active requests.
- For Llama 3.1 8B with grouped-query attention, 4096 tokens is roughly 0.5 GiB of KV cache per request in FP16/BF16.
- If the model uses more KV heads, the cache can be much larger.
- KV cache memory is one of the main constraints in production LLM serving.

---

## Next

The KV cache is the foundation for several later topics:

- PagedAttention
- continuous batching
- prefix caching
- KV cache quantization
- chunked prefill
