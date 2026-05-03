# Prefill and Decode

This chapter explains the two phases that define almost every LLM serving system:

> **Prefill** processes the prompt.  
> **Decode** generates the response.

They use the same model weights, but they behave like very different workloads. Prefill is usually more compute-oriented. Decode is often more memory-bandwidth-oriented. Understanding this distinction is the foundation for understanding continuous batching, PagedAttention, prefix caching, chunked prefill, speculative decoding, and KV cache quantization.

---

## 1. Why this distinction matters

A decoder-only LLM does not generate a full answer in one forward pass. It receives a prompt, processes it, and then generates output tokens one by one.

A simple request looks like this:

<p align="center">
  <strong>prompt tokens → prefill → first token → decode loop → remaining tokens</strong>
</p>

The important part is that the first stage and the repeated generation stage have different hardware behavior.

| Phase | Main job | Token pattern | Main bottleneck |
|---|---|---|---|
| Prefill | Process the prompt | Many input tokens at once | GPU compute |
| Decode | Generate output | One token per step | Memory bandwidth |

A serving optimization should almost always be classified by asking:

> Is this optimization helping prefill, decode, KV cache memory, or scheduling?

If that question is unclear, the optimization is probably not well understood yet.

---

## 2. The request timeline

For one request, the serving timeline is roughly:

<p align="center">
  <strong>tokenization → queueing → prefill → first output token → decode step 1 → decode step 2 → ... → stop</strong>
</p>

The user experiences this timeline through two latency metrics:

| Metric | Meaning | Mostly affected by |
|---|---|---|
| TTFT | Time to first token | queueing, tokenization, prefill |
| TPOT / ITL | Time per output token / inter-token latency | decode |

A simplified latency model is:

<p align="center">
  <strong>TTFT ≈ queue time + tokenization time + prefill time</strong>
</p>

For a response with `M` generated tokens:

<p align="center">
  <strong>total latency ≈ TTFT + M × TPOT</strong>
</p>

So prefill determines how quickly the model starts answering. Decode determines how quickly the answer continues.

---

## 3. Prefill: what happens

During prefill, the model processes the entire prompt.

Let the prompt contain `N` tokens:

<p align="center">
  <strong>x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>N</sub>)</strong>
</p>

All these prompt tokens are already known. That means the model can process them together through the Transformer stack.

For each Transformer layer, the model computes attention and feed-forward network outputs for all prompt positions. In attention, the hidden states are projected into queries, keys, and values:

<p align="center">
  <strong>Q = XW<sub>Q</sub></strong><br>
  <strong>K = XW<sub>K</sub></strong><br>
  <strong>V = XW<sub>V</sub></strong>
</p>

Then attention is computed as:

<p align="center">
  <strong>Attention(Q, K, V) = softmax((QK<sup>T</sup>) / √d) V</strong>
</p>

During prefill, the model also writes the prompt keys and values into the KV cache. This cache will be reused during decode.

Prefill has three main outputs:

1. hidden states for the prompt,
2. initial KV cache entries for all prompt tokens,
3. logits used to sample the first output token.

---

## 4. Why prefill is usually compute-bound

Prefill processes many tokens together. This produces large matrix operations.

Large matrix operations are good for GPUs because they provide a lot of parallel work. The GPU can keep many compute units busy.

A rough mental model is:

<p align="center">
  <strong>prefill work ∝ model size × prompt length</strong>
</p>

More prompt tokens mean more computation before the first token can be produced.

In practical terms:

- short prompt → smaller prefill cost
- long prompt → larger prefill cost
- very long prompt → TTFT can become large

This is why long-context prompts often feel slow before the first token appears.

---

## 5. Decode: what happens

After prefill produces the first output token, the model enters the decode loop.

At decode step `t`, the model predicts the next token:

<p align="center">
  <strong>P(y<sub>t</sub> | x, y<sub>&lt;t</sub>)</strong>
</p>

The model uses:

- the original prompt `x`,
- all previously generated tokens `y_<t`,
- the KV cache created so far.

The key difference from prefill is that decode only processes the newest token at each step.

A simplified decode loop:

<p align="center">
  <strong>previous token → model step → next-token logits → sample token → append KV → repeat</strong>
</p>

At every decode step, the model:

1. takes the previous token,
2. computes a new query, key, and value,
3. reads cached keys and values from previous tokens,
4. computes attention for the new token,
5. produces logits,
6. samples or selects the next token,
7. appends the new key and value to the KV cache.

---

## 6. Why decode is usually memory-bandwidth-bound

Decode produces one token at a time. That creates less compute per step than prefill.

However, each decode step still needs to read a lot of data:

- model weights,
- cached keys,
- cached values,
- intermediate activations.

A rough mental model is:

<p align="center">
  <strong>decode cost per token ≈ weight reads + KV cache reads + small compute step</strong>
</p>

The GPU may have enough compute capability to perform the arithmetic quickly, but it still has to wait for data to arrive from high-bandwidth memory.

This is why decode is often memory-bandwidth-bound:

> The GPU has massive compute power, but each decode step does too little arithmetic relative to the amount of data it must read.

This also explains why simply having more theoretical FLOPs does not always improve decode speed. Memory bandwidth and cache layout matter heavily.

---

## 7. Prefill vs decode side by side

| Dimension | Prefill | Decode |
|---|---|---|
| Input | Full prompt | Previous token + KV cache |
| Token count per step | Many tokens | One token |
| Parallelism across sequence | High | Low |
| Main output | First token + initial KV cache | Subsequent tokens |
| Main latency metric | TTFT | TPOT / ITL |
| Main hardware bottleneck | Compute throughput | Memory bandwidth |
| Typical operation shape | Matrix-matrix | Matrix-vector-like |
| Scheduling challenge | Long prefills can block decode | Many small steps must be batched |

The most important intuition:

> Prefill is a big parallel computation.  
> Decode is a repeated memory-access workload.

---

## 8. Concrete example: Llama 3.1 8B with a 512-token prompt

This section uses Llama 3.1 8B as a reference model. Exact model details can vary by implementation, but the following values are useful for building intuition.

Approximate assumptions:

| Quantity | Symbol | Value |
|---|---:|---:|
| Number of layers | `L` | 32 |
| Hidden size | `d_model` | 4096 |
| Attention heads | `H` | 32 |
| KV heads | `H_kv` | 8 |
| Head dimension | `d_head` | 128 |
| Prompt length | `N` | 512 |
| Precision | `b` | 2 bytes for FP16/BF16 |

Llama 3.1 8B uses grouped-query attention, so the number of KV heads is smaller than the number of query heads. That matters for KV cache size.

---

## 9. Approximate prefill compute

A very rough serving-level estimate is that a forward pass through an `8B` parameter model costs about:

<p align="center">
  <strong>~2 × parameters FLOPs per token</strong>
</p>

For an 8B model:

<p align="center">
  <strong>FLOPs per token ≈ 2 × 8B = 16B FLOPs</strong>
</p>

For a 512-token prompt:

<p align="center">
  <strong>prefill FLOPs ≈ 512 × 16B = 8192B FLOPs</strong>
</p>

That is:

<p align="center">
  <strong>prefill FLOPs ≈ 8.2 trillion FLOPs</strong>
</p>

This is a rough estimate, not an exact profiler result. It is useful because it shows why prefill can be compute-heavy: a long prompt requires a large amount of forward-pass computation before the first token appears.

Important nuance:

- The feed-forward network and projection layers dominate much of the model FLOPs.
- Attention also contributes, especially as context length grows.
- Exact FLOPs depend on architecture details and implementation.

For intuition, the message is enough:

> Longer prompts increase TTFT because prefill must process all prompt tokens before generation begins.

---

## 10. KV cache size after a 512-token prefill

The KV cache size formula is:

<p align="center">
  <strong>KV cache bytes = 2 × L × H<sub>kv</sub> × d<sub>head</sub> × T × b</strong>
</p>

Where:

| Symbol | Meaning |
|---|---|
| `2` | one tensor for K and one tensor for V |
| `L` | number of layers |
| `H_kv` | number of KV heads |
| `d_head` | head dimension |
| `T` | sequence length |
| `b` | bytes per element |

For the 512-token prompt:

<p align="center">
  <strong>KV bytes = 2 × 32 × 8 × 128 × 512 × 2</strong>
</p>

Step by step:

<p align="center">
  <strong>2 × 32 = 64</strong><br>
  <strong>64 × 8 = 512</strong><br>
  <strong>512 × 128 = 65,536</strong><br>
  <strong>65,536 × 512 = 33,554,432</strong><br>
  <strong>33,554,432 × 2 = 67,108,864 bytes</strong>
</p>

So:

<p align="center">
  <strong>KV cache for 512 tokens ≈ 67 MB</strong>
</p>

This is only for one request. With many concurrent requests, KV memory accumulates quickly.

---

## 11. During decode, how much data moves per step?

After prefill, each new generated token extends the sequence by one.

For one additional token, the new KV written is:

<p align="center">
  <strong>new KV bytes per token = 2 × L × H<sub>kv</sub> × d<sub>head</sub> × b</strong>
</p>

Using the same Llama 3.1 8B assumptions:

<p align="center">
  <strong>2 × 32 × 8 × 128 × 2 = 131,072 bytes</strong>
</p>

So each generated token adds approximately:

<p align="center">
  <strong>~128 KB of KV cache per request</strong>
</p>

But writing the new KV is not the only memory movement. The model must also read:

- model weights,
- cached keys and values from previous tokens,
- current activations,
- output projection data.

For attention specifically, the amount of cached KV read grows with context length. At a 512-token context, the cached KV footprint is already about 67 MB for that request. As the generated sequence gets longer, the attention step reads over a larger history.

The important intuition:

> Decode keeps adding only one token, but each step attends over a growing history.

---

## 12. Why batching helps decode

A single decode step for one request is inefficient because it uses the model weights to generate only one token.

Batching helps because multiple requests can share the same weight reads. Instead of loading weights for one token, the system loads weights and applies them across many active requests.

Conceptually:

<p align="center">
  <strong>one request → low reuse of loaded weights</strong><br>
  <strong>many requests batched → better reuse of loaded weights</strong>
</p>

This is why serving engines try to batch decode steps from many requests together.

However, batching is difficult because:

- requests arrive at different times,
- prompts have different lengths,
- outputs finish at different times,
- KV cache memory differs across requests.

This is the motivation for continuous batching.

---

## 13. Why long prefill can interfere with decode

A long prefill can occupy the GPU with a large compute-heavy operation.

Meanwhile, existing requests may be waiting for their next decode step.

If the scheduler is naive, long prefills can increase the per-token latency of ongoing generations.

This creates a scheduling problem:

> How can the engine process new prompts without hurting ongoing decode latency too much?

Techniques such as chunked prefill exist because of this problem. Instead of processing a very long prompt in one large block, the engine can split prefill into chunks and interleave it with decode work.

---

## 14. Optimization map

The prefill/decode distinction helps organize the entire handbook.

| Optimization | Mainly helps | Why |
|---|---|---|
| Continuous batching | Decode throughput | Batches many one-token decode steps |
| PagedAttention | KV cache memory | Reduces fragmentation and improves cache management |
| Prefix caching | Prefill | Reuses shared prompt prefixes |
| Chunked prefill | Scheduling | Interleaves long prefills with decode |
| FlashAttention | Attention efficiency | Reduces memory movement inside attention |
| KV cache quantization | Decode memory | Reduces KV cache footprint |
| Weight quantization | Memory and throughput | Reduces model weight memory bandwidth |
| Speculative decoding | Decode latency | Generates multiple candidate tokens faster |

The key is to always ask what phase the optimization targets.

---

## 15. Common misconceptions

### Misconception 1: “Inference is just one forward pass.”

For one generated token, yes. For a full response, no. A response requires one prefill pass and many decode steps.

### Misconception 2: “The GPU is always compute-bound.”

Not during decode. Decode often waits on memory movement.

### Misconception 3: “Long prompts only affect memory.”

Long prompts affect both prefill latency and KV cache memory.

### Misconception 4: “Batching always improves user experience.”

Batching improves throughput, but can hurt latency if scheduling is poor.

---

## 16. Key takeaways

- LLM inference has two major phases: prefill and decode.
- Prefill processes all prompt tokens together.
- Prefill builds the initial KV cache.
- Prefill usually affects TTFT.
- Decode generates one token at a time.
- Decode repeatedly reads and extends the KV cache.
- Decode usually affects TPOT / ITL.
- Prefill is usually more compute-oriented.
- Decode is often more memory-bandwidth-oriented.
- A 512-token prompt for Llama 3.1 8B requires a large amount of forward-pass compute.
- The resulting KV cache for that prompt is roughly tens of megabytes per request.
- Serving optimizations make sense only when you know which phase they target.

---

## Next

The next file goes deeper into the memory structure that makes decode efficient:

[`kv-cache-explained.md`](./kv-cache-explained.md)
