# The Anatomy of LLM Inference

This chapter answers one question from a systems perspective:

> What actually happens, step by step, when you send a prompt to an LLM and receive a response back?

At a high level, LLM inference is the process of generating an output token sequence from an input prompt using a model whose parameters are already fixed.

<p align="center">
  <strong>x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)</strong><br>
  <em>input prompt tokens</em>
</p>

<p align="center">
  <strong>y = (y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>m</sub>)</strong><br>
  <em>generated output tokens</em>
</p>

The model generates the response **autoregressively**. That means it does not produce the whole answer in one shot. It repeatedly predicts the next token, appends that token to the sequence, and then predicts the next one.

<p align="center">
  <strong>P(y | x) = ∏<sub>t=1</sub><sup>m</sup> P(y<sub>t</sub> | x, y<sub>&lt;t</sub>)</strong>
</p>

Read this as:

> The probability of the full response is built from many next-token prediction steps.

At generation step `t`, the model predicts token `y_t` using both the original prompt `x` and all previously generated tokens `y_<t`.

Inference is therefore very different from training. There is no backpropagation, no gradient computation, and no optimizer step. The model weights are frozen. The serving system only performs forward-pass execution over fixed model weights.

At small scale, inference may look like a simple `model.generate()` call. At production scale, however, inference becomes a systems problem dominated by latency, throughput, batching, GPU memory, memory bandwidth, and KV cache management.

---

## 1. End-to-end request lifecycle

The full inference pipeline is shown below.

![End-to-end LLM inference pipeline](../assets/overview/llm-inference-pipeline.png)

<p align="center">
  <em>Figure 1. End-to-end lifecycle of an LLM inference request, from raw prompt to streamed response.</em>
</p>

A request flows through the following stages.

### 1. Raw prompt

The user sends raw text to the serving system.

Example:

```text
Explain why decode is memory-bound in LLM inference.
```

### 2. Tokenization

The tokenizer converts raw text into integer token IDs.

If the input text is `s`, tokenization maps it into a discrete sequence:

<p align="center">
  <strong>s → (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)</strong>
</p>

The model does not directly operate on characters or words. It operates on token IDs.

### 3. Input token IDs

The token IDs are passed into the model. These tokens form the prompt sequence.

### 4. Prefill

The model processes the entire prompt. During this phase, all prompt tokens are processed together, and the initial KV cache is built.

### 5. KV cache initialized

The model stores key and value tensors for the prompt tokens. These cached tensors will be reused during decoding.

### 6. First output token

After prefill, the model produces the first generated token. This marks the end of the **time-to-first-token**, or **TTFT**, path.

### 7. Decode loop

The model enters the autoregressive generation loop. It repeatedly generates one new token at a time.

### 8. Detokenization

Generated token IDs are converted back into text.

### 9. Streamed response

The generated text is streamed back to the client.

A useful mental model is:

> **Prefill** determines how quickly generation can begin.  
> **Decode** determines how quickly the rest of the response arrives.

---

## 2. Inference vs. training

Training and inference both use Transformer layers, but they are optimized for very different objectives.

| Aspect | Training | Inference |
|---|---|---|
| Model weights | Updated | Frozen |
| Backpropagation | Yes | No |
| Gradients | Required | Not used |
| Optimizer state | Required | Not used |
| Execution pattern | Forward + backward | Forward only |
| Sequence handling | Highly parallel | Prefill parallel, decode sequential |
| Main concern | Learning efficiently | Serving efficiently |

A concise summary is:

> Training is about learning model parameters.  
> Inference is about serving tokens efficiently with fixed parameters.

This distinction matters because most production bottlenecks in LLM serving are not caused by learning. They are caused by repeatedly moving tokens through fixed weights while managing GPU memory and concurrent requests.

---

## 3. The two phases that define everything

The most important distinction in LLM serving is the difference between **prefill** and **decode**.

![Prefill vs decode](../assets/overview/prefill-vs-decode.png)

<p align="center">
  <em>Figure 2. Prefill and decode have fundamentally different execution patterns, bottlenecks, and serving implications.</em>
</p>

---

### 3.1 Prefill

During **prefill**, the model processes the entire prompt at once.

If the prompt contains `n` tokens, the model computes hidden states for all `n` prompt positions through the Transformer stack. Because many prompt tokens are processed together, prefill usually gives the GPU enough parallel work to keep its compute units busy.

Inside a self-attention layer, the standard attention operation is:

<p align="center">
  <strong>Attention(Q, K, V) = softmax((QK<sup>T</sup>) / √d) V</strong>
</p>

Where:

| Symbol | Meaning |
|---|---|
| `Q` | query matrix |
| `K` | key matrix |
| `V` | value matrix |
| `d` | attention head dimension |
| `K^T` | transpose of the key matrix |

During prefill, the model computes `Q`, `K`, and `V` for the prompt tokens. The key and value tensors are then stored in the KV cache.

Key properties of prefill:

- input: the **entire prompt**
- execution: all prompt tokens processed together
- hardware tendency: more **compute-bound**
- main role: build the initial KV cache
- output: produces the **first output token**
- user-facing metric: mainly affects **TTFT**

Intuitively:

> Prefill is the phase where the model reads and digests the prompt.

---

### 3.2 Decode

After the first token is produced, the model enters the **decode** phase.

At decode step `t`, the model predicts:

<p align="center">
  <strong>P(y<sub>t</sub> | x, y<sub>&lt;t</sub>)</strong>
</p>

This means the next token `y_t` is generated using:

- the original prompt `x`
- all previously generated tokens `y_<t`

Unlike prefill, decode is sequential across time. The model cannot generate `y_t` before it has generated `y_(t-1)`.

A simplified decode loop is:

<p align="center">
  <strong>y<sub>t</sub> ~ P(next token | x, y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>t-1</sub>)</strong>
</p>

Then the generated token is appended to the sequence:

<p align="center">
  <strong>(y<sub>1</sub>, ..., y<sub>t-1</sub>) → (y<sub>1</sub>, ..., y<sub>t-1</sub>, y<sub>t</sub>)</strong>
</p>

Key properties of decode:

- input: previous token + KV cache
- execution: one token generated at a time
- hardware tendency: more **memory-bandwidth-bound**
- main role: read and extend the KV cache
- output: produces subsequent output tokens
- user-facing metric: mainly affects **TPOT / ITL**

A simple way to think about it is:

> Prefill processes a whole prompt.  
> Decode repeats a small forward step many times.

---

## 4. Why prefill and decode behave differently

Although prefill and decode use the same model weights, they stress the hardware differently.

### Prefill is more compute-oriented

In prefill, many tokens are processed together. This gives the GPU a large amount of parallel work. The computation looks more like matrix-matrix multiplication, which can use GPU compute units efficiently.

A rough mental model is:

<p align="center">
  <strong>Prefill work ∝ model size × prompt length</strong>
</p>

As prompt length increases, prefill work increases because more input tokens must be processed.

### Decode is more memory-oriented

In decode, the model generates only one token per step for each sequence. The amount of computation per step is smaller, but the system still has to repeatedly access:

- model weights
- the relevant KV cache

A rough mental model is:

<p align="center">
  <strong>Decode cost per token ≈ weight reads + KV cache reads + small compute step</strong>
</p>

This is why decode often becomes limited not by peak FLOPs, but by memory bandwidth.

The key systems question is:

> Is the bottleneck compute, or memory movement?

That single question explains why many LLM serving optimizations exist.

---

## 5. KV cache: the central serving primitive

The KV cache is the core mechanism that makes autoregressive generation efficient.

![KV cache overview](../assets/overview/kv-cache-overview.png)

<p align="center">
  <em>Figure 3. The KV cache stores past keys and values so that decode can reuse them instead of recomputing them from scratch.</em>
</p>

---

### 5.1 What problem does the KV cache solve?

Without a KV cache, every new decode step would require recomputing keys and values for all previous tokens.

That would be extremely inefficient.

Instead, the model stores the **key** and **value** tensors from previous tokens and reuses them at future steps.

At decode step `t`:

1. a new query `q_t` is computed for the current token
2. the model attends over cached past keys and values
3. the new `K_t` and `V_t` are appended to the cache

This turns repeated recomputation into incremental reuse.

The basic trade-off is:

> KV cache saves compute by spending memory.

---

### 5.2 Conceptual decode step

At a high level, decode step `t` computes a query, key, and value for the current token:

<p align="center">
  <strong>q<sub>t</sub> = x<sub>t</sub>W<sub>Q</sub></strong><br>
  <strong>K<sub>t</sub> = x<sub>t</sub>W<sub>K</sub></strong><br>
  <strong>V<sub>t</sub> = x<sub>t</sub>W<sub>V</sub></strong>
</p>

The new query `q_t` attends over the cached keys and values:

<p align="center">
  <strong>Attention(q<sub>t</sub>, K<sub>≤t</sub>, V<sub>≤t</sub>) = softmax((q<sub>t</sub>K<sub>≤t</sub><sup>T</sup>) / √d) V<sub>≤t</sub></strong>
</p>

Then the newly computed `K_t` and `V_t` are appended to the cache:

<p align="center">
  <strong>K<sub>≤t</sub> = [K<sub>&lt;t</sub> ; K<sub>t</sub>]</strong><br>
  <strong>V<sub>≤t</sub> = [V<sub>&lt;t</sub> ; V<sub>t</sub>]</strong>
</p>

The semicolon means concatenation along the sequence dimension.

This is the operational meaning of the KV cache:

> Store the past once, then reuse it at every future decode step.

---

### 5.3 KV cache size formula

The KV cache introduces one of the most important memory pressures in LLM serving.

A standard size formula is:

<p align="center">
  <strong>KV cache bytes = 2 × L × H<sub>kv</sub> × d<sub>head</sub> × T × b</strong>
</p>

Where:

| Symbol | Meaning |
|---|---|
| `L` | number of Transformer layers |
| `H_kv` | number of KV heads |
| `d_head` | dimension of each attention head |
| `T` | sequence length |
| `b` | bytes per element |
| `2` | one tensor for keys and one tensor for values |

This formula reveals the main scaling behavior:

<p align="center">
  <strong>KV cache size ∝ T</strong>
</p>

So as sequence length grows, KV cache memory grows linearly.

For a single request, longer context means a larger cache. For a serving system, the total KV cache footprint also scales with the number of active requests:

<p align="center">
  <strong>Total KV memory ≈ Σ<sub>r=1</sub><sup>R</sup> KV memory of request r</strong>
</p>

Here, `R` is the number of active requests.

This is one of the main reasons LLM serving is hard: even if the model weights fit on the GPU, the active KV caches may become the real memory bottleneck.

---

## 6. The two serving personas: latency and throughput

Any serving system must balance two competing goals.

### Latency

Latency measures how fast the system responds to an individual user.

The two most important latency metrics are:

| Metric | Meaning | Mainly affected by |
|---|---|---|
| TTFT | Time to first token | queueing + tokenization + prefill |
| TPOT / ITL | Time per output token / inter-token latency | decode |

A simplified view is:

<p align="center">
  <strong>TTFT ≈ queue time + tokenization time + prefill time</strong>
</p>

For the full response, total generation time can be approximated as:

<p align="center">
  <strong>Total latency ≈ TTFT + m × TPOT</strong>
</p>

Where `m` is the number of generated output tokens.

### Throughput

Throughput measures how much total work the system completes over time.

Common throughput metrics include:

<p align="center">
  <strong>tokens/sec = total generated tokens / wall-clock time</strong>
</p>

and:

<p align="center">
  <strong>requests/sec = completed requests / wall-clock time</strong>
</p>

Latency and throughput are often in tension:

> Low latency prefers immediate execution.  
> High throughput prefers larger or smarter batching.

That trade-off is a major theme in LLM serving systems.

---

## 7. Why batching is non-trivial

Batching sounds simple in theory: combine requests and run them together.

In practice, LLM serving is harder because:

- prompt lengths differ
- output lengths differ
- requests arrive continuously
- some sequences finish early
- new sequences arrive while older ones are still decoding

Naive batching can waste GPU capacity because finished requests leave empty slots, while new requests may be forced to wait.

Modern engines therefore need sophisticated schedulers rather than static batching.

This is why later topics in the handbook matter:

- continuous batching
- paged attention
- chunked prefill
- prefix caching
- KV cache quantization

---

## 8. A compact mental model

When reading about any serving optimization, ask:

1. Does it target **prefill** or **decode**?
2. Does it reduce **compute**, **memory movement**, or **memory footprint**?
3. Does it improve **TTFT**, **TPOT**, or **throughput**?
4. Does it affect the **KV cache**?
5. Does it help with **batching and scheduling**?

Those five questions are enough to organize most of the field.

---

## 9. Key takeaways

- LLM inference is forward-only execution with frozen weights.
- Generation is autoregressive:

<p align="center">
  <strong>P(y | x) = ∏<sub>t=1</sub><sup>m</sup> P(y<sub>t</sub> | x, y<sub>&lt;t</sub>)</strong>
</p>

- Prefill processes the prompt and usually behaves more like a compute-heavy phase.
- Decode generates tokens one by one and is often more memory-bandwidth-sensitive.
- The KV cache avoids recomputation and makes decoding efficient.
- KV cache memory grows with sequence length:

<p align="center">
  <strong>KV cache size ∝ T</strong>
</p>

- TTFT is strongly shaped by prefill.
- TPOT is strongly shaped by decode.
- Production LLM serving is fundamentally a systems problem, not just a modeling problem.

---

## Next

The next two files make this chapter more concrete:

1. [`prefill-and-decode.md`](./prefill-and-decode.md)  
   A numerical breakdown of the two phases.

2. [`kv-cache-explained.md`](./kv-cache-explained.md)  
   A deeper explanation of attention, caching, and memory scaling.
