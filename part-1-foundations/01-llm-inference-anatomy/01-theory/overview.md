# The Anatomy of LLM Inference

This chapter answers one question from a systems perspective:

> What actually happens, step by step, when you send a prompt to an LLM and receive a response back?

At first, LLM inference looks simple. A user sends text, the model returns text. But inside a production serving system, that request passes through tokenization, GPU execution, KV cache allocation, scheduling, decoding, detokenization, and streaming.

This chapter builds the mental model from the ground up. We start with the simplest view of inference, then gradually move toward the systems constraints that dominate real LLM serving.

---

## 1. The simplest view: text in, text out

The most basic view of inference is this:

<p align="center">
  <strong>prompt → model → response</strong>
</p>

The user provides a prompt, and the model generates a response. For example:

```text
Prompt:
Explain why decode is memory-bound in LLM inference.

Response:
Decode is memory-bound because each generation step must read model weights and cached key/value tensors while producing only a small amount of new computation...
```

This view is useful, but incomplete. The model does not actually see words or sentences. It sees **tokens**.

So a slightly better view is:

<p align="center">
  <strong>raw text → tokens → model → output tokens → text</strong>
</p>

And the full serving view is more detailed still.

---

## 2. The full request lifecycle

A single inference request follows the lifecycle below.

<p align="center">
  <img src="../../assets/01-llm-inference-anatomy/01-theory/overview/llm-inference-pipeline.png" alt="End-to-end LLM inference pipeline" width="900">
</p>

<p align="center">
  <em>Figure 1. End-to-end lifecycle of an LLM inference request, from raw prompt to streamed response.</em>
</p>

A request moves through these stages:

| Stage                | What happens                        | Why it matters                                                |
| -------------------- | ----------------------------------- | ------------------------------------------------------------- |
| Raw prompt           | The user sends text                 | This is the human-facing input                                |
| Tokenization         | Text becomes token IDs              | The model operates on tokens, not raw text                    |
| Input token IDs      | Tokens are prepared for the model   | This is the actual model input                                |
| Prefill              | The prompt is processed in parallel | This builds the initial KV cache and produces the first token |
| KV cache initialized | Past keys and values are stored     | This makes future decode steps efficient                      |
| First output token   | The first generated token appears   | This determines time-to-first-token                           |
| Decode loop          | One token is generated at a time    | This dominates long response generation                       |
| Detokenization       | Output token IDs become text        | Needed for human-readable output                              |
| Streamed response    | Text is sent back incrementally     | Improves perceived latency                                    |

A useful high-level mental model is:

> **Prefill** determines how quickly generation can begin.
> **Decode** determines how quickly the rest of the response arrives.

---

## 3. What inference means mathematically

Let the input prompt be represented as a sequence of tokens:

<p align="center">
  <strong>x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)</strong>
</p>

Let the generated response be:

<p align="center">
  <strong>y = (y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>m</sub>)</strong>
</p>

The model parameters are fixed during inference:

<p align="center">
  <strong>θ = fixed model weights</strong>
</p>

A decoder-only LLM generates the response **autoregressively**. This means it predicts one token at a time:

<p align="center">
  <strong>P(y | x) = ∏<sub>t=1</sub><sup>m</sup> P(y<sub>t</sub> | x, y<sub>&lt;t</sub>)</strong>
</p>

Read this as:

> The full response is built from many next-token predictions.

At generation step `t`, the model predicts token `y_t` using:

* the original prompt `x`
* all previously generated tokens `y_<t`

This is the key reason generation is sequential. The model cannot generate token `y_t` before token `y_(t-1)` exists.

---

## 4. Inference vs. training

Training and inference both run Transformer layers, but they solve different problems.

Training updates the model. Inference serves the model.

| Aspect              | Training            | Inference                               |
| ------------------- | ------------------- | --------------------------------------- |
| Model weights       | Updated             | Frozen                                  |
| Backpropagation     | Yes                 | No                                      |
| Gradients           | Required            | Not used                                |
| Optimizer state     | Required            | Not used                                |
| Execution pattern   | Forward + backward  | Forward only                            |
| Main goal           | Learn parameters    | Generate tokens efficiently             |
| Main system concern | Training throughput | Latency, throughput, memory, scheduling |

During training, the system must store activations, compute gradients, and update weights. During inference, none of that happens. The model only performs forward passes using fixed weights.

A concise summary:

> Training is about learning model parameters.
> Inference is about moving tokens through fixed parameters as efficiently as possible.

This distinction matters because most serving bottlenecks are not learning bottlenecks. They are memory, scheduling, and token-generation bottlenecks.

---

## 5. The two phases that define LLM serving

The most important distinction in this handbook is the difference between **prefill** and **decode**.

<p align="center">
  <img src="../../assets/01-llm-inference-anatomy/01-theory/overview/prefill-vs-decode.png" alt="Prefill vs Decode" width="900">
</p>

<p align="center">
  <em>Figure 2. Prefill and decode have fundamentally different execution patterns, bottlenecks, and serving implications.</em>
</p>

Although both phases use the same model weights, they behave very differently.

---

## 6. Prefill: processing the prompt

During **prefill**, the model processes the entire input prompt.

If the prompt contains `n` tokens, the model computes hidden states for all `n` prompt positions. This phase is relatively parallel because all prompt tokens are already known.

Inside a self-attention layer, attention can be written as:

<p align="center">
  <strong>Attention(Q, K, V) = softmax((QK<sup>T</sup>) / √d) V</strong>
</p>

Where:

| Symbol | Meaning                     |
| ------ | --------------------------- |
| `Q`    | query matrix                |
| `K`    | key matrix                  |
| `V`    | value matrix                |
| `d`    | attention head dimension    |
| `K^T`  | transpose of the key matrix |

During prefill, the model computes `Q`, `K`, and `V` for the prompt tokens. The key and value tensors are stored in the KV cache so that future decode steps can reuse them.

Key properties of prefill:

| Property          | Prefill                        |
| ----------------- | ------------------------------ |
| Input             | Entire prompt                  |
| Token behavior    | Many tokens processed together |
| Parallelism       | High across prompt tokens      |
| Main role         | Build the initial KV cache     |
| Output            | First generated token          |
| Main user metric  | TTFT                           |
| Hardware tendency | More compute-oriented          |

A rough mental model is:

<p align="center">
  <strong>prefill work ∝ model size × prompt length</strong>
</p>

Longer prompts usually increase prefill time because more prompt tokens must be processed before the first output token can be produced.

Intuitively:

> Prefill is the phase where the model reads and digests the prompt.

---

## 7. Decode: generating the response

After prefill produces the first output token, the model enters the **decode** phase.

At decode step `t`, the model predicts:

<p align="center">
  <strong>P(y<sub>t</sub> | x, y<sub>&lt;t</sub>)</strong>
</p>

That means the next token depends on the original prompt and all previously generated tokens.

A simplified decode step is:

<p align="center">
  <strong>y<sub>t</sub> ~ P(next token | x, y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>t-1</sub>)</strong>
</p>

Then the generated token is appended to the sequence:

<p align="center">
  <strong>(y<sub>1</sub>, ..., y<sub>t-1</sub>) → (y<sub>1</sub>, ..., y<sub>t-1</sub>, y<sub>t</sub>)</strong>
</p>

The loop repeats until the model emits a stop token, reaches a maximum output length, or the server stops generation.

Key properties of decode:

| Property          | Decode                            |
| ----------------- | --------------------------------- |
| Input             | Previous token + KV cache         |
| Token behavior    | One token generated at a time     |
| Parallelism       | Sequential across generation time |
| Main role         | Read and extend the KV cache      |
| Output            | Subsequent generated tokens       |
| Main user metric  | TPOT / ITL                        |
| Hardware tendency | More memory-bandwidth-oriented    |

A useful comparison:

> Prefill processes a whole prompt once.
> Decode repeats a smaller step many times.

---

## 8. Why decode is often memory-bound

A GPU has enormous compute capability, but computation is only useful if data arrives fast enough.

During decode, each step produces only one new token per sequence, but the system still has to repeatedly access:

* model weights
* cached keys and values
* intermediate activations

A rough mental model is:

<p align="center">
  <strong>decode cost per token ≈ weight reads + KV cache reads + small compute step</strong>
</p>

This is why decode is often limited by memory bandwidth rather than raw FLOPs.

The important systems question is:

> Is this phase waiting for compute, or waiting for memory movement?

Many LLM serving optimizations exist because the answer is often different for prefill and decode.

| Phase   | Common bottleneck  | Intuition                                          |
| ------- | ------------------ | -------------------------------------------------- |
| Prefill | Compute throughput | Many tokens create large matrix operations         |
| Decode  | Memory bandwidth   | Each step reads a lot of data to produce one token |

---

## 9. KV cache: the central serving primitive

The KV cache is the mechanism that makes autoregressive decoding practical.

<p align="center">
  <img src="../../assets/01-llm-inference-anatomy/01-theory/overview/kv-cache-overview.png" alt="KV Cache Overview" width="900">
</p>

<p align="center">
  <em>Figure 3. The KV cache stores past keys and values so that decode can reuse them instead of recomputing them from scratch.</em>
</p>

Without a KV cache, every new decode step would need to recompute keys and values for all previous tokens. That would be extremely inefficient.

Instead, the model stores previous keys and values once, then reuses them at future decode steps.

At decode step `t`:

1. compute a new query `q_t` for the current token
2. attend over cached past keys and values
3. compute new `K_t` and `V_t`
4. append `K_t` and `V_t` to the cache

The basic trade-off is:

> KV cache saves compute by spending memory.

---

## 10. Conceptual KV cache math

At a high level, the current token representation `x_t` is projected into query, key, and value vectors:

<p align="center">
  <strong>q<sub>t</sub> = x<sub>t</sub>W<sub>Q</sub></strong><br>
  <strong>K<sub>t</sub> = x<sub>t</sub>W<sub>K</sub></strong><br>
  <strong>V<sub>t</sub> = x<sub>t</sub>W<sub>V</sub></strong>
</p>

The new query attends over cached keys and values:

<p align="center">
  <strong>Attention(q<sub>t</sub>, K<sub>≤t</sub>, V<sub>≤t</sub>) = softmax((q<sub>t</sub>K<sub>≤t</sub><sup>T</sup>) / √d) V<sub>≤t</sub></strong>
</p>

After the step finishes, the new key and value are appended:

<p align="center">
  <strong>K<sub>≤t</sub> = [K<sub>&lt;t</sub> ; K<sub>t</sub>]</strong><br>
  <strong>V<sub>≤t</sub> = [V<sub>&lt;t</sub> ; V<sub>t</sub>]</strong>
</p>

The semicolon means concatenation along the sequence dimension.

The operational meaning is simple:

> Store the past once, then reuse it at every future decode step.

---

## 11. KV cache memory growth

The KV cache introduces one of the most important memory pressures in LLM serving.

A standard size formula is:

<p align="center">
  <strong>KV cache bytes = 2 × L × H<sub>kv</sub> × d<sub>head</sub> × T × b</strong>
</p>

Where:

| Symbol   | Meaning                                       |
| -------- | --------------------------------------------- |
| `L`      | number of Transformer layers                  |
| `H_kv`   | number of KV heads                            |
| `d_head` | dimension of each attention head              |
| `T`      | sequence length                               |
| `b`      | bytes per element                             |
| `2`      | one tensor for keys and one tensor for values |

This formula shows the most important scaling rule:

<p align="center">
  <strong>KV cache size ∝ T</strong>
</p>

As sequence length grows, KV cache memory grows linearly.

For a serving system, total KV memory also grows with the number of active requests:

<p align="center">
  <strong>Total KV memory ≈ Σ<sub>r=1</sub><sup>R</sup> KV memory of request r</strong>
</p>

Here, `R` is the number of active requests.

This is one of the main reasons LLM serving is hard. Even if the model weights fit on the GPU, the active KV caches may become the real memory bottleneck.

---

## 12. Latency: TTFT and TPOT

From a user's perspective, latency is not one number. Two metrics matter most.

| Metric     | Full name                                   | Meaning                                        | Mainly affected by              |
| ---------- | ------------------------------------------- | ---------------------------------------------- | ------------------------------- |
| TTFT       | Time to first token                         | How long until generation starts               | queueing, tokenization, prefill |
| TPOT / ITL | Time per output token / inter-token latency | How fast tokens arrive after generation starts | decode                          |

A simplified latency decomposition is:

<p align="center">
  <strong>TTFT ≈ queue time + tokenization time + prefill time</strong>
</p>

For a response with `m` generated tokens:

<p align="center">
  <strong>total latency ≈ TTFT + m × TPOT</strong>
</p>

This explains why both phases matter. A system with slow prefill feels unresponsive. A system with slow decode feels sluggish during long answers.

---

## 13. Throughput: the system-level view

Throughput measures how much work the serving system completes over time.

Common throughput metrics include:

<p align="center">
  <strong>tokens/sec = total generated tokens / wall-clock time</strong>
</p>

<p align="center">
  <strong>requests/sec = completed requests / wall-clock time</strong>
</p>

Latency and throughput are often in tension:

> Low latency prefers immediate execution.
> High throughput prefers batching and high GPU utilization.

This trade-off is why LLM serving engines need scheduling logic. They are not just running a model. They are deciding which requests should run, when they should run, and how GPU memory should be shared.

---

## 14. Why batching is non-trivial

Batching sounds simple: combine requests and run them together.

In practice, LLM serving is harder because:

* prompt lengths differ
* output lengths differ
* requests arrive continuously
* some sequences finish early
* new sequences arrive while older ones are still decoding
* KV cache memory grows dynamically

Naive static batching wastes capacity. Finished requests leave empty slots, while new requests may be forced to wait.

Modern inference engines therefore use more advanced scheduling and memory management techniques, such as:

* continuous batching
* paged KV cache allocation
* chunked prefill
* prefix caching
* KV cache quantization

Each of these techniques becomes easier to understand once the prefill/decode/KV-cache mental model is clear.

---

## 15. The core mental checklist

When reading about any LLM serving optimization, ask these questions:

1. Does it target **prefill**, **decode**, or both?
2. Does it reduce **compute**, **memory movement**, or **memory footprint**?
3. Does it improve **TTFT**, **TPOT**, or **throughput**?
4. Does it change how the **KV cache** is stored or accessed?
5. Does it improve **batching** or **scheduling**?
6. Does it preserve model quality?
7. Does it increase implementation complexity?

These questions turn LLM serving from a collection of tricks into a structured systems problem.

---

## 16. Key takeaways

* LLM inference is forward-only execution with frozen weights.
* The model operates on tokens, not raw text.
* Decoder-only LLMs generate autoregressively.
* Prefill processes the prompt and builds the initial KV cache.
* Decode generates one token at a time.
* Prefill is usually more compute-oriented.
* Decode is often more memory-bandwidth-oriented.
* The KV cache avoids recomputation but consumes GPU memory.
* KV cache memory grows with sequence length and active requests.
* TTFT is strongly shaped by prefill.
* TPOT is strongly shaped by decode.
* Throughput depends heavily on batching, scheduling, and memory management.
* Production LLM serving is a systems problem, not just a modeling problem.

---

## Next

The next two files make this chapter more concrete:

1. [`prefill-and-decode.md`](./prefill-and-decode.md)
   A numerical breakdown of prefill and decode.

2. [`kv-cache-explained.md`](./kv-cache-explained.md)
   A deeper explanation of attention, caching, and memory scaling.
