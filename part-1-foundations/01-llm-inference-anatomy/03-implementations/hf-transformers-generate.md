# HuggingFace Transformers: How `model.generate()` Works

This note walks through the HuggingFace Transformers `model.generate()` implementation to show how the theory from `01-theory/` maps to real code. The goal is not a complete API reference — it is to understand where prefill happens, how the KV cache is allocated and passed between steps, and why this implementation is simple but unsuitable for production serving.

All file references point to the Transformers repository. The structure is stable but line numbers change across versions — use the file paths to navigate, not line numbers.

**Key files:**
- `src/transformers/generation/utils.py` — the main generation loop
- `src/transformers/models/llama/modeling_llama.py` — Llama-specific forward pass and KV cache
- `src/transformers/cache_utils.py` — KV cache data structures

---

## 1. Entry point: `model.generate()`

Everything starts in `generation/utils.py`, in the `GenerationMixin.generate()` method.

The method does four things before any tokens are generated:

1. **Validates and prepares inputs** — tokenizes if needed, handles attention masks, sets generation config
2. **Selects a generation strategy** — greedy, beam search, sampling, etc.
3. **Calls the appropriate inner loop** — e.g. `_sample()`, `_greedy_search()`, `_beam_search()`
4. **Post-processes and returns** — decodes output token IDs back to text if `return_dict_in_generate=False`

For typical autoregressive generation with sampling or greedy decoding, the path goes:

```
generate()
  └── _sample()  or  _greedy_search()
        └── model.forward()  ← called once per token
```

The inner loop is where prefill and decode happen — but HuggingFace does not separate them explicitly. Both are just calls to `model.forward()`.

---

## 2. Where prefill happens

There is no function called `prefill()` in HuggingFace Transformers. Prefill is implicitly the **first call** to `model.forward()` inside the generation loop.

In `_greedy_search()` (or `_sample()`), the loop starts with the full input sequence:

```python
# First iteration: input_ids contains the full prompt
# model_kwargs["past_key_values"] is None
outputs = self(
    input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,  # None on first call
    use_cache=use_cache,
    **model_kwargs,
)
```

When `past_key_values=None`, the model processes all `N` prompt tokens together. This is prefill. The forward pass:

- Runs all `N` tokens through every Transformer layer
- Computes attention over all `N` positions in parallel
- Returns `past_key_values` — the initial KV cache — alongside the output logits

After the first call, the loop extracts the next token from the logits:

```python
next_token_logits = outputs.logits[:, -1, :]  # last position only
next_tokens = torch.argmax(next_token_logits, dim=-1)  # greedy
```

And appends it to the running sequence:

```python
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

---

## 3. How the KV cache is allocated

The KV cache in HuggingFace is returned from `model.forward()` as `past_key_values` — a tuple of tuples.

Structure:

```python
past_key_values = (
    (key_layer_0, value_layer_0),   # layer 0
    (key_layer_1, value_layer_1),   # layer 1
    ...
    (key_layer_L, value_layer_L),   # layer L-1
)
```

Each `key_layer_i` is a tensor of shape `(batch_size, num_kv_heads, seq_len, head_dim)`.

This is a standard Python tuple — not a pre-allocated fixed-size buffer. Every decode step, the model:

1. Computes new key and value tensors for the current token
2. Concatenates them onto the existing cache tensors along the sequence dimension
3. Returns the enlarged tuple as the new `past_key_values`

The concatenation in `modeling_llama.py` (inside the attention module):

```python
if past_key_value is not None:
    # Append new key/value to existing cache
    key_states = torch.cat([past_key_value[0], key_states], dim=2)
    value_states = torch.cat([past_key_value[1], value_states], dim=2)
```

This means a new tensor is allocated and the old one is discarded at every decode step. For a 512-token response, this is 512 allocations and deallocations of tensors that grow from `(batch, heads, 1, head_dim)` to `(batch, heads, N+512, head_dim)`.

---

## 4. How the KV cache is passed between steps

After the first forward pass, the generation loop passes `past_key_values` back into the next call:

```python
past_key_values = outputs.past_key_values

# Subsequent iterations: input_ids contains only the new token
outputs = self(
    next_tokens.unsqueeze(-1),      # shape: (batch, 1)
    past_key_values=past_key_values, # growing cache from previous steps
    use_cache=True,
    **model_kwargs,
)
```

On decode steps (after the first), `input_ids` contains only the single new token. The model uses `past_key_values` to attend over the full history without reprocessing it.

The attention mask is also updated to cover the full sequence length including cached tokens, so the model knows how many positions exist in the cache.

---

## 5. The `DynamicCache` class

In recent versions of Transformers (4.36+), `past_key_values` is wrapped in a `DynamicCache` object defined in `cache_utils.py`.

`DynamicCache` provides a slightly cleaner interface:

```python
class DynamicCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(self, key_states, value_states, layer_idx):
        # Concatenate new key/values onto existing cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

The mechanism is identical to the raw tuple approach — concatenation at every step — but encapsulated in a class. The memory behavior is the same: growing tensors, repeated allocation.

---

## 6. Why this implementation is simple but slow for serving

The HuggingFace implementation is correct and readable. It is the right implementation for single-request inference, fine-tuning evaluation, and experimentation. It is not suitable for production serving at scale for several reasons.

**No continuous batching**

The generation loop runs a fixed batch to completion before accepting new requests. If a batch of 8 requests is running and 4 finish early, the remaining 4 continue running alone with no new requests filling the empty slots. GPU utilization drops proportionally.

**No paged memory management**

KV cache tensors grow by concatenation. Each decode step allocates a new, slightly larger tensor. For a 1024-token response, the KV cache for that request is allocated and deallocated 1024 times, with each new tensor slightly larger than the last.

There is no pre-allocation of a fixed memory pool. There is no concept of KV blocks or pages. Memory fragmentation accumulates over many requests.

**No memory-aware scheduling**

The scheduler does not know how much KV cache memory a request will consume. It cannot evict low-priority requests to free memory for high-priority ones. It cannot predict when a request will finish. It cannot make admission control decisions based on available KV memory.

**No overlap between prefill and decode**

Prefill and decode run sequentially in the same loop. A long prefill blocks all ongoing decode steps. There is no interleaving, no chunking, no priority separation between the two phases.

**Single-process, single-GPU by default**

The HuggingFace implementation does not natively support multi-GPU tensor parallelism at the serving level. Serving large models requires manual setup with `device_map` or integration with external frameworks.

---

## 7. Comparing to production engines

| Property | HuggingFace `generate()` | vLLM / SGLang |
|---|---|---|
| KV cache allocation | Dynamic concatenation | Pre-allocated paged blocks |
| Batching | Static, fixed at start | Continuous, iteration-level |
| Memory management | Python GC | Custom allocator |
| Prefill/decode scheduling | Sequential, no separation | Separated, configurable |
| Multi-GPU support | Manual / limited | Native tensor parallelism |
| Throughput | Low | High (10-40× at scale) |
| Code complexity | Low | High |
| Suitable for | Research, single requests | Production serving |

---

## 8. What to read in the source

If you want to trace through the code yourself, this is the recommended path:

1. **`generation/utils.py` → `generate()`** — Read the docstring and the strategy selection logic. Understand that prefill and decode are not separated here.

2. **`generation/utils.py` → `_greedy_search()` or `_sample()`** — Find the inner `while` loop. Trace how `input_ids` and `past_key_values` are updated each iteration.

3. **`cache_utils.py` → `DynamicCache.update()`** — Read the concatenation logic. Note that a new tensor is created every step.

4. **`models/llama/modeling_llama.py` → `LlamaAttention.forward()`** — Find where `past_key_value` is consumed and where new key/value tensors are computed and concatenated. This is the layer-level KV cache operation.

5. **`models/llama/modeling_llama.py` → `LlamaModel.forward()`** — See how `past_key_values` is passed through all layers and how the output `past_key_values` tuple is assembled.

Reading these five entry points gives you the complete picture of how a single request flows through HuggingFace Transformers from `generate()` to the final token.

---

## 9. Key takeaways

- HuggingFace does not separate prefill and decode — both are calls to `model.forward()`.
- Prefill is the first forward call with the full prompt and `past_key_values=None`.
- Decode is every subsequent call with a single token and the growing `past_key_values`.
- The KV cache is a tuple of tensors that grows by concatenation at every step.
- `DynamicCache` wraps this in a class but uses the same concatenation mechanism.
- This implementation is correct but allocates memory dynamically at every decode step.
- There is no continuous batching, paged memory, or prefill/decode scheduling.
- Production serving engines (vLLM, SGLang) replace almost every part of this pipeline while keeping the same model weights and forward pass logic.