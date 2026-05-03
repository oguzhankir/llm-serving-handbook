# vLLM: How the Inference Pipeline Is Structured Differently

This note gives a high-level view of how vLLM structures the inference pipeline compared to HuggingFace Transformers. The goal is to show concretely how the problems identified in `hf-transformers-generate.md` are solved — not to document the full vLLM codebase.

vLLM internals (scheduler algorithm, PagedAttention kernel, tensor parallelism) are covered in depth in Part 2 and Part 3. This note focuses on the request lifecycle: how a request enters vLLM, how prefill and decode are separated, and how the KV cache is managed differently.

**Key files:**
- `vllm/engine/llm_engine.py` — the main engine, coordinates all components
- `vllm/core/scheduler.py` — decides which requests run and in what order
- `vllm/worker/worker.py` — executes model forward passes on GPU
- `vllm/attention/backends/` — attention kernels including PagedAttention
- `vllm/sequence.py` — data structures for sequences and KV block tables

---

## 1. The architecture: three layers

vLLM separates concerns into three distinct layers that HuggingFace conflates into one loop:

```
┌─────────────────────────────────────┐
│           LLMEngine                 │  ← orchestration, request lifecycle
├─────────────────────────────────────┤
│           Scheduler                 │  ← decides what runs each step
├─────────────────────────────────────┤
│           Worker(s)                 │  ← executes forward pass on GPU
└─────────────────────────────────────┘
```

`LLMEngine` receives requests, manages their lifecycle from arrival to completion, and coordinates the scheduler and workers.

`Scheduler` runs before every forward pass and decides: which sequences are in the running batch, which new prompts to prefill, which sequences to preempt if memory is tight.

`Worker` executes the actual model forward pass. For multi-GPU setups, there is one worker per GPU. Workers handle KV cache allocation and the PagedAttention kernel.

---

## 2. How a request enters the system

A request arrives via `LLMEngine.add_request()`.

vLLM immediately creates a `Sequence` object and adds it to the scheduler's waiting queue. A `Sequence` tracks:

- The token IDs (prompt + generated so far)
- The current status: `WAITING`, `RUNNING`, or `SWAPPED`
- A **block table** — a mapping from logical KV positions to physical memory blocks

Nothing is allocated yet. No GPU memory is touched at this point.

```python
# Simplified from llm_engine.py
def add_request(self, request_id, prompt, sampling_params):
    seq = Sequence(seq_id, prompt_token_ids, block_size)
    seq_group = SequenceGroup(request_id, [seq], sampling_params)
    self.scheduler.add_seq_group(seq_group)
```

The sequence sits in `scheduler.waiting` until the scheduler decides to run it.

---

## 3. The scheduler: what runs each step

Every forward pass starts with `scheduler.schedule()`. This is the key difference from HuggingFace.

The scheduler runs a policy each iteration that produces a `SchedulerOutputs` object containing:

- `scheduled_seq_groups` — the sequences that will run this step
- `num_prefill_groups` — how many of those are in prefill (first pass)
- `num_batched_tokens` — total tokens being processed
- `blocks_to_swap_in/out` — KV blocks to move between GPU and CPU

The scheduler separates sequences into two categories:

**Prefill sequences** — sequences in their first forward pass. The full prompt is processed. The scheduler tries to fit as many new prompts as possible without exceeding memory limits.

**Decode sequences** — sequences that have already completed prefill and are generating tokens. Each contributes exactly one token to the batch this step.

A typical batch might look like:

```
Step N:
  Prefill:  [request_7 (512 tokens)]
  Decode:   [request_1 (1 token), request_2 (1 token), request_3 (1 token), ...]
  Total:    512 + N_decode tokens processed together
```

This is continuous batching: new requests join mid-flight, and prefill and decode are interleaved within the same forward pass.

---

## 4. KV cache: blocks instead of growing tensors

This is the most fundamental difference from HuggingFace.

vLLM pre-allocates a fixed pool of KV blocks at startup, determined by available GPU memory after loading the model weights. Each block holds KV tensors for a fixed number of tokens (default: 16 tokens per block).

```
GPU memory pool:
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │  ← physical KV blocks
└────┴────┴────┴────┴────┴────┴────┴────┘
```

Each sequence has a **block table** — a mapping from logical positions to physical blocks:

```
Sequence 1 block table:  [B2, B5, B7]   ← holds tokens 0-47 across 3 blocks
Sequence 2 block table:  [B0, B3]        ← holds tokens 0-31 across 2 blocks
```

Blocks are allocated from the pool as sequences grow, and returned to the pool when sequences finish. There is no concatenation, no growing tensors, no per-step allocation.

The `BlockSpaceManager` in `vllm/core/block_manager.py` manages this pool. It tracks which blocks are free, allocates blocks for new sequences, and handles block sharing for prefix caching.

---

## 5. How prefill and decode execute differently

In the Worker, the forward pass receives a `ModelInput` that contains both prefill and decode sequences in the same batch.

For **prefill sequences**, the full prompt token IDs are passed. The attention kernel processes all prompt positions and writes the computed KV tensors into the allocated blocks.

For **decode sequences**, only the single new token ID is passed. The attention kernel reads from the block table to locate all cached KV blocks for that sequence, performs attention over the full history, and writes the new KV for the current token.

Both happen in the same `model.forward()` call. The PagedAttention kernel handles the mixed batch — some positions doing full attention over a prompt, others doing single-token attention with a block table lookup.

This is the core insight:

> In vLLM, prefill and decode are not separate code paths — they are handled by the same attention kernel with different inputs. The separation exists in the scheduler, not in the forward pass.

---

## 6. What PagedAttention changes at the kernel level

Standard attention in HuggingFace assumes contiguous KV tensors:

```python
# Standard attention: K and V are contiguous (batch, heads, seq_len, head_dim)
attn_output = flash_attn(query, key_cache, value_cache)
```

PagedAttention replaces this with a kernel that reads KV from non-contiguous blocks via the block table:

```python
# PagedAttention: K and V are in non-contiguous blocks
# block_tables maps logical positions to physical block indices
attn_output = paged_attn(query, key_cache_pool, value_cache_pool, block_tables)
```

The kernel walks the block table to gather KV tensors from wherever they happen to be in the pool. From the attention computation's perspective, the result is identical. From the memory management perspective, sequences no longer need contiguous memory regions.

This is why vLLM can run many more concurrent requests than HuggingFace on the same hardware: no large contiguous allocations, no fragmentation, no over-reservation.

---

## 7. A request from arrival to completion

Tracing a single request through vLLM end to end:

```
1. add_request()
   └── Create Sequence, add to scheduler.waiting

2. scheduler.schedule()  [step N]
   └── Sequence moves from waiting → running
   └── BlockSpaceManager allocates initial KV blocks
   └── Sequence added to prefill batch

3. worker.execute_model()  [step N]
   └── Forward pass with prompt tokens
   └── PagedAttention writes KV for all prompt tokens into blocks
   └── Returns logits → sample first output token
   └── Sequence status: prefill complete, first token generated

4. scheduler.schedule()  [step N+1]
   └── Sequence is now a decode sequence
   └── One new block allocated if needed
   └── Added to decode batch alongside other running sequences

5. worker.execute_model()  [step N+1 ... N+M]
   └── Each step: single token input, block table read, new KV written
   └── Token sampled and appended

6. Sequence hits stop condition (EOS token or max_tokens)
   └── scheduler marks sequence as finished
   └── BlockSpaceManager frees all blocks back to pool
   └── Result returned via AsyncLLMEngine
```

---

## 8. What vLLM does not change

vLLM replaces the serving infrastructure around the model, not the model itself.

The same model weights, the same Transformer layers, the same attention computation (mathematically) are used. The differences are all in:

- how memory is pre-allocated and managed (block pool instead of dynamic tensors)
- how batches are formed (continuous, iteration-level instead of static)
- how the attention kernel reads KV (block table lookup instead of contiguous tensor)
- how scheduling decisions are made (memory-aware, prefill/decode aware)

From the model's perspective, it is still doing the same forward pass.

---

## 9. Key takeaways

- vLLM separates orchestration (LLMEngine), scheduling (Scheduler), and execution (Worker) into distinct layers.
- Requests are queued as `Sequence` objects before any GPU memory is touched.
- The scheduler runs before every forward pass and decides which sequences run, in what mode (prefill or decode).
- KV cache is a pre-allocated block pool, not dynamically growing tensors. Each block holds 16 tokens by default.
- Sequences have block tables that map logical token positions to physical memory blocks.
- Prefill and decode run in the same forward pass — the separation is in the scheduler and the inputs, not in the model code.
- PagedAttention reads KV from non-contiguous blocks via block table lookup — the mathematical result is identical to standard attention.
- vLLM does not change model weights or the Transformer computation — it changes everything around it.
- Detailed vLLM scheduler and PagedAttention internals are covered in Part 2, Topics 07 and 08.