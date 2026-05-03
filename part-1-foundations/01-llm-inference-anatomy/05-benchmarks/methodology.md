# Measurement Methodology

This file defines how metrics are measured across all experiments in Topic 01. Consistent methodology makes results reproducible and comparable across runs, hardware, and frameworks.

---

## TTFT — Time to First Token

**Definition:** Wall-clock time from the moment the model receives the tokenized input to the moment the first output token is available.

**How it is measured:**

```python
torch.cuda.synchronize()
t_start = time.perf_counter()

output = model.generate(**inputs, max_new_tokens=1)

torch.cuda.synchronize()
t_end = time.perf_counter()

ttft_ms = (t_end - t_start) * 1000
```

`max_new_tokens=1` isolates prefill from the decode loop. `torch.cuda.synchronize()` ensures the GPU has finished work before recording the timestamp — without this, `time.perf_counter()` measures CPU submission time, not actual GPU completion.

**What TTFT includes:** tokenization (if not pre-tokenized), queue time, prefill compute, first token sampling.

**What TTFT excludes:** subsequent decode steps, detokenization.

---

## TPOT — Time Per Output Token

**Definition:** Mean inter-token latency during the decode phase, excluding the first token (which includes prefill cost).

**How it is measured:**

A `StoppingCriteria` callback records a timestamp after each token is generated:

```python
class TimingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        torch.cuda.synchronize()
        token_times.append(time.perf_counter() * 1000)
        return len(token_times) >= output_length
```

Inter-token latencies are computed from consecutive timestamps:

```python
inter_token_latencies = [
    token_times[i] - token_times[i - 1]
    for i in range(1, len(token_times))
]
tpot_mean = sum(inter_token_latencies) / len(inter_token_latencies)
```

The first token is excluded because `token_times[0]` captures the combined prefill + first decode step. Starting from index 1 isolates pure decode latency.

**Reported statistics:** mean, P50, P95 across all inter-token intervals in a run.

---

## Throughput

**Definition:** Total output tokens generated divided by total wall-clock time.

**How it is measured:**

```python
t_start = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=N)
t_end = time.perf_counter()

total_tokens = sum(len(o.token_ids) for o in outputs)
throughput = total_tokens / (t_end - t_start)
```

For batched measurements, all requests in the batch are counted together.

**Note:** Throughput and latency are in tension. Single-request throughput is not meaningful — throughput is only a useful metric when measuring concurrent load.

---

## GPU Memory

**Two complementary methods:**

### `torch.cuda.memory_allocated()`

Reports PyTorch's view of allocated GPU memory. Used for tracking KV cache growth during generation.

```python
torch.cuda.synchronize()
allocated_gb = torch.cuda.memory_allocated() / 1e9
```

**Limitation:** Does not include memory allocated outside PyTorch (e.g. CUDA context, driver overhead, other processes).

### `nvidia-smi`

Reports total GPU memory in use at the process level. Used for validating that the model loaded correctly and no memory leaks exist.

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Limitation:** Includes PyTorch's memory cache (`torch.cuda.memory_reserved()`), which may be higher than actually allocated memory.

**Preferred approach:** Use `torch.cuda.memory_allocated()` for fine-grained experiment measurements. Use `nvidia-smi` for sanity checks before and after loading the model.

---

## Warmup Runs

GPU kernels are JIT-compiled on first execution. The first 1-2 forward passes are always slower due to compilation and caching overhead.

**Policy:** All experiments run 2 warmup passes before recording measurements. Warmup results are discarded.

```python
WARMUP_RUNS = 2
MEASUREMENT_RUNS = 3
```

**Warmup prompt:** A short fixed prompt (64 tokens) is used for warmup regardless of the prompt lengths being measured. This avoids inflating the warmup cost for short-prompt experiments.

---

## Number of Repetitions and Variance

Each measurement is repeated 3 times. Results are averaged unless stated otherwise.

**Reported statistics:**

- TTFT: mean across 3 runs
- TPOT: mean and P95 across all inter-token intervals from all runs combined
- GPU memory: single measurement after the stable region is reached

**Variance is expected to be low** for these workloads on a dedicated GPU. If P95 TPOT is more than 20% above mean, it indicates interference from other processes or thermal throttling — check `nvidia-smi` and re-run.

---

## Hardware and Software Versions

All measurements in this topic were collected on:


| Component    | Value                   |
| ------------ | ----------------------- |
| GPU          | NVIDIA A100-SXM4-40GB   |
| CUDA         | 12.8                    |
| PyTorch      | 2.4+                    |
| Transformers | 4.44.0                  |
| vLLM         | 0.8.5                   |
| Model        | meta-llama/Llama-3.1-8B |
| Precision    | BF16                    |


Results may differ on other hardware. The qualitative pattern (prefill linear, decode flat) holds across GPU generations, but absolute numbers depend on HBM bandwidth and compute throughput of the specific GPU.

---

## Reproducibility

To reproduce the measurements in this topic:

```bash
cd part-1-foundations/01-llm-inference-anatomy/04-experiments
pip install -r requirements.txt

huggingface-cli login  # requires access to meta-llama/Llama-3.1-8B

# exp-01
cd exp-01-prefill-vs-decode-profiling
python profile_hf.py --model meta-llama/Llama-3.1-8B --output results_hf.json
python profile_vllm.py --model meta-llama/Llama-3.1-8B --output results_vllm.json

# exp-02
cd ../exp-02-kv-cache-memory-measurement
python theoretical_calc.py
python measure_kv_cache.py --model meta-llama/Llama-3.1-8B --output results_kv.json
```

Raw results are stored in `numbers/` as JSON files.