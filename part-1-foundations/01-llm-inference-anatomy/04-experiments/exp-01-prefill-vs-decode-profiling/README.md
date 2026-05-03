# Experiment 01 — Prefill vs Decode Profiling

## Hypothesis

Prefill latency scales roughly linearly with prompt length because it is compute-bound — more tokens means more parallel matrix multiplications. Decode latency per token remains roughly constant regardless of prompt length because it is memory-bandwidth-bound — the bottleneck is loading model weights, not the size of the KV cache being read.

GPU utilization should be high during prefill and low during single-request decode.

---

## Setup

**Hardware:** Single A100 40GB (or equivalent — H100, A6000)  
**Model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Precision:** BF16  
**Frameworks:** HuggingFace Transformers 4.44+, vLLM 0.6+, PyTorch 2.4+

Install dependencies:

```bash
pip install transformers torch accelerate vllm
```

---

## Files

| File | Purpose |
|---|---|
| `profile_hf.py` | Measures TTFT and per-token decode latency using HF Transformers + `torch.profiler` |
| `profile_vllm.py` | Measures the same metrics using vLLM's internal timing |
| `results.json` | Raw measurements (generated after running scripts) |
| `analysis.ipynb` | Plots and interpretation (run after results are collected) |

---

## What to run

```bash
# HuggingFace measurements
python profile_hf.py --output results_hf.json

# vLLM measurements
python profile_vllm.py --output results_vllm.json
```

Merge outputs into `results.json` before running the notebook.

---

## What to measure

| Metric | Expected result | Why |
|---|---|---|
| Prefill latency vs prompt length | Roughly linear | Compute-bound: FLOPs ∝ N |
| Per-token decode latency vs prompt length | Roughly flat | Memory-bound: bottleneck is weight loading, not KV size |
| Per-token decode latency vs output length | Slowly increasing | Longer KV cache reads at each step |
| GPU utilization during prefill | High (>70%) | Many tokens → large matrix multiplications |
| GPU utilization during decode (batch=1) | Low (<10%) | Matrix-vector multiply, memory-bound |

---

## Prompt lengths tested

`[64, 128, 256, 512, 1024, 2048, 4096]` tokens

Output lengths: fixed at 128 tokens for latency measurements, variable for the decode-vs-output-length sweep.

---

## Expected plots (analysis.ipynb)

1. **Prefill time vs prompt length** — should be a roughly straight line
2. **Decode TPOT vs prompt length** — should be nearly flat (prompt length should not affect decode speed much)
3. **Decode TPOT vs output position** — slight increase as KV cache grows
4. **TTFT decomposition** — how much of total latency is prefill vs queue vs tokenization
5. **GPU utilization timeline** — high spike during prefill, low plateau during decode