# Experiment 02 — KV Cache Memory Measurement

## Hypothesis

The KV cache memory consumed during generation matches the theoretical formula:

```
KV bytes = 2 × L × H_kv × d_head × T × b
```

For Llama 3.1 8B (L=32, H_kv=8, d_head=128, b=2 bytes for BF16), the cache should grow at approximately 128 KiB per generated token. At some sequence length, the cumulative KV cache will exceed the model weight memory (~16 GB).

---

## Setup

**Hardware:** Single A100 40GB (or equivalent)  
**Models:** `meta-llama/Llama-3.1-8B-Instruct` (primary), `meta-llama/Llama-3.1-70B-Instruct` (optional, requires multi-GPU)  
**Precision:** BF16

```bash
pip install transformers torch accelerate
```

---

## Files

| File | Purpose |
|---|---|
| `measure_kv_cache.py` | Generates tokens and measures GPU memory at each step |
| `theoretical_calc.py` | Computes expected KV cache sizes from model config |
| `results.json` | Raw measurements (generated after running scripts) |
| `analysis.ipynb` | Plots: measured vs theoretical, crossover point |

---

## What to run

```bash
# Compute theoretical values
python theoretical_calc.py

# Measure actual GPU memory
python measure_kv_cache.py --output results_kv.json

# Merge and analyze
# Open analysis.ipynb
```

---

## What to measure

| Measurement | Expected result |
|---|---|
| GPU memory after model load, before generation | ~16 GB for 8B BF16 |
| GPU memory after prefill (512 tokens) | +~64 MB (KV for 512 tokens) |
| Memory increment per generated token | ~128 KiB (one token across all layers) |
| Sequence length where KV cache = model weights | ~125,000 tokens for 8B GQA |
| Same crossover for MHA (32 KV heads) | ~31,250 tokens |

---

## Expected plots (analysis.ipynb)

1. **Measured vs theoretical KV cache size** — should closely match, validating the formula
2. **KV cache growth over generation steps** — linear slope
3. **KV cache as fraction of model weights** — crossover point highlighted
4. **8B vs 70B comparison** — larger model, larger cache, different crossover point