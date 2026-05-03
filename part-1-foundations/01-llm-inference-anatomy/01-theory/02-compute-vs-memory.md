# Compute vs Memory: The Real Bottleneck in LLM Inference

> 📖 Recommended order: You should read this after `01-overview.md`.

This chapter explains one of the most important ideas in LLM systems:

> Why GPUs are not always fully utilized during inference.

You already know:

- Prefill is compute-heavy  
- Decode is memory-heavy  

Now we answer the deeper question:

> What does that actually mean at the hardware level?

---

## 1. The simplest mental model

A GPU does two things:

1. Compute → multiply, add, apply functions  
2. Move data → load weights, activations, KV cache  

<p align="center">
  <strong>Performance = compute speed AND data movement speed</strong>
</p>

If either is slow → the whole system is slow.

---

## 2. GPU architecture (simplified)

We can think of a GPU as two main parts:

| Component | Role |
|---|---|
| Compute units (SMs) | Do math (FLOPs) |
| Memory (HBM) | Store and load data |

<p align="center">
  <strong>Compute (fast) ↔ Memory (slower)</strong>
</p>

Important fact:

> GPUs can compute faster than they can load data.

---

## 3. Compute-bound vs Memory-bound

Every workload falls into one of these categories:

### Compute-bound

The GPU is busy doing math.

Compute time > Memory wait time

- GPU cores fully utilized  
- Bottleneck = FLOPs  

---

### Memory-bound

The GPU is waiting for data.

Memory wait time > Compute time

- GPU cores partially idle  
- Bottleneck = memory bandwidth  

---

## 4. Roofline intuition

<p align="center">
  <strong>Performance = min(compute limit, memory limit)</strong>
</p>

---

## 5. Arithmetic intensity

<p align="center">
  <strong>Arithmetic intensity = FLOPs / bytes moved</strong>
</p>

- High → compute-bound  
- Low → memory-bound  

---

## 6. Connection to LLM inference

### Prefill

- Many tokens processed together  
- Large matrix multiplications  
- High compute reuse  

<p align="center">
  <strong>Compute-bound</strong>
</p>

---

### Decode

- One token at a time  
- Large KV cache reads  
- Very small compute  

<p align="center">
  <strong>Memory-bound</strong>
</p>

---

## 7. Why decode underutilizes GPU

- Loads weights + KV cache  
- Does small compute  
- Waits for memory  

<p align="center">
  <strong>GPU waits for data</strong>
</p>

---

## 8. Ferrari analogy

> GPU = Ferrari  
> Prefill = highway  
> Decode = city traffic  

---

## 9. Why batching helps

- Increases compute per memory load  
- Improves utilization  

---

## 10. KV cache impact

<p align="center">
  <strong>KV size ∝ sequence length</strong>
</p>

Longer context → more memory reads → slower decode  

---

## 11. System implications

| Phase | Bottleneck | Optimization |
|---|---|---|
| Prefill | Compute | Kernel efficiency |
| Decode | Memory | KV cache optimization |

---

## 12. Key takeaway

> LLM inference is often limited by memory movement, not compute.
