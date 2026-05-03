"""
measure_kv_cache.py

Measures actual GPU memory consumption as the KV cache grows during generation.
Samples memory at each decode step to track growth precisely.

Usage:
    python measure_kv_cache.py --output results_kv.json
    python measure_kv_cache.py --model meta-llama/Llama-3.1-8B-Instruct --max-new-tokens 512
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPT_LENGTHS = [64, 256, 512, 1024, 2048]
MAX_NEW_TOKENS = 256
FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models have transformed natural language processing. "
    "GPU memory bandwidth is often the bottleneck during autoregressive decoding. "
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    step: int               # 0 = after prefill, 1+ = after each decode token
    sequence_length: int    # total tokens in sequence at this point
    allocated_gb: float     # torch.cuda.memory_allocated()
    reserved_gb: float      # torch.cuda.memory_reserved()
    kv_cache_gb: float      # estimated: allocated - baseline (after model load)


@dataclass
class ExperimentResult:
    model: str
    hardware: str
    dtype: str
    prompt_length: int
    max_new_tokens: int
    model_weight_gb: float
    snapshots: list


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_memory_gb() -> tuple[float, float]:
    """Return (allocated_gb, reserved_gb) after CUDA sync."""
    torch.cuda.synchronize()
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )


def build_prompt(tokenizer, target_length: int) -> str:
    text = FILLER_TEXT * ((target_length // len(tokenizer.encode(FILLER_TEXT))) + 2)
    tokens = tokenizer.encode(text)[:target_length]
    return tokenizer.decode(tokens)


# ---------------------------------------------------------------------------
# Per-step memory measurement
# ---------------------------------------------------------------------------

class MemoryRecordingCriteria(StoppingCriteria):
    """
    Records GPU memory after each generated token.
    Stops generation after `max_steps` tokens.
    """

    def __init__(self, max_steps: int, baseline_gb: float):
        self.max_steps = max_steps
        self.baseline_gb = baseline_gb
        self.step = 0
        self.snapshots: list[MemorySnapshot] = []

    def __call__(self, input_ids, scores, **kwargs):
        torch.cuda.synchronize()
        allocated, reserved = get_memory_gb()
        seq_len = input_ids.shape[1]

        self.snapshots.append(MemorySnapshot(
            step=self.step,
            sequence_length=seq_len,
            allocated_gb=allocated,
            reserved_gb=reserved,
            kv_cache_gb=max(0.0, allocated - self.baseline_gb),
        ))

        self.step += 1
        return self.step >= self.max_steps


# ---------------------------------------------------------------------------
# Main measurement function
# ---------------------------------------------------------------------------

def measure_kv_growth(
    model,
    tokenizer,
    prompt_length: int,
    max_new_tokens: int,
    model_weight_gb: float,
) -> ExperimentResult:
    """
    Measure KV cache memory growth during generation for a given prompt length.
    """
    prompt = build_prompt(tokenizer, prompt_length)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    actual_prompt_length = inputs["input_ids"].shape[1]

    # Memory before generation (after model weights loaded)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated, _ = get_memory_gb()

    criteria = MemoryRecordingCriteria(
        max_steps=max_new_tokens,
        baseline_gb=baseline_allocated,
    )

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([criteria]),
        )

    return ExperimentResult(
        model=model.config._name_or_path,
        hardware=torch.cuda.get_device_name(0),
        dtype="bfloat16",
        prompt_length=actual_prompt_length,
        max_new_tokens=max_new_tokens,
        model_weight_gb=model_weight_gb,
        snapshots=[asdict(s) for s in criteria.snapshots],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--output", default="results_kv.json")
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=PROMPT_LENGTHS,
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    torch.cuda.synchronize()

    # Baseline: memory used by model weights alone
    model_weight_gb, _ = get_memory_gb()
    print(f"Model weights: {model_weight_gb:.2f} GB")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")

    all_results = []

    for prompt_length in args.prompt_lengths:
        print(f"\nPrompt length: {prompt_length} tokens, max_new_tokens: {args.max_new_tokens}")
        torch.cuda.empty_cache()

        result = measure_kv_growth(
            model,
            tokenizer,
            prompt_length,
            args.max_new_tokens,
            model_weight_gb,
        )

        if result.snapshots:
            final = result.snapshots[-1]
            print(f"  Final KV cache: {final['kv_cache_gb'] * 1024:.1f} MB")
            print(f"  Sequence length at end: {final['sequence_length']}")
            print(
                f"  KV per token (avg): "
                f"{final['kv_cache_gb'] * 1024 / max(args.max_new_tokens, 1):.3f} MB"
            )

        all_results.append(asdict(result))

    output = {
        "model": args.model,
        "model_weight_gb": model_weight_gb,
        "experiments": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()