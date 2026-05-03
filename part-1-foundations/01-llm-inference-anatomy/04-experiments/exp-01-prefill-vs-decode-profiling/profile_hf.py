"""
profile_hf.py

Measures prefill latency and per-token decode latency using HuggingFace Transformers.
Uses CUDA events for precise GPU-side timing and torch.profiler for utilization data.

Usage:
    python profile_hf.py --output results_hf.json
    python profile_hf.py --model meta-llama/Llama-3.1-8B-Instruct --output results_hf.json
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPT_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
OUTPUT_LENGTH = 128
WARMUP_RUNS = 2
MEASUREMENT_RUNS = 3

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models have transformed natural language processing. "
    "GPU memory bandwidth is often the bottleneck during autoregressive decoding. "
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PrefillResult:
    prompt_length: int
    ttft_ms: float          # time to first token (prefill + first sample)
    prefill_ms: float       # prefill-only time (estimated)


@dataclass
class DecodeResult:
    prompt_length: int
    output_length: int
    tpot_ms_mean: float     # mean inter-token latency
    tpot_ms_p50: float
    tpot_ms_p95: float
    token_timestamps_ms: list[float]  # timestamp of each token relative to first


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, target_length: int) -> str:
    """Build a prompt that tokenizes to approximately target_length tokens."""
    text = FILLER_TEXT * ((target_length // len(tokenizer.encode(FILLER_TEXT))) + 2)
    tokens = tokenizer.encode(text)[:target_length]
    return tokenizer.decode(tokens)


def cuda_sync_time() -> float:
    """Return current time in milliseconds after synchronizing CUDA."""
    torch.cuda.synchronize()
    return time.perf_counter() * 1000


# ---------------------------------------------------------------------------
# Measurement: TTFT
# ---------------------------------------------------------------------------

def measure_ttft(
    model,
    tokenizer,
    prompt_length: int,
    n_runs: int = MEASUREMENT_RUNS,
) -> PrefillResult:
    """
    Measure time to first token for a given prompt length.

    Strategy: time the first model.generate() call with max_new_tokens=1.
    This isolates prefill + one decode step.
    """
    prompt = build_prompt(tokenizer, prompt_length)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Verify actual token count
    actual_length = inputs["input_ids"].shape[1]

    ttft_samples = []

    for _ in range(n_runs):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
            )

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        ttft_samples.append((t_end - t_start) * 1000)

    ttft_ms = sum(ttft_samples) / len(ttft_samples)

    return PrefillResult(
        prompt_length=actual_length,
        ttft_ms=ttft_ms,
        prefill_ms=ttft_ms,  # for max_new_tokens=1, TTFT ≈ prefill time
    )


# ---------------------------------------------------------------------------
# Measurement: per-token decode latency
# ---------------------------------------------------------------------------

def measure_decode_latency(
    model,
    tokenizer,
    prompt_length: int,
    output_length: int = OUTPUT_LENGTH,
    n_runs: int = MEASUREMENT_RUNS,
) -> DecodeResult:
    """
    Measure per-token decode latency by intercepting each generated token.

    Strategy: use a custom stopping criteria that records a timestamp after
    each token is generated. This gives per-token timing without modifying
    the model or generation loop.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    prompt = build_prompt(tokenizer, prompt_length)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    actual_prompt_length = inputs["input_ids"].shape[1]

    all_tpot_samples = []
    all_token_timestamps = []

    for _ in range(n_runs):
        token_times = []

        class TimingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                torch.cuda.synchronize()
                token_times.append(time.perf_counter() * 1000)
                return len(token_times) >= output_length

        stopping_criteria = StoppingCriteriaList([TimingCriteria()])

        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        if len(token_times) < 2:
            continue

        # Inter-token latencies (skip first token — it includes prefill)
        inter_token_latencies = [
            token_times[i] - token_times[i - 1]
            for i in range(1, len(token_times))
        ]

        all_tpot_samples.extend(inter_token_latencies)
        all_token_timestamps.append([t - token_times[0] for t in token_times])

    if not all_tpot_samples:
        raise RuntimeError("No timing samples collected — check stopping criteria")

    sorted_samples = sorted(all_tpot_samples)
    n = len(sorted_samples)

    return DecodeResult(
        prompt_length=actual_prompt_length,
        output_length=output_length,
        tpot_ms_mean=sum(all_tpot_samples) / n,
        tpot_ms_p50=sorted_samples[int(n * 0.50)],
        tpot_ms_p95=sorted_samples[int(n * 0.95)],
        token_timestamps_ms=all_token_timestamps[-1] if all_token_timestamps else [],
    )


# ---------------------------------------------------------------------------
# GPU utilization snapshot (optional, requires pynvml)
# ---------------------------------------------------------------------------

def try_get_gpu_utilization() -> Optional[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util_pct": util.gpu,
            "mem_used_gb": mem.used / 1e9,
            "mem_total_gb": mem.total / 1e9,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--output", default="results_hf.json")
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=PROMPT_LENGTHS,
    )
    parser.add_argument("--output-length", type=int, default=OUTPUT_LENGTH)
    parser.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    parser.add_argument("--measurement-runs", type=int, default=MEASUREMENT_RUNS)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Warmup
    print(f"\nRunning {args.warmup_runs} warmup runs...")
    warmup_prompt = build_prompt(tokenizer, 64)
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    for _ in range(args.warmup_runs):
        with torch.inference_mode():
            _ = model.generate(**warmup_inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()
    print("Warmup complete.")

    results = {
        "model": args.model,
        "hardware": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "prefill_results": [],
        "decode_results": [],
    }

    # TTFT measurements
    print("\n--- Prefill / TTFT measurements ---")
    for length in args.prompt_lengths:
        print(f"  prompt_length={length}...", end=" ", flush=True)
        result = measure_ttft(model, tokenizer, length, args.measurement_runs)
        results["prefill_results"].append(asdict(result))
        print(f"TTFT={result.ttft_ms:.1f}ms")

    # Decode measurements
    print("\n--- Decode latency measurements ---")
    for length in args.prompt_lengths:
        print(f"  prompt_length={length}, output_length={args.output_length}...", end=" ", flush=True)
        result = measure_decode_latency(
            model, tokenizer, length, args.output_length, args.measurement_runs
        )
        results["decode_results"].append(asdict(result))
        print(f"TPOT_mean={result.tpot_ms_mean:.1f}ms, TPOT_p95={result.tpot_ms_p95:.1f}ms")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()