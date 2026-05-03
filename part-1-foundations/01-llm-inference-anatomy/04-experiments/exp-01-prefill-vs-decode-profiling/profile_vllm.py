"""
profile_vllm.py

Measures prefill latency and per-token decode latency using vLLM.
Uses vLLM's built-in metrics and request-level timing.

Usage:
    python profile_vllm.py --output results_vllm.json
    python profile_vllm.py --model meta-llama/Llama-3.1-8B-Instruct --output results_vllm.json
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


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
class RequestTiming:
    prompt_length: int
    output_length: int
    ttft_ms: float
    total_latency_ms: float
    tpot_ms_mean: float
    token_timestamps_ms: list[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(length: int) -> str:
    """Build a prompt of approximately `length` tokens using filler text."""
    return (FILLER_TEXT * ((length // len(FILLER_TEXT.split())) + 2))[:length * 5]


# ---------------------------------------------------------------------------
# Offline measurement via LLM (synchronous)
# ---------------------------------------------------------------------------

def measure_with_offline_engine(
    model: str,
    prompt_lengths: list[int],
    output_length: int,
    warmup_runs: int,
    measurement_runs: int,
) -> list[RequestTiming]:
    """
    Use vLLM's offline LLM interface to measure TTFT and TPOT.

    vLLM's offline interface does not natively expose per-token timestamps,
    so we use the async streaming interface for per-token timing and the
    offline interface for aggregate metrics.
    """
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_length,
    )

    results = []

    # Warmup
    print(f"Running {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        llm.generate([build_prompt(64)], sampling_params)
    print("Warmup complete.")

    for prompt_length in prompt_lengths:
        prompt = build_prompt(prompt_length)
        run_timings = []

        for _ in range(measurement_runs):
            t_start = time.perf_counter()
            output = llm.generate([prompt], sampling_params)[0]
            t_end = time.perf_counter()

            total_ms = (t_end - t_start) * 1000
            actual_output_length = len(output.outputs[0].token_ids)
            actual_prompt_length = len(output.prompt_token_ids)

            # vLLM RequestOutput includes metrics in recent versions
            metrics = getattr(output, "metrics", None)
            if metrics is not None:
                ttft_ms = (metrics.first_token_time - metrics.arrival_time) * 1000
            else:
                # Fallback: estimate TTFT from total latency
                # TTFT ≈ total - (output_length - 1) × TPOT
                tpot_est = total_ms / max(actual_output_length, 1)
                ttft_ms = total_ms - (actual_output_length - 1) * tpot_est

            run_timings.append({
                "total_ms": total_ms,
                "ttft_ms": ttft_ms,
                "actual_prompt_length": actual_prompt_length,
                "actual_output_length": actual_output_length,
            })

        avg_total = sum(r["total_ms"] for r in run_timings) / len(run_timings)
        avg_ttft = sum(r["ttft_ms"] for r in run_timings) / len(run_timings)
        avg_output_length = run_timings[-1]["actual_output_length"]
        tpot_mean = (avg_total - avg_ttft) / max(avg_output_length - 1, 1)

        results.append(RequestTiming(
            prompt_length=run_timings[-1]["actual_prompt_length"],
            output_length=avg_output_length,
            ttft_ms=avg_ttft,
            total_latency_ms=avg_total,
            tpot_ms_mean=tpot_mean,
            token_timestamps_ms=[],  # not available in offline mode
        ))

        print(
            f"  prompt_length={run_timings[-1]['actual_prompt_length']}, "
            f"TTFT={avg_ttft:.1f}ms, "
            f"TPOT={tpot_mean:.1f}ms"
        )

    return results


# ---------------------------------------------------------------------------
# Streaming measurement via AsyncLLMEngine (per-token timestamps)
# ---------------------------------------------------------------------------

async def measure_streaming_async(
    model: str,
    prompt_lengths: list[int],
    output_length: int,
    warmup_runs: int,
    measurement_runs: int,
) -> list[RequestTiming]:
    """
    Use vLLM's async streaming interface to capture per-token timestamps.
    Each token arrival is recorded as it streams out.
    """
    engine_args = AsyncEngineArgs(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_length,
    )

    async def run_single(prompt: str, request_id: str) -> RequestTiming:
        token_timestamps = []
        t_start = time.perf_counter()

        async for output in engine.generate(prompt, sampling_params, request_id):
            torch_time = time.perf_counter()
            token_timestamps.append((torch_time - t_start) * 1000)

        t_end = time.perf_counter()

        if len(token_timestamps) < 2:
            inter_token_latencies = [0.0]
        else:
            inter_token_latencies = [
                token_timestamps[i] - token_timestamps[i - 1]
                for i in range(1, len(token_timestamps))
            ]

        return RequestTiming(
            prompt_length=len(prompt.split()),  # approximate
            output_length=len(token_timestamps),
            ttft_ms=token_timestamps[0] if token_timestamps else 0.0,
            total_latency_ms=(t_end - t_start) * 1000,
            tpot_ms_mean=sum(inter_token_latencies) / len(inter_token_latencies),
            token_timestamps_ms=token_timestamps,
        )

    # Warmup
    print(f"Running {warmup_runs} warmup runs (streaming)...")
    for i in range(warmup_runs):
        await run_single(build_prompt(64), f"warmup_{i}")
    print("Warmup complete.")

    results = []
    for prompt_length in prompt_lengths:
        prompt = build_prompt(prompt_length)
        run_results = []

        for run_idx in range(measurement_runs):
            r = await run_single(prompt, f"measure_{prompt_length}_{run_idx}")
            run_results.append(r)

        # Average across runs
        avg = RequestTiming(
            prompt_length=run_results[-1].prompt_length,
            output_length=run_results[-1].output_length,
            ttft_ms=sum(r.ttft_ms for r in run_results) / len(run_results),
            total_latency_ms=sum(r.total_latency_ms for r in run_results) / len(run_results),
            tpot_ms_mean=sum(r.tpot_ms_mean for r in run_results) / len(run_results),
            token_timestamps_ms=run_results[-1].token_timestamps_ms,
        )
        results.append(avg)

        print(
            f"  prompt_length≈{prompt_length}, "
            f"TTFT={avg.ttft_ms:.1f}ms, "
            f"TPOT={avg.tpot_ms_mean:.1f}ms"
        )

    return results


def measure_streaming(model, prompt_lengths, output_length, warmup_runs, measurement_runs):
    return asyncio.run(
        measure_streaming_async(model, prompt_lengths, output_length, warmup_runs, measurement_runs)
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
    parser.add_argument("--output", default="results_vllm.json")
    parser.add_argument(
        "--mode",
        choices=["offline", "streaming"],
        default="offline",
        help="offline: aggregate metrics only. streaming: per-token timestamps.",
    )
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

    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Output length: {args.output_length}")

    if args.mode == "offline":
        timings = measure_with_offline_engine(
            args.model,
            args.prompt_lengths,
            args.output_length,
            args.warmup_runs,
            args.measurement_runs,
        )
    else:
        timings = measure_streaming(
            args.model,
            args.prompt_lengths,
            args.output_length,
            args.warmup_runs,
            args.measurement_runs,
        )

    results = {
        "model": args.model,
        "mode": args.mode,
        "output_length": args.output_length,
        "timings": [asdict(t) for t in timings],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()