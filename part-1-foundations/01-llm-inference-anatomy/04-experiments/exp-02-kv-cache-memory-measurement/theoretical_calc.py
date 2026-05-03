"""
theoretical_calc.py

Computes theoretical KV cache sizes from model config.
Validates against the formula:

    KV bytes = 2 × L × H_kv × d_head × T × b

Run standalone to print a reference table, or import and call compute_kv_cache_bytes()
directly.

Usage:
    python theoretical_calc.py
    python theoretical_calc.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_attention_heads: int
    hidden_size: int
    bytes_per_element: int = 2  # BF16 / FP16


# ---------------------------------------------------------------------------
# Known model specs (for offline use without HuggingFace)
# ---------------------------------------------------------------------------

KNOWN_MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        name="Llama 3.1 8B",
        num_layers=32,
        num_kv_heads=8,       # GQA
        head_dim=128,
        num_attention_heads=32,
        hidden_size=4096,
    ),
    "llama-3.1-70b": ModelSpec(
        name="Llama 3.1 70B",
        num_layers=80,
        num_kv_heads=8,       # GQA
        head_dim=128,
        num_attention_heads=64,
        hidden_size=8192,
    ),
    "llama-3.1-405b": ModelSpec(
        name="Llama 3.1 405B",
        num_layers=126,
        num_kv_heads=8,       # GQA
        head_dim=128,
        num_attention_heads=128,
        hidden_size=16384,
    ),
    "llama-2-7b": ModelSpec(
        name="Llama 2 7B",
        num_layers=32,
        num_kv_heads=32,      # full MHA
        head_dim=128,
        num_attention_heads=32,
        hidden_size=4096,
    ),
    "llama-2-70b": ModelSpec(
        name="Llama 2 70B",
        num_layers=80,
        num_kv_heads=8,       # GQA
        head_dim=128,
        num_attention_heads=64,
        hidden_size=8192,
    ),
    "mistral-7b": ModelSpec(
        name="Mistral 7B v0.1",
        num_layers=32,
        num_kv_heads=8,       # GQA
        head_dim=128,
        num_attention_heads=32,
        hidden_size=4096,
    ),
}


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def compute_kv_cache_bytes(spec: ModelSpec, sequence_length: int) -> int:
    """
    KV cache bytes = 2 × L × H_kv × d_head × T × b

    Returns bytes as an integer.
    """
    return (
        2
        * spec.num_layers
        * spec.num_kv_heads
        * spec.head_dim
        * sequence_length
        * spec.bytes_per_element
    )


def bytes_per_token(spec: ModelSpec) -> int:
    """KV cache bytes added per generated token."""
    return compute_kv_cache_bytes(spec, sequence_length=1)


def crossover_sequence_length(spec: ModelSpec, model_weight_gb: float) -> float:
    """
    Sequence length at which KV cache equals model weight memory.

    Useful for understanding when KV cache becomes the dominant memory consumer.
    """
    model_weight_bytes = model_weight_gb * 1e9
    bytes_per_tok = bytes_per_token(spec)
    return model_weight_bytes / bytes_per_tok


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_model_summary(spec: ModelSpec, model_weight_gb: float | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"  {spec.name}")
    print(f"{'='*60}")
    print(f"  Layers:           {spec.num_layers}")
    print(f"  KV heads:         {spec.num_kv_heads}")
    print(f"  Head dim:         {spec.head_dim}")
    print(f"  Attn heads:       {spec.num_attention_heads}")
    print(f"  KV ratio:         {spec.num_kv_heads}/{spec.num_attention_heads} "
          f"({'GQA' if spec.num_kv_heads < spec.num_attention_heads else 'MHA'})")
    print(f"  Bytes per element:{spec.bytes_per_element} (BF16/FP16)")
    print(f"  KV bytes/token:   {bytes_per_token(spec) / 1024:.1f} KiB")
    print()

    sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    print(f"  {'Seq length':>12} | {'KV cache (MB)':>14} | {'KV cache (GB)':>14}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*14}")

    for seq_len in sequence_lengths:
        kv_bytes = compute_kv_cache_bytes(spec, seq_len)
        kv_mb = kv_bytes / 1e6
        kv_gb = kv_bytes / 1e9
        print(f"  {seq_len:>12,} | {kv_mb:>14.1f} | {kv_gb:>14.3f}")

    if model_weight_gb is not None:
        crossover = crossover_sequence_length(spec, model_weight_gb)
        print()
        print(f"  Model weights:    {model_weight_gb:.1f} GB")
        print(f"  KV = weights at:  {crossover:,.0f} tokens")


def print_comparison_table(specs: list[ModelSpec]) -> None:
    print(f"\n{'Model':<20} {'KV heads':>10} {'KV/token (KiB)':>16} {'@4096 tokens (MB)':>20} {'@32k tokens (GB)':>18}")
    print("-" * 90)
    for spec in specs:
        kv_per_tok = bytes_per_token(spec) / 1024
        kv_4096 = compute_kv_cache_bytes(spec, 4096) / 1e6
        kv_32k = compute_kv_cache_bytes(spec, 32768) / 1e9
        print(
            f"{spec.name:<20} {spec.num_kv_heads:>10} {kv_per_tok:>16.1f} "
            f"{kv_4096:>20.1f} {kv_32k:>18.2f}"
        )


# ---------------------------------------------------------------------------
# Load from HuggingFace config (optional)
# ---------------------------------------------------------------------------

def spec_from_hf_config(model_name: str) -> ModelSpec:
    """Load model spec from HuggingFace config."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)

    num_kv_heads = getattr(
        config, "num_key_value_heads",
        getattr(config, "num_attention_heads", None)
    )
    head_dim = getattr(
        config, "head_dim",
        config.hidden_size // config.num_attention_heads
    )

    return ModelSpec(
        name=model_name,
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_attention_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name to load config from. "
             "If not provided, uses built-in specs.",
    )
    parser.add_argument(
        "--model-weight-gb",
        type=float,
        default=None,
        help="Model weight memory in GB for crossover calculation.",
    )
    args = parser.parse_args()

    if args.model:
        print(f"Loading config from HuggingFace: {args.model}")
        spec = spec_from_hf_config(args.model)
        print_model_summary(spec, args.model_weight_gb)
    else:
        # Print summary for all known models
        print("\nKV Cache Theoretical Calculator")
        print("Formula: KV bytes = 2 × L × H_kv × d_head × T × b")
        print("Precision: BF16 (2 bytes per element)")

        # Llama 3.1 8B — primary reference model for this handbook
        print_model_summary(KNOWN_MODELS["llama-3.1-8b"], model_weight_gb=16.0)

        # 8B MHA vs GQA comparison
        mha_spec = ModelSpec(
            name="Llama 2 7B (MHA, 32 KV heads)",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            num_attention_heads=32,
            hidden_size=4096,
        )
        print_model_summary(mha_spec, model_weight_gb=14.0)

        # Comparison table
        print(f"\n\n{'='*90}")
        print("  Model comparison at a glance")
        print(f"{'='*90}")
        print_comparison_table([
            KNOWN_MODELS["llama-3.1-8b"],
            KNOWN_MODELS["llama-2-7b"],
            KNOWN_MODELS["mistral-7b"],
            KNOWN_MODELS["llama-3.1-70b"],
            KNOWN_MODELS["llama-3.1-405b"],
        ])

        # Key insight
        print(f"\n\nKey insight:")
        spec_gqa = KNOWN_MODELS["llama-3.1-8b"]
        spec_mha = KNOWN_MODELS["llama-2-7b"]
        kv_gqa = compute_kv_cache_bytes(spec_gqa, 4096) / 1e6
        kv_mha = compute_kv_cache_bytes(spec_mha, 4096) / 1e6
        print(f"  Llama 3.1 8B (GQA, {spec_gqa.num_kv_heads} KV heads): {kv_gqa:.0f} MB at 4096 tokens")
        print(f"  Llama 2 7B  (MHA, {spec_mha.num_kv_heads} KV heads): {kv_mha:.0f} MB at 4096 tokens")
        print(f"  GQA is {kv_mha / kv_gqa:.1f}× more memory-efficient at the same context length.")


if __name__ == "__main__":
    main()