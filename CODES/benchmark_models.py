"""Run one SANZI benchmark for every local model and precision."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from backend.app import GenerationSettings, LocalLLMEngine, MODELS, PRECISIONS


PROMPT = (
    "Explain local language model inference on a personal computer in a short "
    "paragraph."
)
SETTINGS = GenerationSettings(
    max_length=150,
    seed=42,
    temperature=0.7,
    top_p=0.9,
)
CONFIGURATIONS = [
    ("qwen", "fp16"),
    ("qwen", "int8"),
    ("qwen", "int4"),
    ("llama", "fp16"),
    ("llama", "int8"),
    ("llama", "int4"),
]
RESULTS_PATH = Path(__file__).resolve().parent / "benchmark_results.csv"


def benchmark_configuration(
    engine: LocalLLMEngine,
    model_id: str,
    precision: str,
) -> dict[str, Any]:
    """Load and benchmark one model and precision configuration."""
    result: dict[str, Any] = {
        "model": MODELS[model_id]["full_name"],
        "precision": PRECISIONS[precision]["name"],
        "device": "",
        "load_time_seconds": "",
        "generation_time_seconds": "",
        "output_tokens": "",
        "tokens_per_second": "",
        "error": "",
    }

    original_generate = None
    try:
        engine.load(model_id, precision)
        result["device"] = engine.device_name
        result["load_time_seconds"] = engine.load_time_seconds

        if engine.model is None:
            raise RuntimeError("Model loading completed without a model instance.")

        timing: dict[str, float] = {}
        original_generate = engine.model.generate

        def timed_generate(*args: Any, **kwargs: Any):
            started_at = time.perf_counter()
            try:
                return original_generate(*args, **kwargs)
            finally:
                timing["generation_time_seconds"] = time.perf_counter() - started_at

        engine.model.generate = timed_generate
        _, _, output_tokens, _ = engine.generate(
            [{"role": "user", "content": PROMPT}],
            SETTINGS,
        )

        generation_time = timing["generation_time_seconds"]
        result["generation_time_seconds"] = generation_time
        result["output_tokens"] = output_tokens
        result["tokens_per_second"] = output_tokens / generation_time
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if original_generate is not None and engine.model is not None:
            engine.model.generate = original_generate
        engine.unload()

    return result


def write_results(results: list[dict[str, Any]]) -> None:
    """Save benchmark results as CSV."""
    fieldnames = [
        "model",
        "precision",
        "device",
        "load_time_seconds",
        "generation_time_seconds",
        "output_tokens",
        "tokens_per_second",
        "error",
    ]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def format_number(value: Any, decimal_places: int) -> str:
    """Format measured values while leaving failed measurements blank."""
    if value == "" or value is None:
        return ""
    return f"{float(value):.{decimal_places}f}"


def print_results(results: list[dict[str, Any]]) -> None:
    """Print benchmark results as an aligned terminal table."""
    headers = [
        "Model",
        "Precision",
        "Device",
        "Load (s)",
        "Generation (s)",
        "Output tokens",
        "Tokens/s",
        "Error",
    ]
    rows = [
        [
            str(result["model"]),
            str(result["precision"]),
            str(result["device"]),
            format_number(result["load_time_seconds"], 3),
            format_number(result["generation_time_seconds"], 3),
            str(result["output_tokens"]),
            format_number(result["tokens_per_second"], 2),
            str(result["error"]),
        ]
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def print_row(row: list[str]) -> None:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    print_row(headers)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print_row(row)


def main() -> None:
    """Run all six one-shot benchmark configurations."""
    engine = LocalLLMEngine()
    results: list[dict[str, Any]] = []
    engine.unload()

    print(f"Prompt: {PROMPT}")
    print(
        "Settings: seed=42, temperature=0.7, top_p=0.9, "
        "max_output_tokens=150\n"
    )

    for index, (model_id, precision) in enumerate(CONFIGURATIONS, start=1):
        model_name = MODELS[model_id]["full_name"]
        precision_name = PRECISIONS[precision]["name"]
        print(
            f"[{index}/{len(CONFIGURATIONS)}] "
            f"Benchmarking {model_name} - {precision_name}...",
            flush=True,
        )
        result = benchmark_configuration(engine, model_id, precision)
        results.append(result)
        if result["error"]:
            print(f"  Failed: {result['error']}", flush=True)
        else:
            print(
                f"  Completed: {result['output_tokens']} output tokens at "
                f"{result['tokens_per_second']:.2f} tokens/s",
                flush=True,
            )

    write_results(results)
    print("\nResults\n")
    print_results(results)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
