"""Repeat the original SANZI benchmark five times per configuration.

This runner intentionally imports the existing SANZI model engine and preserves
the prompt, generation settings, configuration order, and timing method used by
the original one-run benchmark.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SANZI_PROJECT_ROOT = Path(
    "/Users/hasanalbanna/Desktop/ss2026/Hardware for nn Seminar/SANZI_CODES"
)
sys.path.insert(0, str(SANZI_PROJECT_ROOT))

from backend.app import GenerationSettings, LocalLLMEngine, MODELS, PRECISIONS  # noqa: E402


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
RUNS_PER_CONFIGURATION = 5
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(__file__).resolve().parent / f"sanzi_repeated_benchmark_{RUN_ID}"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_runs.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
METADATA_PATH = RESULTS_DIR / "metadata.json"

RAW_FIELDS = [
    "run",
    "model",
    "precision",
    "device",
    "load_time_seconds",
    "generation_time_seconds",
    "output_tokens",
    "tokens_per_second",
    "success",
    "error",
]

SUMMARY_FIELDS = [
    "model",
    "precision",
    "average_load_time_seconds",
    "average_generation_time_seconds",
    "average_tokens_per_second",
    "minimum_tokens_per_second",
    "maximum_tokens_per_second",
    "standard_deviation_tokens_per_second",
    "successful_runs",
    "total_runs",
]


def benchmark_configuration(
    engine: LocalLLMEngine,
    model_id: str,
    precision: str,
    run_number: int,
) -> dict[str, Any]:
    """Load and benchmark one model/precision run using the original method."""
    result: dict[str, Any] = {
        "run": run_number,
        "model": MODELS[model_id]["full_name"],
        "precision": PRECISIONS[precision]["name"],
        "device": "",
        "load_time_seconds": "",
        "generation_time_seconds": "",
        "output_tokens": "",
        "tokens_per_second": "",
        "success": False,
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
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if original_generate is not None and engine.model is not None:
            engine.model.generate = original_generate
        engine.unload()

    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_id, precision_id in CONFIGURATIONS:
        model_name = MODELS[model_id]["full_name"]
        precision_name = PRECISIONS[precision_id]["name"]
        configuration_runs = [
            result
            for result in results
            if result["model"] == model_name and result["precision"] == precision_name
        ]
        successful = [result for result in configuration_runs if result["success"]]

        load_times = [float(result["load_time_seconds"]) for result in successful]
        generation_times = [
            float(result["generation_time_seconds"]) for result in successful
        ]
        throughputs = [float(result["tokens_per_second"]) for result in successful]

        summaries.append(
            {
                "model": model_name,
                "precision": precision_name,
                "average_load_time_seconds": statistics.mean(load_times) if load_times else "",
                "average_generation_time_seconds": (
                    statistics.mean(generation_times) if generation_times else ""
                ),
                "average_tokens_per_second": (
                    statistics.mean(throughputs) if throughputs else ""
                ),
                "minimum_tokens_per_second": min(throughputs) if throughputs else "",
                "maximum_tokens_per_second": max(throughputs) if throughputs else "",
                "standard_deviation_tokens_per_second": (
                    statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
                ),
                "successful_runs": len(successful),
                "total_runs": len(configuration_runs),
            }
        )
    return summaries


def build_observations(results: list[dict[str, Any]]) -> list[str]:
    observations: list[str] = []
    failures = [result for result in results if not result["success"]]
    if failures:
        observations.append(f"{len(failures)} run(s) failed; see raw_runs.csv for errors.")

    for model_id, precision_id in CONFIGURATIONS:
        model_name = MODELS[model_id]["full_name"]
        precision_name = PRECISIONS[precision_id]["name"]
        successful = [
            result
            for result in results
            if result["success"]
            and result["model"] == model_name
            and result["precision"] == precision_name
        ]
        token_counts = {int(result["output_tokens"]) for result in successful}
        if len(token_counts) > 1:
            observations.append(
                f"{model_name} / {precision_name} produced different output-token "
                f"counts across identical runs: {sorted(token_counts)}."
            )
        throughputs = [float(result["tokens_per_second"]) for result in successful]
        if len(throughputs) > 1:
            average = statistics.mean(throughputs)
            widest_deviation = max(abs(value - average) for value in throughputs)
            if average and widest_deviation / average >= 0.15:
                observations.append(
                    f"{model_name} / {precision_name} had a throughput run at least "
                    "15% away from its configuration mean."
                )

    if not observations:
        observations.append(
            "No failed runs, output-token inconsistencies, or >=15% throughput deviations detected."
        )
    return observations


def write_metadata(results: list[dict[str, Any]]) -> None:
    import kernels
    import torch
    import transformers

    metadata = {
        "benchmark_started_utc": datetime.now(timezone.utc).isoformat(),
        "source_project": str(SANZI_PROJECT_ROOT),
        "original_benchmark_results": str(
            SANZI_PROJECT_ROOT / "benchmark_results.csv"
        ),
        "prompt": PROMPT,
        "settings": {
            "seed": SETTINGS.seed,
            "temperature": SETTINGS.temperature,
            "top_p": SETTINGS.top_p,
            "max_output_tokens": SETTINGS.max_length,
        },
        "runs_per_configuration": RUNS_PER_CONFIGURATION,
        "device_requested": "Apple M1 / MPS",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kernels": getattr(kernels, "__version__", "unknown"),
        "observations": build_observations(results),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    engine = LocalLLMEngine()
    results: list[dict[str, Any]] = []
    engine.unload()

    print(f"Prompt: {PROMPT}")
    print(
        "Settings: seed=42, temperature=0.7, top_p=0.9, "
        "max_output_tokens=150"
    )
    print(f"Runs per configuration: {RUNS_PER_CONFIGURATION}")
    print(f"Results directory: {RESULTS_DIR}\n")

    total_runs = len(CONFIGURATIONS) * RUNS_PER_CONFIGURATION
    completed_runs = 0
    for model_id, precision in CONFIGURATIONS:
        for run_number in range(1, RUNS_PER_CONFIGURATION + 1):
            completed_runs += 1
            model_name = MODELS[model_id]["full_name"]
            precision_name = PRECISIONS[precision]["name"]
            print(
                f"[{completed_runs}/{total_runs}] {model_name} / "
                f"{precision_name} / run {run_number}",
                flush=True,
            )
            result = benchmark_configuration(
                engine,
                model_id,
                precision,
                run_number,
            )
            results.append(result)
            write_csv(RAW_RESULTS_PATH, RAW_FIELDS, results)
            if result["success"]:
                print(
                    f"  load={result['load_time_seconds']:.3f}s, "
                    f"generation={result['generation_time_seconds']:.3f}s, "
                    f"output={result['output_tokens']}, "
                    f"throughput={result['tokens_per_second']:.2f} tokens/s",
                    flush=True,
                )
            else:
                print(f"  FAILED: {result['error']}", flush=True)

    summaries = summarize_results(results)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDS, summaries)
    write_metadata(results)
    print(f"\nRaw results: {RAW_RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
