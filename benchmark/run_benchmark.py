"""Run SENTINEL benchmark prompts (pre-LLM evaluation by default)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import (
    blocking_layer,
    category_summary,
    false_positive,
    pipeline_blocked,
    summarize,
)
from main import IAJDAPipeline, configure_logging, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_benchmark(*, with_llm: bool = False) -> None:
    """Execute benchmark/prompts.csv and write results files."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    pipeline = IAJDAPipeline(config, clean_mode=False, use_color=False)

    prompts_path = PROJECT_ROOT / "benchmark" / "prompts.csv"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.csv"
    summary_path = results_dir / "summary.json"
    category_path = results_dir / "category_summary.json"

    rows: list[dict[str, Any]] = []
    with prompts_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for item in reader:
            result = pipeline.evaluate(
                item["prompt"],
                enabled_layers=frozenset({1, 2, 3, 4, 5}),
                skip_llm=not with_llm,
                record_latency=with_llm,
            )
            category = item["category"]
            blocked = pipeline_blocked(result)
            row = {
                "id": item["id"],
                "category": category,
                "source": item.get("source", ""),
                "variant": item.get("variant", ""),
                "prompt": item["prompt"],
                "blocked": blocked,
                "false_positive": false_positive(result, category),
                "blocking_layer": blocking_layer(result),
                "final_response": str(result["final_response"]),
                "total_ms": f"{result['latency']['total_ms']:.2f}",
                "layer1_ms": f"{result['latency']['layer1_ms']:.2f}",
                "layer2_ms": f"{result['latency']['layer2_ms']:.2f}",
                "layer3_ms": f"{result['latency']['layer3_ms']:.2f}",
                "layer4_ms": f"{result['latency']['layer4_ms']:.2f}",
                "layer5_ms": f"{result['latency']['layer5_ms']:.2f}",
            }
            rows.append(row)

    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    categories = category_summary(rows)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with category_path.open("w", encoding="utf-8") as file:
        json.dump(categories, file, indent=2)

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {category_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SENTINEL benchmark.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Invoke the local LLM (requires LM Studio). Default is pre-LLM evaluation.",
    )
    args = parser.parse_args()
    run_benchmark(with_llm=args.with_llm)
