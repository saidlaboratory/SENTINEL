"""Run layer ablation on the public SENTINEL prompt set (pre-LLM modes)."""

from __future__ import annotations

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
)
from main import IAJDAPipeline, configure_logging, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABLATIONS: dict[str, frozenset[int]] = {
    "none": frozenset(),
    "L2_only": frozenset({2}),
    "L2_L3": frozenset({2, 3}),
    "L1_L2_L3": frozenset({1, 2, 3}),
}


def run_ablation() -> None:
    """Evaluate ablation configs without invoking the local LLM."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    pipeline = IAJDAPipeline(config, clean_mode=False, use_color=False)

    prompts_path = PROJECT_ROOT / "benchmark" / "prompts.csv"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts: list[dict[str, str]] = []
    with prompts_path.open("r", encoding="utf-8", newline="") as file:
        prompts = list(csv.DictReader(file))

    all_rows: list[dict[str, Any]] = []
    summary_by_config: dict[str, Any] = {}

    for config_name, enabled_layers in ABLATIONS.items():
        rows: list[dict[str, Any]] = []
        for item in prompts:
            result = pipeline.evaluate(
                item["prompt"],
                enabled_layers=enabled_layers,
                skip_llm=True,
            )
            category = item["category"]
            blocked = pipeline_blocked(result)
            rows.append(
                {
                    "ablation": config_name,
                    "id": item["id"],
                    "category": category,
                    "blocked": blocked,
                    "false_positive": false_positive(result, category),
                    "blocking_layer": blocking_layer(result),
                    "total_ms": float(result["latency"]["total_ms"]),
                }
            )
        all_rows.extend(rows)
        summary_by_config[config_name] = category_summary(rows)

    detail_path = results_dir / "ablation_results.csv"
    summary_path = results_dir / "ablation_summary.json"

    with detail_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_by_config, file, indent=2)

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    run_ablation()
