"""Run IAJDA benchmark prompts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import harmful_compliance, is_refusal, summarize, token_count
from main import IAJDAPipeline, configure_logging, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_benchmark() -> None:
    """Execute benchmark/prompts.csv and write results files."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    pipeline = IAJDAPipeline(config, clean_mode=False, use_color=False)

    prompts_path = PROJECT_ROOT / "benchmark" / "prompts.csv"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.csv"
    summary_path = results_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    with prompts_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for item in reader:
            result = pipeline.run(item["prompt"])
            response = str(result["final_response"])
            row = {
                "id": item["id"],
                "attack_type": item["attack_type"],
                "prompt": item["prompt"],
                "response": response,
                "refused": is_refusal(response),
                "harmful_compliance": harmful_compliance(response),
                "response_tokens": token_count(response),
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
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    run_benchmark()
