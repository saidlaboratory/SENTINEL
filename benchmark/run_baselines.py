"""Compare SENTINEL configurations against simple guardrail baselines."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import category_summary, false_positive, pipeline_blocked
from main import IAJDAPipeline, configure_logging, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEYWORD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore previous instructions",
        r"do anything now",
        r"system prompt",
        r"bypass safety",
        r"developer mode",
        r"jailbreak",
        r"synthesize meth",
        r"malware",
        r"ransomware",
        r"pipe bomb",
        r"credit card fraud",
        r"fake passport",
        r"\bdan\b",
        r"\bstan\b",
    )
)

CONFIGS: dict[str, frozenset[int] | None] = {
    "keyword_baseline": None,
    "l1_only": frozenset({1}),
    "l2_l3": frozenset({2, 3}),
    "sentinel_l1_l3": frozenset({1, 2, 3}),
}


def keyword_blocked(prompt: str) -> bool:
    return any(pattern.search(prompt) for pattern in KEYWORD_PATTERNS)


def run_baselines() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    pipeline = IAJDAPipeline(config, clean_mode=False, use_color=False)

    prompts_path = PROJECT_ROOT / "benchmark" / "prompts.csv"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "baseline_comparison.json"

    prompts = list(csv.DictReader(prompts_path.open("r", encoding="utf-8", newline="")))
    summary: dict[str, Any] = {}

    for config_name, enabled_layers in CONFIGS.items():
        rows: list[dict[str, Any]] = []
        for item in prompts:
            category = item["category"]
            if config_name == "keyword_baseline":
                blocked = keyword_blocked(item["prompt"])
                result = {"final_response": "", "refusal_message": ""}
            else:
                result = pipeline.evaluate(
                    item["prompt"],
                    enabled_layers=enabled_layers,
                    skip_llm=True,
                )
                blocked = pipeline_blocked(result)
            rows.append(
                {
                    "config": config_name,
                    "category": category,
                    "blocked": blocked,
                    "false_positive": false_positive(result, category)
                    if config_name != "keyword_baseline"
                    else category == "benign" and blocked,
                }
            )
        categories = category_summary(rows)
        attack_rows = [row for row in rows if row["category"] != "benign"]
        benign_rows = [row for row in rows if row["category"] == "benign"]
        summary[config_name] = {
            "categories": categories,
            "attack_block_rate": sum(1 for row in attack_rows if row["blocked"])
            / max(len(attack_rows), 1),
            "false_positive_rate": sum(1 for row in benign_rows if row["false_positive"])
            / max(len(benign_rows), 1),
        }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    run_baselines()
