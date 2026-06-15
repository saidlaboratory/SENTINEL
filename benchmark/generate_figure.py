"""Generate paper figure from baseline comparison and SENTINEL results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
PAPER = PROJECT_ROOT / "paper"

ATTACK_CATEGORIES = [
    ("direct_harmful", "Direct"),
    ("context_manipulation", "Context"),
    ("roleplay_persona", "Roleplay"),
    ("encoded_obfuscation", "Encoded"),
    ("prompt_injection", "Injection"),
    ("prompt_leakage", "Leakage"),
    ("multi_step_escalation", "Multi-step"),
    ("agent_delegation", "Agent"),
    ("indirect_injection", "Indirect"),
]


def load_block_rates(path: Path, config_name: str) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = data[config_name]["categories"]
    return {
        key: float(value["block_rate"]) * 100.0
        for key, value in categories.items()
        if key != "benign"
    }


def main() -> None:
    baseline_path = RESULTS / "baseline_comparison.json"
    if not baseline_path.exists():
        raise FileNotFoundError("Run benchmark/run_baselines.py first.")

    keyword_rates = load_block_rates(baseline_path, "keyword_baseline")
    sentinel_rates = load_block_rates(baseline_path, "sentinel_l1_l3")

    labels = [label for _, label in ATTACK_CATEGORIES]
    x = np.arange(len(labels))
    width = 0.35

    keyword_values = [
        keyword_rates.get(key, 0.0) for key, _ in ATTACK_CATEGORIES
    ]
    sentinel_values = [
        sentinel_rates.get(key, 0.0) for key, _ in ATTACK_CATEGORIES
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, keyword_values, width, label="Keyword baseline", color="#f4a261")
    ax.bar(x + width / 2, sentinel_values, width, label="SENTINEL L1--3", color="#2a6fbb")
    ax.set_ylabel("Block rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output = PAPER / "fig1_refusal_paraphrase.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
