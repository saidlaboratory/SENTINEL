"""Generate Track B baseline vs SENTINEL figure for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "results" / "endtoend_summary.json"
PAPER = PROJECT_ROOT / "paper"

CATEGORIES = [
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


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError("Run benchmark/run_endtoend.py first.")

    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if data["baseline"]["attack_prompts"] == 0:
        raise RuntimeError("No adversarial prompts completed yet.")

    baseline = data["baseline"]["categories"]
    attr_cats = data["attribution"]["categories"]
    labels = [label for _, label in CATEGORIES]
    x = np.arange(len(labels))
    width = 0.35

    baseline_vals = [
        baseline.get(key, {}).get("block_rate", 0.0) * 100.0 for key, _ in CATEGORIES
    ]
    sentinel_vals = [
        (
            attr_cats.get(key, {}).get(
                "end_to_end_block_rate",
                1.0 - attr_cats.get(key, {}).get("true_asr_rate", 0.0),
            )
            * 100.0
        )
        for key, _ in CATEGORIES
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline refusal", color="#f4a261")
    ax.bar(x + width / 2, sentinel_vals, width, label="SENTINEL end-to-end", color="#2a6fbb")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output = PAPER / "fig12_trackb_endtoend.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
