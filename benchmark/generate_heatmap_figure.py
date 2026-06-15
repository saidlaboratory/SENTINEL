"""Generate attack-category x ablation heatmap (fig7) from Track A results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
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

CONFIGS = [
    ("none", "None"),
    ("L2_L3", "L2+L3"),
    ("L1_L2_L3", "L1+L2+L3"),
]


def main() -> None:
    summary_path = RESULTS / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run benchmark/run_ablation.py first.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matrix = np.array(
        [
            [
                summary[config][key]["block_rate"] * 100.0
                for key, _ in CATEGORIES
            ]
            for config, _ in CONFIGS
        ]
    )

    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([label for _, label in CATEGORIES], rotation=30, ha="right")
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_yticklabels([label for _, label in CONFIGS])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value >= 55 else "black"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Track A block rate (%) by category and ablation config")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Block rate (%)")
    fig.tight_layout()

    output = PAPER / "fig7_jailbreak_heatmap.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
