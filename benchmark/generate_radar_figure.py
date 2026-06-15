"""Generate per-category defense profile radar (fig9) from reproducible results."""

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


def load_rates(path: Path, *keys: str) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    node = data
    for key in keys:
        node = node[key]
    return {
        category: float(node[category]["block_rate"]) * 100.0
        for category, _ in CATEGORIES
        if category in node
    }


def main() -> None:
    baseline_path = RESULTS / "baseline_comparison.json"
    endtoend_path = RESULTS / "endtoend_summary.json"
    if not baseline_path.exists() or not endtoend_path.exists():
        raise FileNotFoundError("Run baselines and rescore_trackb first.")

    track_a = load_rates(baseline_path, "sentinel_l1_l3", "categories")
    track_b_e2e = {
        key: float(value.get("end_to_end_block_rate", 1.0 - value["true_asr_rate"])) * 100.0
        for key, value in json.loads(endtoend_path.read_text())["attribution"][
            "categories"
        ].items()
    }
    track_b_baseline = load_rates(endtoend_path, "baseline", "categories")

    labels = [label for _, label in CATEGORIES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    series = [
        ("Track A L1--3", [track_a.get(key, 0.0) for key, _ in CATEGORIES], "#2a6fbb"),
        (
            "Track B baseline refusal",
            [track_b_baseline.get(key, 0.0) for key, _ in CATEGORIES],
            "#f4a261",
        ),
        (
            "Track B + SENTINEL (e2e)",
            [track_b_e2e.get(key, 0.0) for key, _ in CATEGORIES],
            "#264653",
        ),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"polar": True})
    for name, values, color in series:
        closed = np.concatenate([values, [values[0]]])
        ax.plot(angles, closed, color=color, linewidth=2, label=name)
        ax.fill(angles, closed, color=color, alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=False, fontsize=8)
    fig.tight_layout()

    output = PAPER / "fig9_radar.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
