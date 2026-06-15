"""Generate Track A layer coverage vs latency tradeoff figure (fig11)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
PAPER = PROJECT_ROOT / "paper"


def median_ms(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return float(median(float(row["total_ms"]) for row in rows))


def attack_block_rate(summary: dict, config: str) -> float:
    categories = summary[config]
    attack_total = sum(value["count"] for key, value in categories.items() if key != "benign")
    attack_blocked = sum(value["blocked"] for key, value in categories.items() if key != "benign")
    return 100.0 * attack_blocked / max(attack_total, 1)


def main() -> None:
    ablation_csv = RESULTS / "ablation_results.csv"
    ablation_json = RESULTS / "ablation_summary.json"
    if not ablation_csv.exists() or not ablation_json.exists():
        raise FileNotFoundError("Run benchmark/run_ablation.py first.")

    rows = list(csv.DictReader(ablation_csv.open(encoding="utf-8")))
    summary = json.loads(ablation_json.read_text(encoding="utf-8"))
    l13 = [row for row in rows if row["ablation"] == "L1_L2_L3"]

    layer_stats = []
    for layer, label, color in (
        ("L3_fast", "L3 fast", "#2a6fbb"),
        ("L1", "L1 intent", "#e76f51"),
    ):
        blocked = [row for row in l13 if row["blocking_layer"] == layer and row["blocked"] == "True"]
        layer_stats.append(
            {
                "label": label,
                "count": len(blocked),
                "median_ms": median_ms(blocked),
                "color": color,
            }
        )

    benign_pass = [row for row in l13 if row["category"] == "benign" and row["blocked"] == "False"]
    benign_med = median_ms(benign_pass)

    config_stats = []
    for config, label in (("L2_L3", "L2+L3"), ("L1_L2_L3", "L1+L2+L3")):
        blocked_rows = [row for row in rows if row["ablation"] == config and row["blocked"] == "True"]
        config_stats.append(
            {
                "label": label,
                "coverage": attack_block_rate(summary, config),
                "median_ms": median_ms(blocked_rows) if blocked_rows else benign_med,
            }
        )

    fig, (ax_blocks, ax_tradeoff) = plt.subplots(
        1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.05, 1.2]}
    )

    block_labels = [item["label"] for item in layer_stats]
    block_counts = [item["count"] for item in layer_stats]
    block_colors = [item["color"] for item in layer_stats]
    ax_blocks.bar(block_labels, block_counts, color=block_colors, width=0.55)
    ax_blocks.set_ylabel("Blocks (L1--3, n=67)")
    ax_blocks.set_ylim(0, max(block_counts) * 1.2)
    for idx, item in enumerate(layer_stats):
        ax_blocks.text(
            idx,
            item["count"] + 1.5,
            f"{item['median_ms']:.1f} ms",
            ha="center",
            fontsize=8,
        )
    ax_blocks.set_title("Block path latency", fontsize=9)

    for item in config_stats:
        ax_tradeoff.scatter(
            item["median_ms"],
            item["coverage"],
            s=110,
            color="#264653",
            zorder=3,
        )
        ax_tradeoff.annotate(
            f"{item['label']}\n{item['coverage']:.1f}% @ {item['median_ms']:.1f} ms",
            (item["median_ms"], item["coverage"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax_tradeoff.axhline(config_stats[-1]["coverage"], color="#adb5bd", linestyle=":", linewidth=1)
    ax_tradeoff.set_xlabel("Median latency of blocked prompts (ms)")
    ax_tradeoff.set_ylabel("Attack block rate (%)")
    ax_tradeoff.set_title("Coverage vs latency (ablation)", fontsize=9)
    ax_tradeoff.grid(axis="y", alpha=0.25)
    ax_tradeoff.set_xlim(left=0)
    ax_tradeoff.set_ylim(0, 58)

    fig.suptitle("Track A: layer coverage--latency tradeoff", fontsize=10, y=1.02)
    fig.tight_layout()

    output = PAPER / "fig11_layer_coverage_latency.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
