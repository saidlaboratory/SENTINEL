"""Regenerate all reproducible result artifacts (no LLM unless Track B CSV missing)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def run(script: str) -> None:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "benchmark" / script)], check=True)


def main() -> None:
    if (RESULTS / "endtoend_sentinel.csv").exists():
        run("rescore_trackb.py")
    run("run_ablation.py")
    run("run_baselines.py")
    run("run_benchmark.py")
    run("generate_figure.py")
    run("generate_endtoend_figure.py")
    run("generate_latency_figure.py")
    run("generate_radar_figure.py")
    run("generate_heatmap_figure.py")
    run("update_paper_trackb.py")

    summary = {
        "track_a": {
            "ablation": json.loads((RESULTS / "ablation_summary.json").read_text()),
            "baselines": json.loads((RESULTS / "baseline_comparison.json").read_text()),
            "benchmark": json.loads((RESULTS / "summary.json").read_text()),
        },
        "track_b": json.loads((RESULTS / "endtoend_summary.json").read_text()),
    }
    out = RESULTS / "full_results_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
