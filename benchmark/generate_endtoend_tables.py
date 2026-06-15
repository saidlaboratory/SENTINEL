"""Generate LaTeX table rows from end-to-end benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "results" / "endtoend_summary.json"

CATEGORY_LABELS = {
    "direct_harmful": "Direct harmful",
    "context_manipulation": "Context manip.",
    "roleplay_persona": "Roleplay/persona",
    "encoded_obfuscation": "Encoded/obfusc.",
    "prompt_injection": "Prompt injection",
    "prompt_leakage": "Prompt leakage",
    "multi_step_escalation": "Multi-step escal.",
    "agent_delegation": "Agent delegation",
    "indirect_injection": "Indirect inject.",
    "benign": "Benign FP",
}


def pct(value: float) -> str:
    return f"{value * 100:.0f}"


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError("Run benchmark/run_endtoend.py first.")

    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    baseline = data["baseline"]["categories"]
    attr_cats = data["attribution"]["categories"]
    attr = data["attribution"]
    model = data["model"]
    n = data.get("completed_prompts", data["sentinel"]["total_prompts"])

    print(f"% Model: {model}, completed prompts: {n}")
    print("% Baseline refusal vs SENTINEL layer vs incremental block rates (%)")
    for key, label in CATEGORY_LABELS.items():
        if key not in baseline:
            continue
        if key == "benign":
            s_layer = data["sentinel"]["categories"][key]["block_rate"]
            print(f"{label} & --- & {pct(s_layer)} & --- \\\\")
            continue
        b_block = baseline[key]["block_rate"]
        layer = attr_cats[key]["layer_block_rate"]
        incr = attr_cats[key]["incremental_block_rate"]
        print(f"{label} & {pct(b_block)} & {pct(layer)} & {pct(incr)} \\\\")

    print("\n% Aggregate metrics")
    print(
        f"Baseline ASR: {data['baseline']['attack_success_rate']*100:.1f}% | "
        f"FP: {data['baseline']['false_positive_rate']*100:.1f}%"
    )
    print(
        f"SENTINEL LABR: {attr['layer_attributed_block_rate']*100:.1f}% | "
        f"IBR: {attr['incremental_block_rate']*100:.1f}% | "
        f"true ASR: {attr['true_asr_rate']*100:.1f}% | "
        f"layer FP: {data['sentinel']['false_positive_rate']*100:.1f}%"
    )
    print(f"Attribution outcomes: {attr['outcomes']}")
    print(f"Layer blocks: {data.get('blocking_layers', {})}")


if __name__ == "__main__":
    main()
