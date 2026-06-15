"""Run Track B end-to-end evaluation: baseline LLM vs SENTINEL L1-5."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import (
    ATTRIBUTION_INCREMENTAL,
    baseline_refused,
    blocking_layer,
    category_summary,
    classify_attribution,
    compute_attribution,
    end_to_end_blocked,
    false_positive,
    harmful_compliance,
    is_layer_block,
    is_refusal,
    pipeline_blocked,
    summarize,
)
from layers.llm_core import LocalLLM
from main import IAJDAPipeline, configure_logging, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_baseline_llm(llm: LocalLLM, prompt: str, refusal_message: str) -> dict[str, Any]:
    """Evaluate a prompt with the raw local model only."""
    response = llm.generate(prompt)
    blocked = is_refusal(response) or response.strip() == refusal_message.strip()
    return {
        "final_response": response,
        "blocked": blocked,
        "attack_success": not blocked,
        "harmful_compliance": harmful_compliance(response),
    }


def run_sentinel(
    pipeline: IAJDAPipeline,
    prompt: str,
    *,
    baseline_response: str,
    category: str,
) -> dict[str, Any]:
    """Evaluate a prompt through the full SENTINEL pipeline."""
    result = pipeline.evaluate(
        prompt,
        enabled_layers=frozenset({1, 2, 3, 4, 5}),
        skip_llm=False,
        record_latency=True,
    )
    layer = blocking_layer(result)
    layer_blocked = is_layer_block(layer)
    attribution = classify_attribution(
        category=category,
        blocking_layer=layer,
        sentinel_response=str(result["final_response"]),
        baseline_response=baseline_response,
    )
    base_refused = baseline_refused(baseline_response)
    incremental = attribution == ATTRIBUTION_INCREMENTAL
    true_asr = not layer_blocked and not is_refusal(str(result["final_response"]))

    return {
        **result,
        "blocking_layer": layer,
        "layer_blocked": layer_blocked,
        "incremental_block": incremental,
        "attribution": attribution,
        "baseline_refused": base_refused,
        "true_asr": true_asr,
        "end_to_end_blocked": end_to_end_blocked({**result, "blocking_layer": layer}),
        "blocked": layer_blocked,
        "attack_success": true_asr,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_mode(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    attack_rows = [row for row in rows if row["category"] != "benign"]
    benign_rows = [row for row in rows if row["category"] == "benign"]

    def is_true(value: Any) -> bool:
        return str(value).lower() in {"true", "1", "yes"}

    if mode == "baseline":
        attack_successes = sum(1 for row in attack_rows if is_true(row["attack_success"]))
        benign_fps = sum(1 for row in benign_rows if is_true(row["blocked"]))
        blocked_rows = [
            {
                "blocked": is_true(row["blocked"]),
                "category": row["category"],
                "false_positive": False,
                "total_ms": row.get("total_ms", "0"),
            }
            for row in rows
        ]
        return {
            "total_prompts": len(rows),
            "attack_prompts": len(attack_rows),
            "benign_prompts": len(benign_rows),
            "attack_success_rate": attack_successes / max(len(attack_rows), 1),
            "attack_block_rate": 1.0 - (attack_successes / max(len(attack_rows), 1)),
            "false_positive_count": benign_fps,
            "false_positive_rate": benign_fps / max(len(benign_rows), 1),
            "categories": category_summary(blocked_rows),
        }

    layer_blocks = sum(1 for row in attack_rows if is_true(row.get("layer_blocked", row["blocked"])))
    incremental_blocks = sum(1 for row in attack_rows if is_true(row.get("incremental_block", False)))
    true_asr_count = sum(1 for row in attack_rows if is_true(row.get("true_asr", row["attack_success"])))
    end_to_end_blocks = sum(
        1
        for row in attack_rows
        if is_true(row.get("end_to_end_blocked", False)) or is_true(row["blocked"])
    )
    benign_fps = sum(1 for row in benign_rows if is_true(row.get("false_positive", row["blocked"])))
    blocked_rows = [
        {
            "blocked": is_true(row.get("layer_blocked", row["blocked"])),
            "category": row["category"],
            "false_positive": is_true(row.get("false_positive", False)),
            "total_ms": row.get("total_ms", "0"),
        }
        for row in rows
    ]
    attack_n = max(len(attack_rows), 1)
    return {
        "total_prompts": len(rows),
        "attack_prompts": len(attack_rows),
        "benign_prompts": len(benign_rows),
        "attack_success_rate": true_asr_count / attack_n,
        "attack_block_rate": layer_blocks / attack_n,
        "layer_attributed_block_rate": layer_blocks / attack_n,
        "incremental_block_rate": incremental_blocks / attack_n,
        "incremental_block_count": incremental_blocks,
        "end_to_end_block_rate": end_to_end_blocks / attack_n,
        "true_asr_rate": true_asr_count / attack_n,
        "false_positive_count": benign_fps,
        "false_positive_rate": benign_fps / max(len(benign_rows), 1),
        "categories": category_summary(blocked_rows),
    }


def sanitize_csv(value: str, limit: int = 500) -> str:
    return value[:limit].replace("\n", " ").replace("\r", " ")


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as file:
        return {row["id"] for row in csv.DictReader(file)}


def append_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(
    results_dir: Path,
    model_name: str,
    config_path: Path,
    baseline_path: Path,
    sentinel_path: Path,
) -> dict[str, Any]:
    baseline_rows = list(csv.DictReader(baseline_path.open(encoding="utf-8")))
    sentinel_rows = list(csv.DictReader(sentinel_path.open(encoding="utf-8")))
    baseline_summary = aggregate_mode(baseline_rows, mode="baseline")
    sentinel_summary = aggregate_mode(sentinel_rows, mode="sentinel")
    attribution = compute_attribution(baseline_rows, sentinel_rows)

    if sentinel_rows:
        sentinel_summary["average_latency_ms"] = summarize(
            [
                {
                    "blocked": row.get("layer_blocked", row["blocked"]) == "True",
                    "category": row["category"],
                    "total_ms": row.get("total_ms", "0"),
                }
                for row in sentinel_rows
            ]
        )["average_latency_ms"]

    output = {
        "model": model_name,
        "config": str(config_path),
        "completed_prompts": len(sentinel_rows),
        "baseline": baseline_summary,
        "sentinel": sentinel_summary,
        "attribution": attribution,
        "blocking_layers": attribution["layer_blocks_by_layer"],
    }
    summary_path = results_dir / "endtoend_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    return output


def load_prompts(prompts_path: Path, *, adversarial_first: bool) -> list[dict[str, str]]:
    with prompts_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if adversarial_first:
        rows.sort(key=lambda row: (row["category"] == "benign", int(row["id"])))
    return rows


def run_endtoend(
    config_path: Path,
    *,
    limit: int | None = None,
    resume: bool = True,
    skip_benign_baseline: bool = True,
    adversarial_first: bool = True,
) -> None:
    config = load_config(config_path)
    configure_logging(config)
    pipeline = IAJDAPipeline(config, clean_mode=False, use_color=False)
    llm = pipeline.llm
    refusal_message = str(config["app"]["refusal_message"])
    model_name = str(config["lm_studio"]["model"])

    prompts_path = PROJECT_ROOT / "benchmark" / "prompts.csv"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = results_dir / "endtoend_baseline.csv"
    sentinel_path = results_dir / "endtoend_sentinel.csv"
    completed = load_completed_ids(sentinel_path) if resume else set()

    baseline_fields = [
        "id", "category", "mode", "model", "blocked", "attack_success",
        "harmful_compliance", "response",
    ]
    sentinel_fields = [
        "id", "category", "mode", "model", "layer_blocked", "incremental_block",
        "attribution", "baseline_refused", "true_asr", "end_to_end_blocked",
        "blocked", "attack_success", "false_positive", "blocking_layer",
        "total_ms", "layer4_ms", "layer5_ms", "response",
    ]

    prompt_rows = load_prompts(prompts_path, adversarial_first=adversarial_first)
    total = len(prompt_rows) if limit is None else min(limit, len(prompt_rows))

    for index, item in enumerate(prompt_rows[:total], start=1):
        if item["id"] in completed:
            continue
        prompt = item["prompt"]
        category = item["category"]

        if skip_benign_baseline and category == "benign":
            baseline = {
                "final_response": "[BASELINE_SKIPPED_BENIGN]",
                "blocked": False,
                "attack_success": False,
                "harmful_compliance": False,
            }
        else:
            baseline = run_baseline_llm(llm, prompt, refusal_message)

        baseline_response = sanitize_csv(baseline["final_response"])
        append_row(
            baseline_path,
            {
                "id": item["id"],
                "category": category,
                "mode": "baseline",
                "model": model_name,
                "blocked": baseline["blocked"],
                "attack_success": baseline["attack_success"],
                "harmful_compliance": baseline["harmful_compliance"],
                "response": baseline_response,
            },
            baseline_fields,
        )

        sentinel = run_sentinel(
            pipeline,
            prompt,
            baseline_response=baseline_response,
            category=category,
        )
        append_row(
            sentinel_path,
            {
                "id": item["id"],
                "category": category,
                "mode": "sentinel",
                "model": model_name,
                "layer_blocked": sentinel["layer_blocked"],
                "incremental_block": sentinel["incremental_block"],
                "attribution": sentinel["attribution"],
                "baseline_refused": sentinel["baseline_refused"],
                "true_asr": sentinel["true_asr"],
                "end_to_end_blocked": sentinel["end_to_end_blocked"],
                "blocked": sentinel["layer_blocked"],
                "attack_success": sentinel["true_asr"],
                "false_positive": false_positive(sentinel, category),
                "blocking_layer": sentinel["blocking_layer"],
                "total_ms": f"{sentinel['latency']['total_ms']:.2f}",
                "layer4_ms": f"{sentinel['latency']['layer4_ms']:.2f}",
                "layer5_ms": f"{sentinel['latency']['layer5_ms']:.2f}",
                "response": sanitize_csv(str(sentinel["final_response"])),
            },
            sentinel_fields,
        )
        completed.add(item["id"])
        print(
            f"[{len(completed)}/226] id={item['id']} {category}: "
            f"baseline={'BLOCK' if baseline['blocked'] else 'PASS'} "
            f"layer={'BLOCK' if sentinel['layer_blocked'] else 'PASS'} "
            f"({sentinel['blocking_layer']}, {sentinel['attribution']})",
            flush=True,
        )
        write_summary(
            results_dir,
            model_name,
            config_path,
            baseline_path,
            sentinel_path,
        )

    output = write_summary(
        results_dir,
        model_name,
        config_path,
        baseline_path,
        sentinel_path,
    )
    attr = output["attribution"]
    print(f"\nModel: {model_name}")
    print(
        f"Baseline ASR: {output['baseline']['attack_success_rate']*100:.1f}% | "
        f"block: {output['baseline']['attack_block_rate']*100:.1f}%"
    )
    print(
        f"SENTINEL layer block: {output['sentinel']['layer_attributed_block_rate']*100:.1f}% | "
        f"incremental: {attr['incremental_block_rate']*100:.1f}% | "
        f"true ASR: {attr['true_asr_rate']*100:.1f}% | "
        f"FP: {output['sentinel']['false_positive_rate']*100:.1f}%"
    )
    print(f"Wrote {results_dir / 'endtoend_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Track B end-to-end benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.ollama.yaml",
        help="YAML config with OpenAI-compatible local model settings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional prompt limit for smoke tests.",
    )
    parser.add_argument(
        "--no-adversarial-first",
        action="store_true",
        help="Process prompts in CSV order instead of adversarial-first.",
    )
    args = parser.parse_args()
    run_endtoend(
        args.config,
        limit=args.limit,
        adversarial_first=not args.no_adversarial_first,
    )
