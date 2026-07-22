"""Run AutoDefense's 1/2/3-agent defenses over the SENTINEL 226-prompt set on Ollama.

The defense pipeline itself is AutoDefense's, untouched: we import its detector
agencies and ExplicitMultiAgentDefense and call defense_with_response exactly as
evaluate_helper does. What this runner adds over the stock run_defense_exp.py:
  * points inference at Ollama (localhost:11434, llama3.1:8b) via load_llm_config,
  * a single pass per prompt (repetition=1) to match SENTINEL's single-pass eval,
  * per-prompt wall-clock latency, measured sequentially so it is comparable to
    SENTINEL's sequential latency numbers,
  * resumability (skip prompts already written).

Strategies map to the paper's agent counts: ex-1 = 1-Agent, ex-2 = 2-Agent,
ex-3 = 3-Agent. Requires an AutoDefense checkout (see _paths.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from _paths import autodefense_dir

ROOT: Path  # AutoDefense checkout, resolved in main()


def build_registry():
    from defense.explicit_detector.agency.explicit_1_agent import VanillaJailbreakDetector
    from defense.explicit_detector.agency.explicit_2_agents import AutoGenDetectorV1
    from defense.explicit_detector.agency.explicit_3_agents import AutoGenDetectorThreeAgency
    return {"ex-1": VanillaJailbreakDetector, "ex-2": AutoGenDetectorV1, "ex-3": AutoGenDetectorThreeAgency}


def load_meta() -> dict:
    return json.loads((ROOT / "data" / "prompt" / "sentinel_meta.json").read_text(encoding="utf-8"))


def defend_one(task_agency_cls, raw_response, model, host, port, temperature):
    """One faithful AutoDefense pass; returns (final_content, latency_ms)."""
    from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
    from defense.utility import load_llm_config

    cfg = load_llm_config(model_name=model, host_name=host, port=port,
                          cache_seed=123, temperature=temperature)
    defense = ExplicitMultiAgentDefense(
        task_agency=task_agency_cls(config_list=cfg), config_list=cfg)
    start = time.perf_counter()
    try:
        content = defense.defense_with_response(raw_response)["content"]
    except Exception as exc:  # keep going; record the failure like AutoDefense's ERROR path
        content = f"[DEFENSE_EXCEPTION: {exc}]"
    return content, (time.perf_counter() - start) * 1000.0


def run(strategy, split, registry, model, host, port, temperature):
    model_dir = model.replace("/", "-")
    attack = json.loads((ROOT / "data" / "harmful_output" / model_dir / f"{split}_0.json").read_text(encoding="utf-8"))
    meta = load_meta()

    out_dir = ROOT / "data" / "defense_output" / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{strategy}_{split}.json"
    done = {}
    if out_file.exists():
        done = {r["name"]: r for r in json.loads(out_file.read_text(encoding="utf-8"))}

    task_cls = registry[strategy]
    results = list(done.values())
    total = len(attack)
    for i, item in enumerate(attack, 1):
        name = item["name"]
        if name in done:
            continue
        content, ms = defend_one(task_cls, item["raw_response"], model, host, port, temperature)
        results.append({
            "name": name,
            "category": meta.get(name, {}).get("category", "unknown"),
            "raw_response": item["raw_response"],
            "defense_response": content,
            "latency_ms": round(ms, 2),
        })
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{strategy}/{split} {i}/{total}] {name} ({results[-1]['category']}) {ms:.0f}ms", flush=True)
    print(f"wrote {out_file} ({len(results)} rows)")


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--autodefense-dir", help="Path to an AutoDefense checkout")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--strategies", nargs="+", default=["ex-1", "ex-2", "ex-3"])
    ap.add_argument("--splits", nargs="+", default=["adversarial", "benign"])
    args = ap.parse_args()

    ROOT = autodefense_dir(args.autodefense_dir)
    sys.path.insert(0, str(ROOT))
    registry = build_registry()

    for strategy in args.strategies:
        for split in args.splits:
            run(strategy, split, registry, args.model, args.host, args.port, args.temperature)


if __name__ == "__main__":
    main()
