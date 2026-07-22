"""Convert the SENTINEL 226-prompt benchmark into AutoDefense's JSON schema.

Only the file format changes: prompt text, ordering, labels and metadata are
preserved exactly. AutoDefense's attack.py expects a {name: prompt} dict, and it
wraps adversarial prompts with a jailbreak template (v1) while benign prompts are
passed through untouched (placeholder). We therefore emit two dicts keyed by the
original SENTINEL id, plus a sidecar mapping id -> category/source/variant so the
evaluation can reproduce SENTINEL's per-category metrics.
"""

import argparse
import csv
import json

from _paths import SENTINEL_DIR, autodefense_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--autodefense-dir", help="Path to an AutoDefense checkout")
    args = ap.parse_args()

    csv_path = SENTINEL_DIR / "benchmark" / "prompts.csv"
    out_dir = autodefense_dir(args.autodefense_dir) / "data" / "prompt"

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    adversarial, benign, meta = {}, {}, {}
    for row in rows:
        rid, cat = row["id"], row["category"]
        (benign if cat == "benign" else adversarial)[rid] = row["prompt"]
        meta[rid] = {"category": cat, "source": row["source"], "variant": row["variant"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [
        ("sentinel_adversarial.json", adversarial),
        ("sentinel_benign.json", benign),
        ("sentinel_meta.json", meta),
    ]:
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"adversarial={len(adversarial)} benign={len(benign)} total={len(meta)} -> {out_dir}")
    assert len(meta) == 226 and len(benign) == 100 and len(adversarial) == 126


if __name__ == "__main__":
    main()
