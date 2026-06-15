#!/usr/bin/env bash
# Poll Track B benchmark and refresh paper artifacts when enough data exists.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

while true; do
  if [[ -f results/endtoend_summary.json ]]; then
    attack=$(python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("results/endtoend_summary.json").read_text())
print(d["baseline"]["attack_prompts"])
PY
)
    completed=$(python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("results/endtoend_summary.json").read_text())
print(d.get("completed_prompts", 0))
PY
)
    if (( attack >= 10 )); then
      python3 benchmark/update_paper_trackb.py || true
      python3 benchmark/generate_endtoend_figure.py || true
      (cd paper && pdflatex -interaction=nonstopmode final-paper.tex >/dev/null 2>&1) || true
      echo "$(date -Iseconds) updated paper (completed=$completed attack=$attack)"
    fi
    if (( completed >= 226 )); then
      echo "Track B complete."
      break
    fi
  fi
  sleep 300
done
