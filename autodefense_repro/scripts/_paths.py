"""Shared path resolution for the AutoDefense reproduction.

Nothing here is machine-specific. SENTINEL_DIR is the repo this file lives in;
AUTODEFENSE_DIR points at a local AutoDefense checkout (https://github.com/XHMY/AutoDefense),
taken from the --autodefense-dir flag, the AUTODEFENSE_DIR env var, or a sibling
`AutoDefense/` directory next to the SENTINEL repo, in that order.
"""

import os
from pathlib import Path

SENTINEL_DIR = Path(__file__).resolve().parents[2]
REPRO_DIR = Path(__file__).resolve().parents[1]


def autodefense_dir(cli_value: str | None = None) -> Path:
    candidate = cli_value or os.getenv("AUTODEFENSE_DIR") or str(SENTINEL_DIR.parent / "AutoDefense")
    path = Path(candidate).expanduser().resolve()
    if not (path / "defense" / "utility.py").exists():
        raise SystemExit(
            f"AutoDefense checkout not found at {path}. Clone https://github.com/XHMY/AutoDefense "
            f"and pass --autodefense-dir or set AUTODEFENSE_DIR."
        )
    return path
