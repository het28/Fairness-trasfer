#!/usr/bin/env python3
"""Copy/refresh paper tables from frozen primary CSVs into tables/paper/.

Does not retrain. Does not alter scientific values.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/primary"
POST = ROOT / "results/post_learning"
EXT = ROOT / "results/external"
OUT = ROOT / "tables/paper"
OUT.mkdir(parents=True, exist_ok=True)

MAP = {
    "rq1_global.csv": SRC / "rq1_global.csv",
    "rq1_by_family.csv": SRC / "rq1_by_family.csv",
    "rq2_cells.csv": SRC / "rq2_cells.csv",
    "rq2_regression.csv": SRC / "rq2_regression.csv",
    "rq3_failures.csv": SRC / "rq3_failures.csv",
    "utility_tradeoff.csv": SRC / "utility_tradeoff.csv",
    "post_learning_matched_48.csv": POST / "matched_48_export.csv",
    "external.csv": EXT / "external_summary.csv",
}


def main():
    for dest_name, src in MAP.items():
        if not src.exists():
            # fallbacks
            alt = OUT / dest_name
            if alt.exists():
                print("skip missing", src, "(existing table kept)")
                continue
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, OUT / dest_name)
        print("wrote", OUT / dest_name)
    # also write a compact LaTeX snippet pointing to CSVs
    tex = OUT / "tables_include.tex"
    tex.write_text(
        "% Auto-copied frozen tables for LNCS inclusion.\n"
        "% Prefer \\input of CSV via packages or paste from tables/paper/*.csv\n"
        "% See docs/RESULTS.md for mapping to manuscript claims.\n"
    )
    print("done")


if __name__ == "__main__":
    main()
