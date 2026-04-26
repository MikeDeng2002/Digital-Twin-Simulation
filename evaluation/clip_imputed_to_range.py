"""
clip_imputed_to_range.py — Clip or null LLM-imputed cells that fall outside the
valid scale range for each column (per the MINMAX dict in paired_bootstrap.py).

This catches the documented outlier failure mode where the model emits a free-form
number (e.g. "10000" on a 1-10 scale, or a dollar amount on a 1-2 ordinal) instead
of staying on the requested scale. Two such cells in skill_v3/high were dragging
overall paired-bootstrap accuracy by ~24 points; clipping them brings results in
line with the rest of the column distribution.

Runs after evaluation/json2csv.py and before evaluation/mad_accuracy_evaluation.py.

Usage (from Digital-Twin-Simulation/):
    python evaluation/clip_imputed_to_range.py --config /path/to/eval.yaml [--mode clip|null]
    python evaluation/clip_imputed_to_range.py --csv path/to/responses_llm_imputed_formatted.csv
"""

import argparse, sys, yaml
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_bootstrap import MINMAX, DECILE_GROUP_1, DECILE_GROUP_2  # type: ignore

# Columns whose MINMAX range only applies *after* decile normalization in
# paired_bootstrap._apply_decile_norm. Their raw CSV values are dollar amounts
# (or otherwise unbounded) and must NOT be clipped here.
DECILE_NORMALIZED_COLS = {c.upper() for c in (*DECILE_GROUP_1, *DECILE_GROUP_2)}


def clip_csv(csv_path: Path, mode: str = "clip", verbose: bool = True) -> dict:
    if not csv_path.exists():
        return {"path": str(csv_path), "status": "missing", "n_cells": 0}

    # Row 0 is the canonical short-name header used everywhere downstream;
    # row 1 is the long question text. We preserve both.
    header_short = pd.read_csv(csv_path, nrows=0)
    df_meta = pd.read_csv(csv_path, skiprows=[0], nrows=1, header=None)
    df = pd.read_csv(csv_path, skiprows=[1])

    upper_to_actual = {c.upper(): c for c in df.columns}
    n_clipped = n_nulled = 0
    touched: list[str] = []

    for col_upper, (mn, mx) in MINMAX.items():
        if col_upper in DECILE_NORMALIZED_COLS:
            continue  # decile-normalized at scoring time; raw values are unbounded
        actual = upper_to_actual.get(col_upper)
        if actual is None or actual == "TWIN_ID":
            continue
        s = pd.to_numeric(df[actual], errors="coerce")
        out_of_range = (s < mn) | (s > mx)
        n_bad = int(out_of_range.fillna(False).sum())
        if n_bad == 0:
            continue
        touched.append(f"{actual}:{n_bad}")
        if mode == "clip":
            df.loc[out_of_range.fillna(False), actual] = s[out_of_range].clip(mn, mx)
            n_clipped += n_bad
        elif mode == "null":
            df.loc[out_of_range.fillna(False), actual] = np.nan
            n_nulled += n_bad
        else:
            raise ValueError(f"unknown mode: {mode}")

    # Reassemble: original short-name header, original long-text row, repaired data.
    tmp = csv_path.with_suffix(".clipped.tmp.csv")
    with open(tmp, "w", newline="") as f:
        f.write(",".join(header_short.columns) + "\n")
        df_meta.to_csv(f, header=False, index=False)
        df.to_csv(f, header=False, index=False)
    tmp.replace(csv_path)

    if verbose and (n_clipped or n_nulled):
        print(f"  [{mode}] {csv_path.name}: touched {n_clipped + n_nulled} cells "
              f"({', '.join(touched[:5])}{'...' if len(touched) > 5 else ''})")
    return {"path": str(csv_path), "status": "ok",
            "n_cells": n_clipped + n_nulled, "mode": mode, "columns_touched": touched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config (same one used by json2csv.py)")
    ap.add_argument("--csv", help="Direct path to a formatted CSV")
    ap.add_argument("--mode", choices=["clip", "null"], default="clip")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # llm_imputed lives at top level in some configs, under "waves" in eval_temp0_suite.
        imp = cfg.get("llm_imputed") or cfg.get("waves", {}).get("llm_imputed", {})
        for key in ("output_csv_formatted", "output_csv_labeled", "output_csv"):
            p = imp.get(key)
            if p:
                targets.append(Path(p))
    if args.csv:
        targets.append(Path(args.csv))

    if not targets:
        print("nothing to clip (provide --config or --csv)")
        return

    total = 0
    for t in targets:
        res = clip_csv(t, mode=args.mode)
        total += res.get("n_cells", 0)
    print(f"clip_imputed_to_range: {total} cell(s) {args.mode}ped across {len(targets)} file(s).")


if __name__ == "__main__":
    main()
