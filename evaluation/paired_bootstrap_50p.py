"""
paired_bootstrap_50p.py — Paired bootstrap on the combined 50-persona cohort.

Combines per-persona accuracy from the 20-persona ablation suites (pid_1..20)
and the 30-persona 50p ablation suites (pid_21..50), for v2_inferred and
v3_maximum, reasoning=high, across {bg, bg_dp, bg_ep, bg_dp_ep}.

Runs paired bootstrap with B=10_000 iterations, at sample sizes m=25 and m=50.

Usage:
    python evaluation/paired_bootstrap_50p.py
    python evaluation/paired_bootstrap_50p.py --B 10000 --sizes 25 50
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

from paired_bootstrap import (
    RANGES, PRICING_COLS, _apply_decile_norm, _per_persona_accuracy,
)

BASE = Path("text_simulation")
CONFIG_MAP = {
    "bg":       "bg",
    "bg_dp":    "bg+dp",
    "bg_ep":    "bg+ep",
    "bg_dp_ep": "bg+dp+ep",
}
VERSION_SUITES = {
    # version_label -> list of (suite_dir, expected_pids)
    "v2_inferred": [
        ("text_simulation_output_nano_v2_ablation_v2_temp0",    list(range(1, 21))),
        ("text_simulation_output_nano_v2_ablation_50p_v2_temp0", list(range(21, 51))),
    ],
    "v3_maximum": [
        ("text_simulation_output_nano_v2_ablation_v3_temp0",    list(range(1, 21))),
        ("text_simulation_output_nano_v2_ablation_50p_v3_temp0", list(range(21, 51))),
    ],
}

PRICING_SET = {c.upper() for c in PRICING_COLS}


def load_suite_accuracy(suite_dir: str, condition: str) -> dict[str, dict[int, float]]:
    """Return {'overall': {pid: acc}, 'cognitive_bias': {...}, 'pricing': {...}}."""
    fmt = BASE / suite_dir / condition / "high" / "csv_comparison" / "csv_formatted"
    llm = pd.read_csv(fmt / "responses_llm_imputed_formatted.csv", skiprows=[1])
    w13 = pd.read_csv(fmt / "responses_wave1_3_formatted.csv",      skiprows=[1])
    llm.columns = llm.columns.str.upper()
    w13.columns = w13.columns.str.upper()
    for df in (llm, w13):
        for col in df.columns:
            if col != "TWIN_ID":
                df[col] = pd.to_numeric(df[col], errors="coerce")
    _apply_decile_norm(w13, llm)
    all_cols     = [c for c in llm.columns if c != "TWIN_ID" and c in RANGES]
    pricing_cols = [c for c in all_cols if c in PRICING_SET]
    bias_cols    = [c for c in all_cols if c not in PRICING_SET]
    return {
        "overall":        _per_persona_accuracy(llm.copy(), w13.copy(), all_cols),
        "cognitive_bias": _per_persona_accuracy(llm.copy(), w13.copy(), bias_cols),
        "pricing":        _per_persona_accuracy(llm.copy(), w13.copy(), pricing_cols),
    }


def load_combined_df() -> pd.DataFrame:
    rows = []
    for version, suites in VERSION_SUITES.items():
        for suite_dir, _ in suites:
            for condition, config_label in CONFIG_MAP.items():
                accs = load_suite_accuracy(suite_dir, condition)
                for metric, per_pid in accs.items():
                    for pid, acc in per_pid.items():
                        rows.append({
                            "persona_id": int(pid),
                            "config":     config_label,
                            "metric":     metric,
                            "version":    version,
                            "accuracy":   acc,
                        })
    return pd.DataFrame(rows)


def paired_bootstrap(acc_A: np.ndarray, acc_B: np.ndarray,
                     B: int, m: int, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(acc_A)
    idx = rng.integers(0, n, size=(B, m))
    delta_b = acc_B[idx].mean(axis=1) - acc_A[idx].mean(axis=1)
    rho, _ = pearsonr(acc_A, acc_B)
    return {
        "n":           n,
        "m":           m,
        "delta_point": float(acc_B.mean() - acc_A.mean()),
        "ci_lo":       float(np.percentile(delta_b, 100 * alpha / 2)),
        "ci_hi":       float(np.percentile(delta_b, 100 * (1 - alpha / 2))),
        "se":          float(delta_b.std(ddof=1)),
        "rho":         float(rho),
    }


CONTRASTS = [
    ("bg",      "bg+ep",    "overall"),
    ("bg",      "bg+ep",    "cognitive_bias"),
    ("bg",      "bg+ep",    "pricing"),
    ("bg",      "bg+dp",    "overall"),
    ("bg",      "bg+dp",    "cognitive_bias"),
    ("bg",      "bg+dp",    "pricing"),
    ("bg+ep",   "bg+dp+ep", "overall"),
    ("bg+ep",   "bg+dp+ep", "cognitive_bias"),
    ("bg+ep",   "bg+dp+ep", "pricing"),
    ("bg+dp",   "bg+dp+ep", "overall"),
    ("bg+dp",   "bg+dp+ep", "cognitive_bias"),
    ("bg+dp",   "bg+dp+ep", "pricing"),
    ("bg",      "bg+dp+ep", "overall"),
    ("bg",      "bg+dp+ep", "cognitive_bias"),
    ("bg",      "bg+dp+ep", "pricing"),
]


def run(df: pd.DataFrame, version: str, B: int, sizes: list[int]) -> pd.DataFrame:
    out_rows = []
    for config_A, config_B, metric in CONTRASTS:
        sl = df[(df["version"] == version) & (df["metric"] == metric)]
        wide = sl.pivot_table(index="persona_id", columns="config", values="accuracy")
        if config_A not in wide.columns or config_B not in wide.columns:
            continue
        wide = wide.dropna(subset=[config_A, config_B])
        acc_A = wide[config_A].values.astype(float)
        acc_B = wide[config_B].values.astype(float)
        for m in sizes:
            res = paired_bootstrap(acc_A, acc_B, B=B, m=m)
            sig = not (res["ci_lo"] <= 0 <= res["ci_hi"])
            out_rows.append({
                "version":       version,
                "contrast":      f"{config_B} vs {config_A}",
                "metric":        metric,
                "n_personas":    res["n"],
                "sample_size_m": m,
                "delta_point":   round(res["delta_point"], 4),
                "ci_lo":         round(res["ci_lo"], 4),
                "ci_hi":         round(res["ci_hi"], 4),
                "se":            round(res["se"], 4),
                "rho":           round(res["rho"], 3),
                "significant":   sig,
            })
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B",     type=int, default=10_000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 50])
    parser.add_argument("--out",   default="evaluation/paired_bootstrap_50p_results.csv")
    args = parser.parse_args()

    print("Loading per-persona accuracy (20p + 30p suites)...")
    df = load_combined_df()
    print(f"  total rows: {len(df)}")
    for v in df["version"].unique():
        pids = sorted(df[df["version"] == v]["persona_id"].unique())
        print(f"  {v}: n_personas={len(pids)}  range={pids[0]}..{pids[-1]}")

    all_out = []
    for version in sorted(df["version"].unique()):
        print(f"\n{'='*72}\nPaired bootstrap — {version}  (B={args.B:,}, sizes={args.sizes})\n{'='*72}")
        summary = run(df, version, B=args.B, sizes=args.sizes)
        print(summary.to_string(index=False))
        all_out.append(summary)

    full = pd.concat(all_out, ignore_index=True)
    out_path = Path(args.out)
    full.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
