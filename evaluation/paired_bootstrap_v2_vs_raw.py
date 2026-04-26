"""
paired_bootstrap_v2_vs_raw.py — Paired bootstrap of v2 skill configs vs the raw
baseline (no skill profile) on the same set of personas.

Contrasts:
  - v2 bg        vs raw
  - v2 bg+dp     vs raw
  - v2 bg+dp+ep  vs raw

Each on three metrics: overall, cognitive_bias, pricing.

Usage:
    python evaluation/paired_bootstrap_v2_vs_raw.py
    python evaluation/paired_bootstrap_v2_vs_raw.py --B 10000
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from paired_bootstrap import (
    RANGES, PRICING_COLS, _apply_decile_norm, _per_persona_accuracy,
)

BASE = Path("text_simulation")
PRICING_SET = {c.upper() for c in PRICING_COLS}

V2_FIX = BASE / "text_simulation_output_nano_v2_ablation_fixed_v2_temp0"  # v2 pids 1-20
V2_50P = BASE / "text_simulation_output_nano_v2_ablation_50p_v2_temp0"    # v2 pids 21-50
V3_FIX = BASE / "text_simulation_output_nano_v2_ablation_fixed_v3_temp0"  # v3 pids 1-20
V3_50P = BASE / "text_simulation_output_nano_v2_ablation_50p_v3_temp0"    # v3 pids 21-50

# raw is a single 50-pid suite; v2/v3 configs combine two non-overlapping 20+30 suites
SUITES = {
    "raw":         [BASE / "text_simulation_output_nano_temp0/raw/high"],
    "v2_bg":       [V2_FIX / "bg/high",       V2_50P / "bg/high"],
    "v2_bg_dp":    [V2_FIX / "bg_dp/high",    V2_50P / "bg_dp/high"],
    "v2_bg_ep":    [V2_FIX / "bg_ep/high",    V2_50P / "bg_ep/high"],
    "v2_bg_dp_ep": [V2_FIX / "bg_dp_ep/high", V2_50P / "bg_dp_ep/high"],
    "v3_bg":       [V3_FIX / "bg/high",       V3_50P / "bg/high"],
    "v3_bg_dp":    [V3_FIX / "bg_dp/high",    V3_50P / "bg_dp/high"],
    "v3_bg_ep":    [V3_FIX / "bg_ep/high",    V3_50P / "bg_ep/high"],
    "v3_bg_dp_ep": [V3_FIX / "bg_dp_ep/high", V3_50P / "bg_dp_ep/high"],
}

# Skill configs to compare against raw, in display order
SKILL_KEYS = [
    "v2_bg", "v2_bg_dp", "v2_bg_ep", "v2_bg_dp_ep",
    "v3_bg", "v3_bg_dp", "v3_bg_ep", "v3_bg_dp_ep",
]


def _load_one(suite_dir: Path):
    fmt = suite_dir / "csv_comparison" / "csv_formatted"
    llm = pd.read_csv(fmt / "responses_llm_imputed_formatted.csv", skiprows=[1])
    w13 = pd.read_csv(fmt / "responses_wave1_3_formatted.csv",      skiprows=[1])
    llm.columns = llm.columns.str.upper()
    w13.columns = w13.columns.str.upper()
    for df in (llm, w13):
        for col in df.columns:
            if col != "TWIN_ID":
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return llm, w13


def load_per_persona(suite_dirs) -> dict[str, dict[int, float]]:
    """Load and concat one or more suites (non-overlapping pids), then compute accuracy."""
    if isinstance(suite_dirs, Path):
        suite_dirs = [suite_dirs]
    llm_parts, w13_parts = [], []
    for d in suite_dirs:
        l, w = _load_one(d)
        llm_parts.append(l); w13_parts.append(w)
    llm = pd.concat(llm_parts, ignore_index=True).drop_duplicates(subset="TWIN_ID")
    w13 = pd.concat(w13_parts, ignore_index=True).drop_duplicates(subset="TWIN_ID")
    _apply_decile_norm(w13, llm)
    all_cols     = [c for c in llm.columns if c != "TWIN_ID" and c in RANGES]
    pricing_cols = [c for c in all_cols if c in PRICING_SET]
    bias_cols    = [c for c in all_cols if c not in PRICING_SET]
    return {
        "overall":        _per_persona_accuracy(llm.copy(), w13.copy(), all_cols),
        "cognitive_bias": _per_persona_accuracy(llm.copy(), w13.copy(), bias_cols),
        "pricing":        _per_persona_accuracy(llm.copy(), w13.copy(), pricing_cols),
    }


def paired_bootstrap(acc_A: np.ndarray, acc_B: np.ndarray,
                     B: int, m: int, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n   = len(acc_A)
    idx = rng.integers(0, n, size=(B, m))
    delta_b = acc_B[idx].mean(axis=1) - acc_A[idx].mean(axis=1)
    rho, _ = pearsonr(acc_A, acc_B)
    p_two = float(2 * min((delta_b <= 0).mean(), (delta_b >= 0).mean()))
    return {
        "n":           n,
        "m":           m,
        "delta_point": float(acc_B.mean() - acc_A.mean()),
        "ci_lo":       float(np.percentile(delta_b, 100 * alpha / 2)),
        "ci_hi":       float(np.percentile(delta_b, 100 * (1 - alpha / 2))),
        "se":          float(delta_b.std(ddof=1)),
        "rho":         float(rho),
        "p_two_sided": p_two,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B",     type=int, default=10_000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50])
    ap.add_argument("--out",   default="evaluation/paired_bootstrap_v2_vs_raw.csv")
    args = ap.parse_args()

    accs = {key: load_per_persona(d) for key, d in SUITES.items()}

    rows = []
    for skill_key in SKILL_KEYS:
        version, _, variant = skill_key.partition("_")
        label = f"{version} {variant.replace('_', '+')}"
        for metric in ("overall", "cognitive_bias", "pricing"):
            raw_d   = accs["raw"][metric]
            skill_d = accs[skill_key][metric]
            common  = sorted(set(raw_d) & set(skill_d))
            acc_raw   = np.array([raw_d[p]   for p in common], dtype=float)
            acc_skill = np.array([skill_d[p] for p in common], dtype=float)

            for m in args.sizes:
                r = paired_bootstrap(acc_raw, acc_skill, B=args.B, m=m)
                sig = not (r["ci_lo"] <= 0 <= r["ci_hi"])
                better = version if r["delta_point"] > 0 else ("raw" if r["delta_point"] < 0 else "tie")
                rows.append({
                    "version":       version,
                    "config":        variant.replace("_", "+"),
                    "metric":        metric,
                    "contrast":      f"{label} minus raw",
                    "n_paired_pids": r["n"],
                    "sample_size_m": m,
                    "raw_mean":      round(float(acc_raw.mean()), 4),
                    "skill_mean":    round(float(acc_skill.mean()), 4),
                    "delta_point":   round(r["delta_point"], 4),
                    "ci_lo":         round(r["ci_lo"], 4),
                    "ci_hi":         round(r["ci_hi"], 4),
                    "se":            round(r["se"], 4),
                    "rho":           round(r["rho"], 3),
                    "p_two_sided":   round(r["p_two_sided"], 4),
                    "significant_95":sig,
                    "better":        better,
                })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
