"""
paired_bootstrap_v2_vs_v4.py — Paired bootstrap comparing skill_v2 vs skill_v4
on the same 50 personas, for both bg+ep and bg+dp+ep configurations.

Reuses the per-persona accuracy machinery from paired_bootstrap.py.

Usage:
    python evaluation/paired_bootstrap_v2_vs_v4.py
    python evaluation/paired_bootstrap_v2_vs_v4.py --B 10000
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

# (version, variant) -> directory with csv_comparison/csv_formatted/
SUITES = {
    ("v2", "bg_ep"):    BASE / "text_simulation_output_nano_temp0/skill_v2_bg_ep_50p/high",
    ("v2", "bg_dp_ep"): BASE / "text_simulation_output_nano_temp0/skill_v2_bg_dp_ep_50p/high",
    ("v4", "bg_ep"):    BASE / "text_simulation_output_nano_temp0/skill_v4_bg_ep_50p/high",
    ("v4", "bg_dp_ep"): BASE / "text_simulation_output_nano_temp0/skill_v4_bg_dp_ep_50p/high",
}


def load_per_persona(suite_dir: Path) -> dict[str, dict[int, float]]:
    fmt = suite_dir / "csv_comparison" / "csv_formatted"
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


def paired_bootstrap(acc_A: np.ndarray, acc_B: np.ndarray,
                     B: int, m: int, seed: int = 42, alpha: float = 0.05) -> dict:
    """Sample m indices with replacement, compute mean diff, repeat B times."""
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
    ap.add_argument("--out",   default="evaluation/paired_bootstrap_v2_vs_v4.csv")
    args = ap.parse_args()

    # Load per-persona accuracy for all four suites
    accs = {key: load_per_persona(d) for key, d in SUITES.items()}

    rows = []
    for variant in ("bg_ep", "bg_dp_ep"):
        for metric in ("overall", "cognitive_bias", "pricing"):
            v2 = accs[("v2", variant)][metric]   # {pid: acc}
            v4 = accs[("v4", variant)][metric]
            common = sorted(set(v2) & set(v4))
            acc_v2 = np.array([v2[p] for p in common], dtype=float)
            acc_v4 = np.array([v4[p] for p in common], dtype=float)

            for m in args.sizes:
                r = paired_bootstrap(acc_v2, acc_v4, B=args.B, m=m)
                sig = not (r["ci_lo"] <= 0 <= r["ci_hi"])
                better = "v4" if r["delta_point"] > 0 else ("v2" if r["delta_point"] < 0 else "tie")
                rows.append({
                    "variant":       variant,
                    "metric":        metric,
                    "contrast":      f"v4 minus v2 ({variant})",
                    "n_paired_pids": r["n"],
                    "sample_size_m": m,
                    "v2_mean":       round(float(acc_v2.mean()), 4),
                    "v4_mean":       round(float(acc_v4.mean()), 4),
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
