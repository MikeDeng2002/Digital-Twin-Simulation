"""
bootstrap_ci.py — Bootstrap 95% CI across personas for each ablation_fixed config.

For each config (version × condition):
  - Computes per-persona accuracy for overall, cognitive bias, and product preference
  - Bootstraps: 100 iterations, sample size 5 (with replacement from 20 personas)
  - Reports mean ± 95% CI for each metric

Usage (from Digital-Twin-Simulation/):
    python evaluation/bootstrap_ci.py
    python evaluation/bootstrap_ci.py --n_boot 1000 --sample_size 10
    python evaluation/bootstrap_ci.py --version v2
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

VERSIONS    = ["v1", "v2", "v3"]
CONDITIONS  = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]
BASE        = Path("text_simulation")

# Pricing column keys
PRICING_COLS = {f"{i}_Q295" for i in range(1, 41)}

# Column ranges for MAD normalization (same as eval_by_question_type.py)
def get_minmax():
    mm = {}
    mm.update({f"FALSE CONS. SELF _{i}": (1, 5) for i in list(range(1, 8)) + [10, 11, 12]})
    mm.update({f"FALSE CONS. OTHERS _{i}": (0, 100) for i in list(range(1, 8)) + [10, 11, 12]})
    mm["Q156_1"] = (0, 100); mm["FORM A _1"] = (0, 100)
    mm["Q157"] = (1, 6);  mm["Q158"] = (1, 6)
    for i in [1, 2, 3]: mm[f"Q159_{i}"] = (1, 6); mm[f"Q160_{i}"] = (1, 6)
    mm["Q161"] = (1, 7);  mm["Q162"] = (1, 7)
    for c in ["164", "166", "168", "170"]: mm[f"Q{c}"] = (1, 10)
    for c in ["171","172","173","174","175","176"]: mm[f"Q{c}"] = (1, 5)
    for c in ["177","178","179"]: mm[f"Q{c}"] = (1, 6)
    mm["Q181"] = (0, 20); mm["Q182"] = (0, 20)
    mm["Q183"] = (1, 2);  mm["Q184"] = (1, 2)
    mm["Q189"] = (1, 10); mm["Q190"] = (1, 10); mm["Q191"] = (1, 10)
    mm["Q192"] = (1, 2);  mm["Q193"] = (1, 2)
    mm["Q194"] = (1, 6);  mm["Q195"] = (1, 6)
    mm.update({f"Q198_{i}": (1, 2) for i in range(1, 11)})
    mm.update({f"Q203_{i}": (1, 2) for i in range(1, 7)})
    mm.update({f"NONSEPARABILTY BENE _{i}": (1, 7) for i in range(1, 5)})
    mm.update({f"NONSEPARABILITY RIS _{i}": (1, 7) for i in range(1, 5)})
    mm["OMISSION BIAS "] = (1, 4); mm["DENOMINATOR NEGLECT "] = (1, 2)
    mm.update({f"{i}_Q295": (1, 2) for i in range(1, 41)})
    return mm

MINMAX  = get_minmax()
RANGES  = {col: mx - mn for col, (mn, mx) in MINMAX.items()}

# Decile normalization columns
DECILE_GROUP_1 = ["Q164", "Q166"]
DECILE_GROUP_2 = ["Q168", "Q170"]


def assign_decile(value, thresholds):
    if pd.isna(value): return np.nan
    for i, t in enumerate(thresholds):
        if value <= t: return i + 1
    return 10


def apply_decile_norm(df_w13, df_llm):
    for cols in [DECILE_GROUP_1, DECILE_GROUP_2]:
        existing = [c for c in cols if c in df_w13.columns]
        if not existing: continue
        combined = pd.concat([df_w13[c] for c in existing]).dropna()
        if len(combined) == 0: continue
        thresholds = np.percentile(combined, np.arange(10, 100, 10))
        for col in existing:
            if col in df_w13.columns:
                df_w13[col] = df_w13[col].apply(lambda x: assign_decile(x, thresholds))
            if col in df_llm.columns:
                df_llm[col] = df_llm[col].apply(lambda x: assign_decile(x, thresholds))


def per_persona_accuracy(llm: pd.DataFrame, w13: pd.DataFrame, col_subset=None):
    """Return Series: TWIN_ID → mean MAD accuracy for the given column subset."""
    common_ids = set(llm["TWIN_ID"]) & set(w13["TWIN_ID"])
    llm = llm[llm["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")
    w13 = w13[w13["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")

    all_cols = [c for c in llm.columns if c in RANGES]
    if col_subset is not None:
        all_cols = [c for c in all_cols if c in col_subset]

    persona_acc = {}
    for pid in llm.index:
        accs = []
        for col in all_cols:
            if col not in llm.columns or col not in w13.columns: continue
            l = pd.to_numeric(llm.loc[pid, col], errors="coerce")
            w = pd.to_numeric(w13.loc[pid, col], errors="coerce")
            if pd.isna(l) or pd.isna(w): continue
            accs.append(1 - abs(l - w) / RANGES[col])
        persona_acc[pid] = np.mean(accs) if accs else np.nan

    return pd.Series(persona_acc)


def load_formatted(csv_dir: Path):
    fmt = csv_dir / "csv_formatted"
    try:
        llm = pd.read_csv(fmt / "responses_llm_imputed_formatted.csv", skiprows=[1])
        w13 = pd.read_csv(fmt / "responses_wave1_3_formatted.csv",      skiprows=[1])
    except FileNotFoundError:
        return None, None
    llm.columns = llm.columns.str.upper()
    w13.columns = w13.columns.str.upper()
    for df in [llm, w13]:
        for col in df.columns:
            if col != "TWIN_ID":
                df[col] = pd.to_numeric(df[col], errors="coerce")
    apply_decile_norm(w13, llm)
    return llm, w13


def bootstrap_ci(values: np.ndarray, n_boot: int, sample_size: int, seed: int = 42,
                  replace: bool = True):
    """Bootstrap (or subsample) mean and 95% CI.

    replace=True  → standard bootstrap (with replacement)
    replace=False → subsampling (without replacement); sample_size must be < len(values)
    """
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    effective_size = min(sample_size, len(values)) if not replace else sample_size
    boot_means = [
        rng.choice(values, size=effective_size, replace=replace).mean()
        for _ in range(n_boot)
    ]
    boot_means = np.array(boot_means)
    return round(boot_means.mean(), 4), round(np.percentile(boot_means, 2.5), 4), round(np.percentile(boot_means, 97.5), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot",      type=int, default=100,  help="Bootstrap iterations (default: 100)")
    parser.add_argument("--sample_size", type=int, default=5,    help="Personas per sample (default: 5)")
    parser.add_argument("--no_replace",  action="store_true",    help="Sample WITHOUT replacement (subsampling)")
    parser.add_argument("--version",     choices=VERSIONS + ["all"], default="all")
    args = parser.parse_args()

    replace  = not args.no_replace
    versions = VERSIONS if args.version == "all" else [args.version]
    rows = []

    pricing_set = {c.upper() for c in PRICING_COLS}

    for version in versions:
        suite   = f"nano_v2_ablation_fixed_{version}_temp0"
        base_out = BASE / f"text_simulation_output_{suite}"

        for condition in CONDITIONS:
            trial_dir = base_out / condition / "high"
            csv_dir   = trial_dir / "csv_comparison"

            if not csv_dir.exists():
                print(f"  SKIP {version}/{condition} — no csv_comparison")
                continue

            llm, w13 = load_formatted(csv_dir)
            if llm is None:
                print(f"  SKIP {version}/{condition} — formatted CSVs missing")
                continue

            all_cols     = [c for c in llm.columns if c != "TWIN_ID" and c in RANGES]
            pricing_cols = [c for c in all_cols if c in pricing_set]
            bias_cols    = [c for c in all_cols if c not in pricing_set]

            acc_all     = per_persona_accuracy(llm.copy(), w13.copy(), all_cols)
            acc_bias    = per_persona_accuracy(llm.copy(), w13.copy(), bias_cols)
            acc_pricing = per_persona_accuracy(llm.copy(), w13.copy(), pricing_cols)

            mean_all,  lo_all,  hi_all  = bootstrap_ci(acc_all.values,     args.n_boot, args.sample_size, replace=replace)
            mean_bias, lo_bias, hi_bias = bootstrap_ci(acc_bias.values,    args.n_boot, args.sample_size, replace=replace)
            mean_pric, lo_pric, hi_pric = bootstrap_ci(acc_pricing.values, args.n_boot, args.sample_size, replace=replace)

            print(f"  {version}/{condition:12s}  "
                  f"overall={mean_all:.3f} [{lo_all:.3f},{hi_all:.3f}]  "
                  f"bias={mean_bias:.3f} [{lo_bias:.3f},{hi_bias:.3f}]  "
                  f"pricing={mean_pric:.3f} [{lo_pric:.3f},{hi_pric:.3f}]")

            rows.append({
                "version": version, "condition": condition,
                "overall_mean": mean_all, "overall_ci_lo": lo_all,  "overall_ci_hi": hi_all,
                "bias_mean":    mean_bias, "bias_ci_lo":   lo_bias,  "bias_ci_hi":   hi_bias,
                "pricing_mean": mean_pric, "pricing_ci_lo": lo_pric, "pricing_ci_hi": hi_pric,
            })

    df = pd.DataFrame(rows)
    out = Path("evaluation/ablation_fixed_bootstrap_ci.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    # Summary pivot
    for metric in ["overall", "bias", "pricing"]:
        print(f"\n{'='*60}")
        method = "without replacement" if args.no_replace else "with replacement"
        print(f"{metric.upper()} — mean [95% CI] (n={args.n_boot}, sample={args.sample_size}, {method})")
        print(f"{'='*60}")
        for version in versions:
            sub = df[df["version"] == version]
            print(f"\n  {version}:")
            for _, row in sub.iterrows():
                cond = row["condition"]
                m  = row[f"{metric}_mean"]
                lo = row[f"{metric}_ci_lo"]
                hi = row[f"{metric}_ci_hi"]
                print(f"    {cond:12s}  {m:.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
