"""
paired_bootstrap.py — Paired bootstrap confidence intervals on accuracy differences.

Paired bootstrap exploits the fact that each persona appears in both configurations
being compared. By resampling the same persona indices for both configs, we remove
between-persona variance and isolate the within-persona treatment effect. This gives
narrower, more powerful CIs than an unpaired bootstrap.

Usage (from Digital-Twin-Simulation/):
    python evaluation/paired_bootstrap.py
    python evaluation/paired_bootstrap.py --version v2
    python evaluation/paired_bootstrap.py --B 50000
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from scipy.stats import pearsonr

# ── Column ranges for MAD normalization (shared with other eval scripts) ──────

PRICING_COLS = {f"{i}_Q295" for i in range(1, 41)}

def _get_minmax() -> dict:
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

MINMAX = _get_minmax()
RANGES = {col: mx - mn for col, (mn, mx) in MINMAX.items()}

DECILE_GROUP_1 = ["Q164", "Q166"]
DECILE_GROUP_2 = ["Q168", "Q170"]

# ── Data loading ──────────────────────────────────────────────────────────────

VERSION_SHORT = {"v1": "v1_direct", "v2": "v2_inferred", "v3": "v3_maximum"}
CONFIG_MAP = {
    "bg":       "bg",
    "bg_dp":    "bg+dp",
    "bg_ep":    "bg+ep",
    "bg_dp_ep": "bg+dp+ep",
}
BASE = Path("text_simulation")


def _assign_decile(value: float, thresholds: np.ndarray) -> float:
    if pd.isna(value): return np.nan
    for i, t in enumerate(thresholds):
        if value <= t: return float(i + 1)
    return 10.0


def _apply_decile_norm(df_w13: pd.DataFrame, df_llm: pd.DataFrame) -> None:
    for cols in [DECILE_GROUP_1, DECILE_GROUP_2]:
        existing = [c for c in cols if c in df_w13.columns]
        if not existing: continue
        combined = pd.concat([df_w13[c] for c in existing]).dropna()
        if len(combined) == 0: continue
        thresholds = np.percentile(combined, np.arange(10, 100, 10))
        for col in existing:
            if col in df_w13.columns:
                df_w13[col] = df_w13[col].apply(lambda x: _assign_decile(x, thresholds))
            if col in df_llm.columns:
                df_llm[col] = df_llm[col].apply(lambda x: _assign_decile(x, thresholds))


def _per_persona_accuracy(
    llm: pd.DataFrame, w13: pd.DataFrame, col_subset: Optional[list] = None
) -> dict[int, float]:
    common = set(llm["TWIN_ID"]) & set(w13["TWIN_ID"])
    llm = llm[llm["TWIN_ID"].isin(common)].set_index("TWIN_ID")
    w13 = w13[w13["TWIN_ID"].isin(common)].set_index("TWIN_ID")
    all_cols = [c for c in llm.columns if c in RANGES]
    if col_subset is not None:
        all_cols = [c for c in all_cols if c in col_subset]
    result: dict[int, float] = {}
    for pid in llm.index:
        accs = []
        for col in all_cols:
            if col not in llm.columns or col not in w13.columns: continue
            l = pd.to_numeric(llm.loc[pid, col], errors="coerce")
            w = pd.to_numeric(w13.loc[pid, col], errors="coerce")
            if pd.isna(l) or pd.isna(w): continue
            accs.append(1.0 - abs(l - w) / RANGES[col])
        result[int(pid)] = float(np.mean(accs)) if accs else float("nan")
    return result


def load_accuracy_df(versions: list[str] = ["v1", "v2", "v3"]) -> pd.DataFrame:
    """Load per-persona accuracy into unified long-format DataFrame."""
    pricing_set = {c.upper() for c in PRICING_COLS}
    rows = []

    for version in versions:
        suite    = f"nano_v2_ablation_fixed_{version}_temp0"
        base_out = BASE / f"text_simulation_output_{suite}"
        v_long   = VERSION_SHORT.get(version, version)

        for condition, config_label in CONFIG_MAP.items():
            csv_dir = base_out / condition / "high" / "csv_comparison"
            if not csv_dir.exists():
                continue
            fmt = csv_dir / "csv_formatted"
            try:
                llm = pd.read_csv(fmt / "responses_llm_imputed_formatted.csv", skiprows=[1])
                w13 = pd.read_csv(fmt / "responses_wave1_3_formatted.csv",      skiprows=[1])
            except FileNotFoundError:
                continue
            llm.columns = llm.columns.str.upper()
            w13.columns = w13.columns.str.upper()
            for df in [llm, w13]:
                for col in df.columns:
                    if col != "TWIN_ID":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            _apply_decile_norm(w13, llm)

            all_cols     = [c for c in llm.columns if c != "TWIN_ID" and c in RANGES]
            pricing_cols = [c for c in all_cols if c in pricing_set]
            bias_cols    = [c for c in all_cols if c not in pricing_set]

            for metric, col_subset in [
                ("overall",        all_cols),
                ("cognitive_bias", bias_cols),
                ("pricing",        pricing_cols),
            ]:
                acc_map = _per_persona_accuracy(llm.copy(), w13.copy(), col_subset)
                for pid, acc in acc_map.items():
                    rows.append({
                        "persona_id": pid,
                        "config":     config_label,
                        "metric":     metric,
                        "version":    v_long,
                        "llm_model":  "gpt-5.4-nano",
                        "accuracy":   acc,
                    })

    return pd.DataFrame(rows)


# ── Core paired bootstrap ─────────────────────────────────────────────────────

def paired_bootstrap_ci(
    df: pd.DataFrame,
    config_A: str,
    config_B: str,
    metric: str,
    version: str,
    llm_model: str,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Paired bootstrap CI on Δ = mean(acc_B) - mean(acc_A) across personas.

    Paired vs. unpaired: by resampling the same persona indices for both A and B,
    we remove between-persona variance (some personas are harder than others).
    This isolates the within-persona effect of changing config, giving narrower
    CIs when persona accuracy is correlated across configs (rho_hat > 0).
    """
    # 1. Filter to slice
    mask = (
        (df["metric"] == metric) &
        (df["version"] == version) &
        (df["llm_model"] == llm_model) &
        (df["config"].isin([config_A, config_B]))
    )
    sl = df[mask].copy()

    for cfg in [config_A, config_B]:
        if cfg not in sl["config"].values:
            raise ValueError(f"config '{cfg}' not found for metric={metric}, version={version}")

    # 2. Pivot to wide format
    wide = sl.pivot_table(index="persona_id", columns="config", values="accuracy")
    if config_A not in wide.columns or config_B not in wide.columns:
        raise ValueError(f"Pivot failed — missing configs in wide table")

    # Check for missing personas
    missing_A = wide[wide[config_A].isna()].index.tolist()
    missing_B = wide[wide[config_B].isna()].index.tolist()
    if missing_A or missing_B:
        raise ValueError(
            f"Missing personas — config_A missing: {missing_A}, config_B missing: {missing_B}"
        )

    acc_A = wide[config_A].values.astype(float)
    acc_B = wide[config_B].values.astype(float)
    n = len(acc_A)
    m = sample_size if sample_size is not None else n  # default: resample all n

    # 3. Pearson correlation (diagnostic)
    rho_hat, _ = pearsonr(acc_A, acc_B)

    # 4 & 5. Paired bootstrap
    rng = np.random.default_rng(seed)
    # Vectorized: sample B×m indices, then mean difference per row
    idx = rng.integers(0, n, size=(B, m))
    delta_b = acc_B[idx].mean(axis=1) - acc_A[idx].mean(axis=1)

    # 6. CI
    ci_lo = float(np.percentile(delta_b, 100 * alpha / 2))
    ci_hi = float(np.percentile(delta_b, 100 * (1 - alpha / 2)))

    return {
        "delta_point":       float(acc_B.mean() - acc_A.mean()),
        "ci_lower":          ci_lo,
        "ci_upper":          ci_hi,
        "se_delta":          float(delta_b.std(ddof=1)),
        "rho_hat":           float(rho_hat),
        "n":                 n,
        "bootstrap_deltas":  delta_b,
    }


# ── Wrapper: run target contrasts ─────────────────────────────────────────────

TARGET_CONTRASTS = [
    ("bg",      "bg+ep",    "overall"),
    ("bg",      "bg+dp",    "overall"),
    ("bg+dp",   "bg+dp+ep", "pricing"),
]


def run_target_contrasts(
    df: pd.DataFrame,
    version: str,
    llm_model: str,
    B: int = 10_000,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    for config_A, config_B, metric in TARGET_CONTRASTS:
        res = paired_bootstrap_ci(df, config_A, config_B, metric, version, llm_model,
                                  B=B, sample_size=sample_size)
        rows.append({
            "contrast":    f"{config_B} vs {config_A}",
            "metric":      metric,
            "delta_point": round(res["delta_point"], 4),
            "ci_lower":    round(res["ci_lower"],    4),
            "ci_upper":    round(res["ci_upper"],    4),
            "se_delta":    round(res["se_delta"],    4),
            "rho_hat":     round(res["rho_hat"],     3),
            "significant": not (res["ci_lower"] <= 0 <= res["ci_upper"]),
        })
    return pd.DataFrame(rows)


# ── Unpaired bootstrap for comparison ─────────────────────────────────────────

def unpaired_bootstrap_ci(
    df: pd.DataFrame,
    config_A: str,
    config_B: str,
    metric: str,
    version: str,
    llm_model: str,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Unpaired bootstrap — resample A and B independently. Wider CIs than paired."""
    mask = (df["metric"] == metric) & (df["version"] == version) & (df["llm_model"] == llm_model)
    acc_A = df[mask & (df["config"] == config_A)]["accuracy"].values.astype(float)
    acc_B = df[mask & (df["config"] == config_B)]["accuracy"].values.astype(float)
    rng = np.random.default_rng(seed)
    n = len(acc_A)
    idx_A = rng.integers(0, n, size=(B, n))
    idx_B = rng.integers(0, n, size=(B, n))
    delta_b = acc_B[idx_B].mean(axis=1) - acc_A[idx_A].mean(axis=1)
    return {
        "delta_point": float(acc_B.mean() - acc_A.mean()),
        "ci_lower":    float(np.percentile(delta_b, 100 * alpha / 2)),
        "ci_upper":    float(np.percentile(delta_b, 100 * (1 - alpha / 2))),
        "se_delta":    float(delta_b.std(ddof=1)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",  default="v3_maximum",
                        help="Version string as in df (e.g. v3_maximum)")
    parser.add_argument("--model",    default="gpt-5.4-nano")
    parser.add_argument("--B",           type=int, default=10_000)
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Bootstrap sample size m (default: n=all personas)")
    args = parser.parse_args()

    print("Loading per-persona accuracy data...")
    df = load_accuracy_df(versions=["v1", "v2", "v3"])
    print(f"  {len(df)} rows  |  personas: {sorted(df['persona_id'].unique())[:5]}...")
    print(f"  configs:  {sorted(df['config'].unique())}")
    print(f"  versions: {sorted(df['version'].unique())}")

    version   = args.version
    llm_model = args.model

    print(f"\n{'='*65}")
    print(f"Paired bootstrap — {version}, {llm_model}, B={args.B:,}")
    print(f"{'='*65}")

    m_label = f"m={args.sample_size}" if args.sample_size else f"m=n={df[df['version']==version]['persona_id'].nunique()}"
    print(f"  sample_size: {m_label}")
    summary = run_target_contrasts(df, version, llm_model, B=args.B, sample_size=args.sample_size)
    print(summary.to_string(index=False))

    # Validation checks
    print(f"\n{'─'*65}")
    print("Validation:")
    for _, row in summary.iterrows():
        # Sanity: rho_hat
        rho = row["rho_hat"]
        rho_flag = "" if 0.2 < rho < 0.99 else "  *** UNEXPECTED rho ***"
        print(f"  {row['contrast']:25s}  rho={rho:.3f}{rho_flag}")

    # Compare paired vs unpaired CI width for first contrast
    config_A, config_B, metric = TARGET_CONTRASTS[0]
    paired   = paired_bootstrap_ci(df, config_A, config_B, metric, version, llm_model, B=args.B, sample_size=args.sample_size)
    unpaired = unpaired_bootstrap_ci(df, config_A, config_B, metric, version, llm_model, B=args.B)
    paired_w   = paired["ci_upper"]   - paired["ci_lower"]
    unpaired_w = unpaired["ci_upper"] - unpaired["ci_lower"]
    print(f"\n  Paired vs unpaired CI width for '{config_B} vs {config_A}' ({metric}):")
    print(f"    Paired:   [{paired['ci_lower']:.4f}, {paired['ci_upper']:.4f}]  width={paired_w:.4f}")
    print(f"    Unpaired: [{unpaired['ci_lower']:.4f}, {unpaired['ci_upper']:.4f}]  width={unpaired_w:.4f}")
    if paired_w < unpaired_w:
        print(f"    Paired is {(1 - paired_w/unpaired_w)*100:.1f}% narrower — as expected.")
    else:
        print(f"    *** WARNING: paired CI is WIDER than unpaired — rho may be low ***")

    # Save
    out = Path("evaluation/paired_bootstrap_results_v3.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
