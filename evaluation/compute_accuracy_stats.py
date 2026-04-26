"""
compute_accuracy_stats.py — Compute mean, std, and 95% CI for per-persona accuracy.

For each ablation_fixed config, computes per-persona MAD accuracy then reports:
  - mean  (across all personas)
  - std   (population std across personas)
  - 95% CI (from the bootstrap_ci.py run — reads saved CSV)

Usage (from Digital-Twin-Simulation/):
    python evaluation/compute_accuracy_stats.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

VERSIONS    = ["v1", "v2", "v3"]
CONDITIONS  = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]
BASE        = Path("text_simulation")
PRICING_COLS = {f"{i}_Q295" for i in range(1, 41)}

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

MINMAX = get_minmax()
RANGES = {col: mx - mn for col, (mn, mx) in MINMAX.items()}

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

def load_formatted(csv_dir):
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

def per_persona_accuracy(llm, w13, col_subset=None):
    common = set(llm["TWIN_ID"]) & set(w13["TWIN_ID"])
    llm = llm[llm["TWIN_ID"].isin(common)].set_index("TWIN_ID")
    w13 = w13[w13["TWIN_ID"].isin(common)].set_index("TWIN_ID")
    all_cols = [c for c in llm.columns if c in RANGES]
    if col_subset: all_cols = [c for c in all_cols if c in col_subset]
    result = {}
    for pid in llm.index:
        accs = []
        for col in all_cols:
            if col not in llm.columns or col not in w13.columns: continue
            l = pd.to_numeric(llm.loc[pid, col], errors="coerce")
            w = pd.to_numeric(w13.loc[pid, col], errors="coerce")
            if pd.isna(l) or pd.isna(w): continue
            accs.append(1 - abs(l - w) / RANGES[col])
        result[pid] = np.mean(accs) if accs else np.nan
    return np.array(list(result.values()))

def stats(values):
    v = values[~np.isnan(values)]
    if len(v) == 0: return np.nan, np.nan
    return round(v.mean(), 4), round(v.std(ddof=1), 4)

rows = []
pricing_set = {c.upper() for c in PRICING_COLS}

for version in VERSIONS:
    suite    = f"nano_v2_ablation_fixed_{version}_temp0"
    base_out = BASE / f"text_simulation_output_{suite}"

    for condition in CONDITIONS:
        csv_dir = base_out / condition / "high" / "csv_comparison"
        if not csv_dir.exists():
            print(f"  SKIP {version}/{condition}")
            continue

        llm, w13 = load_formatted(csv_dir)
        if llm is None: continue

        all_cols     = [c for c in llm.columns if c != "TWIN_ID" and c in RANGES]
        pricing_cols = [c for c in all_cols if c in pricing_set]
        bias_cols    = [c for c in all_cols if c not in pricing_set]

        mean_all,  std_all  = stats(per_persona_accuracy(llm.copy(), w13.copy(), all_cols))
        mean_bias, std_bias = stats(per_persona_accuracy(llm.copy(), w13.copy(), bias_cols))
        mean_pric, std_pric = stats(per_persona_accuracy(llm.copy(), w13.copy(), pricing_cols))

        print(f"  {version}/{condition:12s}  "
              f"overall={mean_all:.3f}±{std_all:.3f}  "
              f"bias={mean_bias:.3f}±{std_bias:.3f}  "
              f"pricing={mean_pric:.3f}±{std_pric:.3f}")

        rows.append({"version": version, "condition": condition,
                     "overall_mean": mean_all, "overall_std": std_all,
                     "bias_mean": mean_bias,   "bias_std": std_bias,
                     "pricing_mean": mean_pric, "pricing_std": std_pric})

df = pd.DataFrame(rows)
out = Path("evaluation/ablation_fixed_accuracy_stats.csv")
df.to_csv(out, index=False)
print(f"\nSaved → {out}")

# Summary tables
for metric in ["overall", "bias", "pricing"]:
    print(f"\n{'='*65}")
    print(f"{metric.upper()} — mean ± std (n=20 personas)")
    print(f"{'='*65}")
    for version in VERSIONS:
        sub = df[df["version"] == version]
        print(f"\n  {version}:")
        for _, r in sub.iterrows():
            print(f"    {r['condition']:12s}  {r[f'{metric}_mean']:.3f} ± {r[f'{metric}_std']:.3f}")
