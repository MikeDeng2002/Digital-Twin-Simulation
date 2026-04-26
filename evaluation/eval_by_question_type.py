"""
eval_by_question_type.py — Compute accuracy separately for:
  - Cognitive biases questions
  - Product preferences (pricing) questions

Mirrors mad_accuracy_evaluation.py exactly:
  - Compares LLM predictions against wave1-3 (not wave4)
  - Decile-normalizes Q164/Q166/Q168/Q170 using wave1-3 percentiles
  - Uses same (min, max) column ranges

Usage (from Digital-Twin-Simulation/):
    python evaluation/eval_by_question_type.py --suite nano_temp0
    python evaluation/eval_by_question_type.py --suite mini_temp0
    python evaluation/eval_by_question_type.py --suite nano_v2_temp0
    python evaluation/eval_by_question_type.py --suite nano_v2_ablation_temp0
"""

import re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import sem, t as t_dist

SETTINGS_DEFAULT = [
    "skill_v1", "skill_v2", "skill_v3",
    "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]
SETTINGS_V2       = ["skill_v2_v1", "skill_v2_v2", "skill_v2_v3"]
SETTINGS_ABLATION = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]
SETTINGS_SKILLS_EP = ["bg", "bg_dp", "bg_tools", "bg_ep", "bg_dp_tools", "bg_dp_ep", "bg_dp_tools_ep"]
SETTINGS = SETTINGS_DEFAULT  # kept for backward compat
REASONING_LEVELS = ["none", "low", "medium", "high"]

SUITES = {
    "nano_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_temp0"),
        "settings": SETTINGS_DEFAULT,
        "out_csv":  "evaluation/nano_temp0_by_question_type.csv",
    },
    "mini_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_mini_temp0"),
        "settings": SETTINGS_DEFAULT,
        "out_csv":  "evaluation/mini_temp0_by_question_type.csv",
    },
    "nano_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_temp0"),
        "settings": SETTINGS_V2,
        "out_csv":  "evaluation/nano_v2_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_v2_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_v2_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_v3_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_v3_temp0_by_question_type.csv",
    },
    "nano_skills_ep_ablation_v1_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v1_temp0"),
        "settings": SETTINGS_SKILLS_EP,
        "out_csv":  "evaluation/nano_skills_ep_ablation_v1_temp0_by_question_type.csv",
    },
    "nano_skills_ep_ablation_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v2_temp0"),
        "settings": SETTINGS_SKILLS_EP,
        "out_csv":  "evaluation/nano_skills_ep_ablation_v2_temp0_by_question_type.csv",
    },
    "nano_skills_ep_ablation_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v3_temp0"),
        "settings": SETTINGS_SKILLS_EP,
        "out_csv":  "evaluation/nano_skills_ep_ablation_v3_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_fixed_v1_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v1_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_fixed_v1_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_fixed_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v2_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_fixed_v2_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_fixed_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v3_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_fixed_v3_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_50p_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_50p_v2_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_50p_v2_temp0_by_question_type.csv",
    },
    "nano_v2_ablation_50p_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_50p_v3_temp0"),
        "settings": SETTINGS_ABLATION,
        "out_csv":  "evaluation/nano_v2_ablation_50p_v3_temp0_by_question_type.csv",
    },
}

# Columns that are decile-normalized in mad_accuracy_evaluation.py
DECILE_GROUP_1 = ["Q164", "Q166"]   # normalized together
DECILE_GROUP_2 = ["Q168", "Q170"]   # normalized together


def get_minmax():
    """(min, max) per column — matches mad_accuracy_evaluation.get_default_column_ranges()."""
    mm = {}
    # False consensus self (1–5)
    mm.update({f"FALSE CONS. SELF _{i}": (1, 5) for i in list(range(1, 8)) + [10, 11, 12]})
    # False consensus others (0–100)
    mm.update({f"FALSE CONS. OTHERS _{i}": (0, 100) for i in list(range(1, 8)) + [10, 11, 12]})
    # Base rate (0–100)
    mm["Q156_1"] = (0, 100);  mm["FORM A _1"] = (0, 100)
    # Framing (1–6)
    mm["Q157"] = (1, 6);  mm["Q158"] = (1, 6)
    # Linda conjunction (1–6)
    for i in [1, 2, 3]: mm[f"Q159_{i}"] = (1, 6);  mm[f"Q160_{i}"] = (1, 6)
    # Outcome bias (1–7)
    mm["Q161"] = (1, 7);  mm["Q162"] = (1, 7)
    # Anchoring — DECILE-normalized to 1–10 before comparison
    for c in ["164", "166", "168", "170"]: mm[f"Q{c}"] = (1, 10)
    # Less is more (1–5 or 1–6)
    for c in ["171", "172", "173", "174", "175", "176"]: mm[f"Q{c}"] = (1, 5)
    for c in ["177", "178", "179"]: mm[f"Q{c}"] = (1, 6)
    # Sunk cost (0–20)
    mm["Q181"] = (0, 20);  mm["Q182"] = (0, 20)
    # Absolute vs relative (1–2)
    mm["Q183"] = (1, 2);  mm["Q184"] = (1, 2)
    # WTA/WTP-Thaler (1–10)
    mm["Q189"] = (1, 10);  mm["Q190"] = (1, 10);  mm["Q191"] = (1, 10)
    # Allais (1–2)
    mm["Q192"] = (1, 2);  mm["Q193"] = (1, 2)
    # Myside (1–6)
    mm["Q194"] = (1, 6);  mm["Q195"] = (1, 6)
    # Prob matching vs max (1–2)
    mm.update({f"Q198_{i}": (1, 2) for i in range(1, 11)})
    mm.update({f"Q203_{i}": (1, 2) for i in range(1, 7)})
    # Non-separability (1–7)
    mm.update({f"NONSEPARABILTY BENE _{i}": (1, 7) for i in range(1, 5)})
    mm.update({f"NONSEPARABILITY RIS _{i}": (1, 7) for i in range(1, 5)})
    # Omission (1–4), denominator (1–2)
    mm["OMISSION BIAS "] = (1, 4);  mm["DENOMINATOR NEGLECT "] = (1, 2)
    # Pricing (1–2)
    mm.update({f"{i}_Q295": (1, 2) for i in range(1, 41)})
    return mm


MINMAX = get_minmax()
RANGES = {col: mx - mn for col, (mn, mx) in MINMAX.items()}
PRICING_COLS = {f"{i}_Q295" for i in range(1, 41)}


def get_qid_to_task():
    """Map column names → task label, matching mad_accuracy_evaluation.get_default_qid_to_task()."""
    raw = {
        **{f"False Cons. self _{i}": "false consensus" for i in list(range(1, 8)) + [10, 11, 12]},
        **{f"False cons. others _{i}": "false consensus" for i in list(range(1, 8)) + [10, 11, 12]},
        "Q156_1": "base rate", "Form A _1": "base rate",
        "Q157": "framing problem", "Q158": "framing problem",
        **{f"Q160_{i}": "conjunction problem (Linda)" for i in [1, 2, 3]},
        **{f"Q159_{i}": "conjunction problem (Linda)" for i in [1, 2, 3]},
        "Q161": "outcome bias", "Q162": "outcome bias",
        "Q164": "anchoring and adjustment", "Q166": "anchoring and adjustment",
        "Q168": "anchoring and adjustment", "Q170": "anchoring and adjustment",
        **{f"Q17{i}": "less is more" for i in range(1, 10)},
        "Q181": "sunk cost fallacy", "Q182": "sunk cost fallacy",
        "Q183": "absolute vs. relative savings", "Q184": "absolute vs. relative savings",
        "Q189": "WTA/WTP-Thaler", "Q190": "WTA/WTP-Thaler", "Q191": "WTA/WTP-Thaler",
        "Q192": "Allais", "Q193": "Allais",
        "Q194": "myside", "Q195": "myside",
        **{f"Q198_{i}": "prob matching vs. max" for i in range(1, 11)},
        **{f"Q203_{i}": "prob matching vs. max" for i in range(1, 7)},
        **{f"nonseparabilty bene _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        **{f"nonseparability ris _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        "Omission bias ": "omission",
        "Denominator neglect ": "denominator neglect",
        **{f"{i}_Q295": "pricing" for i in range(1, 41)},
    }
    return {k.upper(): v for k, v in raw.items()}


QID_TO_TASK = get_qid_to_task()


def assign_decile(value, thresholds):
    if pd.isna(value):
        return np.nan
    for i, t in enumerate(thresholds):
        if value <= t:
            return i + 1
    return 10


def apply_decile_normalization(df_w13, df_llm):
    """Apply the same decile normalization as mad_accuracy_evaluation.py."""
    for group, cols in [("G1", DECILE_GROUP_1), ("G2", DECILE_GROUP_2)]:
        existing = [c for c in cols if c in df_w13.columns]
        if not existing:
            continue
        combined = pd.concat([df_w13[c] for c in existing]).dropna()
        if len(combined) == 0:
            continue
        thresholds = np.percentile(combined, np.arange(10, 100, 10))
        for col in existing:
            if col in df_w13.columns:
                df_w13[col] = df_w13[col].apply(lambda x: assign_decile(x, thresholds))
            if col in df_llm.columns:
                df_llm[col] = df_llm[col].apply(lambda x: assign_decile(x, thresholds))


def compute_group_accuracy(df_llm, df_w13, group_cols):
    """Task-level mean-of-means accuracy (LLM vs wave1-3).

    For each respondent: compute mean accuracy per task, then mean over tasks present
    in group_cols.  This matches the aggregation used by mad_accuracy_evaluation.py so
    that overall, cognitive-bias, and pricing numbers are always internally consistent:
    if a config improves every task individually it will also improve every aggregate.
    """
    common_ids = set(df_llm["TWIN_ID"]) & set(df_w13["TWIN_ID"])
    llm = df_llm[df_llm["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")
    w13 = df_w13[df_w13["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")

    respondent_accs = []
    for rid in llm.index:
        if rid not in w13.index:
            continue
        task_accs: dict[str, list] = {}
        for col in group_cols:
            if col not in llm.columns or col not in w13.columns or col not in RANGES:
                continue
            task = QID_TO_TASK.get(col)
            if task is None:
                continue
            lv, wv = llm.at[rid, col], w13.at[rid, col]
            if pd.isna(lv) or pd.isna(wv):
                continue
            task_accs.setdefault(task, []).append(1 - abs(lv - wv) / RANGES[col])

        task_means = [np.mean(v) for v in task_accs.values() if v]
        if task_means:
            respondent_accs.append(np.mean(task_means))

    if not respondent_accs:
        return np.nan, np.nan, np.nan
    arr = np.array(respondent_accs)
    mean = arr.mean()
    if len(arr) > 1:
        se = sem(arr)
        ci_lo, ci_hi = t_dist.interval(0.95, len(arr) - 1, loc=mean, scale=se)
    else:
        ci_lo = ci_hi = np.nan
    return round(mean, 4), round(ci_lo, 4), round(ci_hi, 4)


def load_formatted(csv_dir):
    """Load formatted CSVs; return (df_llm, df_wave1_3) matching mad_accuracy_evaluation.py."""
    fmt = csv_dir / "csv_formatted"
    try:
        llm = pd.read_csv(fmt / "responses_llm_imputed_formatted.csv", skiprows=[1])
        w13 = pd.read_csv(fmt / "responses_wave1_3_formatted.csv",      skiprows=[1])
    except FileNotFoundError:
        return None, None
    for df in [llm, w13]:
        df.columns = df.columns.str.upper()
        for col in df.columns:
            if col != "TWIN_ID":
                df[col] = pd.to_numeric(df[col], errors="coerce")
    # Apply decile normalization (modifies in place)
    apply_decile_normalization(w13, llm)
    return llm, w13


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",     required=True, choices=list(SUITES.keys()))
    parser.add_argument("--setting",   default=None)
    parser.add_argument("--reasoning", default=None)
    args = parser.parse_args()

    suite_cfg      = SUITES[args.suite]
    base_dir       = suite_cfg["base_output"]
    default_settings = suite_cfg["settings"]
    out_csv        = Path(suite_cfg["out_csv"])
    settings   = [args.setting]   if args.setting   else default_settings
    reasonings = [args.reasoning] if args.reasoning else REASONING_LEVELS

    rows = []
    total = len(settings) * len(reasonings)
    done  = 0

    for setting in settings:
        for reasoning in reasonings:
            done += 1
            trial_dir = base_dir / setting / reasoning
            csv_dir   = trial_dir / "csv_comparison"
            label     = f"{setting}__{reasoning}"

            if not csv_dir.exists():
                print(f"[{done}/{total}] {label} — SKIP (no csv_comparison)")
                continue

            llm, w13 = load_formatted(csv_dir)
            if llm is None:
                print(f"[{done}/{total}] {label} — SKIP (formatted CSVs missing)")
                continue

            # Identify column groups (intersect with RANGES so only known cols are used)
            all_cols     = [c for c in llm.columns if c not in ("TWIN_ID", "WAVE") and c in RANGES]
            pricing_cols = [c for c in all_cols if c in PRICING_COLS]
            bias_cols    = [c for c in all_cols if c not in PRICING_COLS]

            acc_all,     ci_lo_all,  ci_hi_all  = compute_group_accuracy(llm, w13, all_cols)
            acc_bias,    ci_lo_bias, ci_hi_bias  = compute_group_accuracy(llm, w13, bias_cols)
            acc_pricing, ci_lo_p,    ci_hi_p     = compute_group_accuracy(llm, w13, pricing_cols)

            print(f"[{done}/{total}] {label:35s}  "
                  f"overall={acc_all:.3f}  "
                  f"bias={acc_bias:.3f}  "
                  f"pricing={acc_pricing:.3f}")

            rows.append({
                "setting": setting, "reasoning": reasoning,
                "overall_acc":  acc_all,
                "bias_acc":     acc_bias,    "bias_ci_lo":    ci_lo_bias, "bias_ci_hi":    ci_hi_bias,
                "pricing_acc":  acc_pricing, "pricing_ci_lo": ci_lo_p,    "pricing_ci_hi": ci_hi_p,
            })

    df = pd.DataFrame(rows)

    for metric, label in [("overall_acc", "Overall"), ("bias_acc", "Cognitive Biases"), ("pricing_acc", "Product Preferences")]:
        print(f"\n{'='*70}")
        print(f"{label} — {args.suite.upper()}")
        print(f"{'='*70}")
        if metric in df.columns:
            piv = df.pivot_table(index="setting", columns="reasoning", values=metric, aggfunc="first")
            piv = piv[[c for c in REASONING_LEVELS if c in piv.columns]]
            piv = piv.reindex([s for s in default_settings if s in piv.index])
            piv["avg"] = piv.mean(axis=1).round(3)
            print(piv.round(3).to_string())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
