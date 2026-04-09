"""
04_analyze.py — OLS regression to estimate per-celebrity influence weights.

For each persona, regresses:
    prediction = intercept + w_baseline * baseline
                + w_saylor * x_saylor + w_wood * x_wood + ... + w_roubini * x_roubini

where x_celebrity = celebrity's price when they appear in the trial, else 0.
This gives a separate influence weight for each celebrity.

Usage (from Digital-Twin-Simulation/):
    poetry run python Bitcoin_experiment/scripts/04_analyze.py
    poetry run python Bitcoin_experiment/scripts/04_analyze.py --type A
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

REPRESENTATION = "v3_maximum"
RESULTS_DIR = Path("Bitcoin_experiment/results") / REPRESENTATION
ANALYSIS_DIR = Path("Bitcoin_experiment/analysis")
CELEBRITIES_DIR = Path("Bitcoin_experiment/data/celebrities")

BULLS = ["michael_saylor", "cathie_wood", "elon_musk", "jack_dorsey", "larry_fink", "donald_trump"]
BEARS = ["warren_buffett", "peter_schiff", "jamie_dimon", "nouriel_roubini"]
ALL_SLUGS = BULLS + BEARS


def load_celebrity_names() -> dict:
    """Map slug → display name."""
    names = {}
    for slug in ALL_SLUGS:
        profile = (CELEBRITIES_DIR / slug / "profile.txt").read_text(encoding="utf-8")
        name = profile.split("\n")[0].replace("Name:", "").strip()
        names[slug] = name
    return names


def load_trials(pid: str, exp_type: str) -> list[dict]:
    folder = "type_ab" if exp_type == "both" else f"type_{exp_type.lower()}"
    path = RESULTS_DIR / folder / f"pid_{pid}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def regression_per_persona(
    pid: str, trials: list[dict], exp_type: str, celeb_names: dict
) -> dict | None:
    """
    OLS: (prediction - baseline) = sum_i w_i * (price_i if celeb_i present else 0)

    Baseline is constant per persona (one fixed API call), so including it alongside
    an intercept would create perfect collinearity. Instead we subtract it from the
    outcome directly, making w_i interpretable as:
        "dollars shifted from prior per $1 of celebrity i's price, when present"

    No intercept term — the model is forced through the origin in shift-space,
    meaning zero celebrity exposure = zero shift from baseline (by construction).
    """
    pred_key = f"type_{exp_type.lower()}_prediction"

    rows = []
    for t in trials:
        pred = t.get(pred_key)
        baseline = t.get("baseline_price")
        if pred is None or baseline is None:
            continue
        present_names = {n["name"]: n["price"] for n in t.get("neighbors", [])}
        row = {"shift": pred - baseline, "baseline": baseline}
        for slug in ALL_SLUGS:
            name = celeb_names[slug]
            row[slug] = present_names.get(name, 0)
        rows.append(row)

    if len(rows) < max(5, len(ALL_SLUGS)):
        return None

    df = pd.DataFrame(rows)

    # Drop celebrity columns with zero variance (never appeared)
    celeb_cols = [s for s in ALL_SLUGS if df[s].std() > 0]
    if not celeb_cols:
        return None

    # No intercept — regress shift directly on celebrity presence
    X = np.column_stack([df[s].values for s in celeb_cols])
    y = df["shift"].values

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    celeb_weights = dict(zip(celeb_cols, coeffs))

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    baseline_val = float(df["baseline"].iloc[0])
    result = {
        "pid": pid,
        "n_trials": len(df),
        "baseline": round(baseline_val, 0),
        "mean_shift": round(float(df["shift"].mean()), 0),
        "std_shift": round(float(df["shift"].std()), 0),
        "r_squared": round(float(r_squared), 4),
        "mean_prediction": round(float((df["shift"] + df["baseline"]).mean()), 0),
    }
    for slug in ALL_SLUGS:
        w = celeb_weights.get(slug)
        result[f"w_{slug}"] = round(float(w), 4) if w is not None else None

    return result


def compare_type_a_vs_b(pid: str, trials: list[dict]) -> dict | None:
    """Compare Type A vs Type B predictions for the same trials."""
    rows = []
    for t in trials:
        pred_a = t.get("type_a_prediction")
        pred_b = t.get("type_b_prediction")
        if pred_a is None or pred_b is None:
            continue
        rows.append({"trial": t["trial"], "type_a": pred_a, "type_b": pred_b,
                     "diff": pred_b - pred_a})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        "pid": pid,
        "n_matched": len(df),
        "mean_diff_b_minus_a": round(float(df["diff"].mean()), 0),
        "std_diff": round(float(df["diff"].std()), 0),
        "corr_a_b": round(float(df["type_a"].corr(df["type_b"])), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    celeb_names = load_celebrity_names()

    folder = "type_ab" if args.type == "both" else f"type_{args.type.lower()}"
    trial_dir = RESULTS_DIR / folder
    if not trial_dir.exists():
        print(f"No results found in {trial_dir}. Run 03_run_trials.py first.")
        return

    pids = sorted(
        [f.stem.replace("pid_", "") for f in trial_dir.glob("pid_*.json")],
        key=lambda x: int(x)
    )
    print(f"Found {len(pids)} personas with results.\n")

    # --- Per-celebrity OLS regression ---
    reg_results = []
    for pid in pids:
        trials = load_trials(pid, args.type)
        result = regression_per_persona(pid, trials, "A", celeb_names)
        if result:
            reg_results.append(result)

    if reg_results:
        df_reg = pd.DataFrame(reg_results)

        # Save full table
        out_path = ANALYSIS_DIR / "w_celeb_per_celebrity_type_a.csv"
        df_reg.to_csv(out_path, index=False)
        print(f"Saved full per-celebrity weights to {out_path}\n")

        # --- Per-celebrity mean influence across personas ---
        w_cols = [f"w_{slug}" for slug in ALL_SLUGS]
        celeb_summary = []
        for slug in ALL_SLUGS:
            col = f"w_{slug}"
            vals = df_reg[col].dropna()
            cat = "BULL" if slug in BULLS else "BEAR"
            celeb_summary.append({
                "slug": slug,
                "name": celeb_names[slug],
                "category": cat,
                "mean_w": round(float(vals.mean()), 4),
                "median_w": round(float(vals.median()), 4),
                "std_w": round(float(vals.std()), 4),
                "pct_positive": round(float((vals > 0).mean()) * 100, 1),
                "n_personas": len(vals),
            })
        df_celeb = pd.DataFrame(celeb_summary).sort_values("mean_w", ascending=False)
        celeb_out = ANALYSIS_DIR / "celebrity_influence_summary.csv"
        df_celeb.to_csv(celeb_out, index=False)

        print("=" * 75)
        print("PER-CELEBRITY INFLUENCE WEIGHTS (mean across personas)")
        print("=" * 75)
        print(f"{'Celebrity':<25} {'Cat':<5} {'Mean w':>8} {'Median w':>10} {'Std':>8} {'% pos':>7} {'n':>5}")
        print("-" * 75)
        for _, row in df_celeb.iterrows():
            print(f"{row['name']:<25} {row['category']:<5} {row['mean_w']:>8.4f} "
                  f"{row['median_w']:>10.4f} {row['std_w']:>8.4f} "
                  f"{row['pct_positive']:>6.1f}% {row['n_personas']:>5}")
        print(f"\nSaved to {celeb_out}")

        # --- Per-persona summary (top/bottom) ---
        # Compute mean absolute influence per persona
        df_reg["mean_abs_w_celeb"] = df_reg[w_cols].abs().mean(axis=1, skipna=True)
        df_reg_sorted = df_reg.sort_values("mean_abs_w_celeb", ascending=False)

        print(f"\n{'='*75}")
        print("TOP 10 MOST CELEBRITY-INFLUENCED PERSONAS")
        print(f"{'='*75}")
        print(f"{'PID':<6} {'baseline':>10} {'mean_shift':>11} {'R²':>6} {'mean|w_celeb|':>14} ", end="")
        for slug in ALL_SLUGS:
            short = slug.split("_")[-1][:6]
            print(f"{short:>8}", end="")
        print()
        print("-" * 75)
        for _, row in df_reg_sorted.head(10).iterrows():
            print(f"pid_{row['pid']:<3} {row['baseline']:>10,.0f} {row['mean_shift']:>11,.0f} "
                  f"{row['r_squared']:>6.3f} {row['mean_abs_w_celeb']:>14.4f} ", end="")
            for slug in ALL_SLUGS:
                val = row.get(f"w_{slug}")
                if val is None:
                    print(f"{'—':>8}", end="")
                else:
                    print(f"{val:>8.3f}", end="")
            print()

    # --- Type A vs B comparison ---
    if args.type == "both":
        ab_results = []
        for pid in pids:
            trials = load_trials(pid, "both")
            result = compare_type_a_vs_b(pid, trials)
            if result:
                ab_results.append(result)

        if ab_results:
            df_ab = pd.DataFrame(ab_results).sort_values("mean_diff_b_minus_a", key=abs, ascending=False)
            out_path_ab = ANALYSIS_DIR / "type_a_vs_b_comparison.csv"
            df_ab.to_csv(out_path_ab, index=False)

            print(f"\n{'='*60}")
            print("TYPE A vs B COMPARISON")
            print(f"{'='*60}")
            print(f"Mean diff (B - A) across all personas: "
                  f"${df_ab['mean_diff_b_minus_a'].mean():,.0f}")
            print(f"Positive = celebrity reasoning pushes price UP in Type B")
            print(f"Negative = celebrity reasoning pulls price DOWN in Type B")
            print(f"\nSaved to {out_path_ab}")


if __name__ == "__main__":
    main()
