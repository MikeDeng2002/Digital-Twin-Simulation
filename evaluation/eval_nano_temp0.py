"""
eval_nano_temp0.py — Evaluate all 40 nano temp=0.0 runs (10 settings × 4 reasoning levels).

Usage (from Digital-Twin-Simulation/):
    python evaluation/eval_nano_temp0.py
    python evaluation/eval_nano_temp0.py --out evaluation/nano_temp0_results.csv
"""

import os
import sys
import yaml
import tempfile
import subprocess
import argparse
import pandas as pd
from pathlib import Path

BASE_EVAL = Path("evaluation/evaluation_basic.yaml")
BASE_OUTPUT = Path("text_simulation/text_simulation_output_nano_temp0")
MODEL = "gpt-5.4-nano"

SETTINGS = [
    "skill_v1", "skill_v2", "skill_v3",
    "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]
REASONING_LEVELS = ["none", "low", "medium", "high"]


def make_eval_config(trial_dir: str, base_config: dict) -> dict:
    cfg = base_config.copy()
    cfg["trial_dir"] = trial_dir
    cfg["model_name"] = MODEL
    cfg["waves"] = {
        "wave1_3": base_config["waves"]["wave1_3"],
        "wave4":   base_config["waves"]["wave4"],
        "llm_imputed": {
            "input_pattern":           f"{trial_dir}/answer_blocks_llm_imputed/pid_{{pid}}_wave4_Q_wave4_A.json",
            "output_csv":              f"{trial_dir}/csv_comparison/responses_llm_imputed.csv",
            "output_csv_formatted":    f"{trial_dir}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv",
            "output_csv_labeled":      f"{trial_dir}/csv_comparison/csv_formatted_label/responses_llm_imputed_label_formatted.csv",
        },
    }
    cfg["evaluation"] = {
        "output_dir":     f"{trial_dir}/accuracy_evaluation",
        "model_name":     MODEL,
        "mad_plot_title": f"nano-temp0 | {Path(trial_dir).parent.name} | {Path(trial_dir).name}",
    }
    return cfg


def run_step(script: str, config_path: str, label: str) -> bool:
    result = subprocess.run(
        [sys.executable, script, "--config", config_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    WARNING [{label}]: {result.stderr[-400:].strip()}")
        return False
    return True


def extract_metrics(trial_dir: str) -> dict:
    xlsx = Path(trial_dir) / "accuracy_evaluation" / "mad_accuracy_summary.xlsx"
    if not xlsx.exists():
        return {}
    try:
        df = pd.read_excel(xlsx)
        llm_row    = df[df.iloc[:, 0].astype(str).str.contains("llm",    case=False, na=False)]
        random_row = df[df.iloc[:, 0].astype(str).str.contains("random", case=False, na=False)]
        human_row  = df[df.iloc[:, 0].astype(str).str.contains("wave4",  case=False, na=False)]
        acc_col = "Mean Accuracy"
        out = {}
        if not llm_row.empty and acc_col in df.columns:
            out["LLM_Accuracy"]  = round(float(llm_row[acc_col].values[0]), 4)
            out["LLM_CI_low"]    = round(float(llm_row["Accuracy 95% CI Lower"].values[0]), 4)
            out["LLM_CI_high"]   = round(float(llm_row["Accuracy 95% CI Upper"].values[0]), 4)
        if not random_row.empty and acc_col in df.columns:
            out["Random_Baseline"] = round(float(random_row[acc_col].values[0]), 4)
        if not human_row.empty and acc_col in df.columns:
            out["Human_Ceiling"] = round(float(human_row[acc_col].values[0]), 4)
        return out
    except Exception as e:
        print(f"    Could not parse {xlsx}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evaluation/nano_temp0_results.csv")
    parser.add_argument("--setting",   default=None, help="Filter to one setting")
    parser.add_argument("--reasoning", default=None, help="Filter to one reasoning level")
    args = parser.parse_args()

    with open(BASE_EVAL) as f:
        base_config = yaml.safe_load(f)

    settings   = [args.setting]   if args.setting   else SETTINGS
    reasonings = [args.reasoning] if args.reasoning else REASONING_LEVELS

    rows = []
    total = len(settings) * len(reasonings)
    done  = 0

    for setting in settings:
        for reasoning in reasonings:
            done += 1
            trial_dir = str(BASE_OUTPUT / setting / reasoning)
            label = f"{setting}__{reasoning}"
            print(f"\n[{done}/{total}] {label}  →  {trial_dir}")

            if not Path(trial_dir).exists():
                print(f"    SKIP: directory not found")
                rows.append({"setting": setting, "reasoning": reasoning, "status": "missing"})
                continue

            # Check if answer_blocks_llm_imputed has data
            imputed_dir = Path(trial_dir) / "answer_blocks_llm_imputed"
            if not imputed_dir.exists() or not list(imputed_dir.glob("*.json")):
                print(f"    SKIP: no imputed answers in {imputed_dir}")
                rows.append({"setting": setting, "reasoning": reasoning, "status": "no_imputed"})
                continue

            cfg = make_eval_config(trial_dir, base_config)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(cfg, f)
                tmp_path = f.name

            try:
                ok1 = run_step("evaluation/json2csv.py",              tmp_path, label)
                ok2 = run_step("evaluation/mad_accuracy_evaluation.py", tmp_path, label)
            finally:
                os.unlink(tmp_path)

            metrics = extract_metrics(trial_dir)
            row = {"setting": setting, "reasoning": reasoning,
                   "status": "ok" if (ok1 and ok2) else "partial"}
            row.update(metrics)
            if metrics:
                print(f"    Accuracy: {metrics.get('LLM_Accuracy', 'N/A')}  "
                      f"(CI [{metrics.get('LLM_CI_low','?')}, {metrics.get('LLM_CI_high','?')}])  "
                      f"| Random: {metrics.get('Random_Baseline','N/A')}  "
                      f"| Human: {metrics.get('Human_Ceiling','N/A')}")
            rows.append(row)

    # Summary table
    df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print("NANO TEMP=0.0 RESULTS")
    print(f"{'='*70}")

    if "LLM_Accuracy" in df.columns:
        pivot = df.pivot_table(index="setting", columns="reasoning",
                               values="LLM_Accuracy", aggfunc="first")
        # Reorder columns
        col_order = [c for c in ["none", "low", "medium", "high"] if c in pivot.columns]
        pivot = pivot[col_order]
        # Reorder rows
        row_order = [s for s in SETTINGS if s in pivot.index]
        pivot = pivot.reindex(row_order)
        print("\nAccuracy by setting × reasoning (higher = better):")
        print(pivot.to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
