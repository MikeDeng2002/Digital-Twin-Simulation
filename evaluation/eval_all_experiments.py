"""
eval_all_experiments.py — Run MAD accuracy evaluation across all 8 experiments
and produce a single comparison table.

Usage (from Digital-Twin-Simulation/):
    poetry run python evaluation/eval_all_experiments.py
"""

import os
import json
import re
import subprocess
import sys
import yaml
import tempfile
import shutil
import pandas as pd
from pathlib import Path

BASE_EVAL = Path("evaluation/evaluation_basic.yaml")

EXPERIMENTS = [
    {"name": "demographic_4mini", "model": "gpt-4.1-mini", "persona": "demog",  "trial_dir": "text_simulation/text_simulation_output_demographic_4mini"},
    {"name": "original_4mini",   "model": "gpt-4.1-mini", "persona": "full",   "trial_dir": "text_simulation/text_simulation_output_4mini"},
    {"name": "skill_v1_4mini",   "model": "gpt-4.1-mini", "persona": "v1",     "trial_dir": "text_simulation/text_simulation_output_skill_v1"},
    {"name": "skill_v2_4mini",   "model": "gpt-4.1-mini", "persona": "v2",     "trial_dir": "text_simulation/text_simulation_output_skill_v2"},
    {"name": "skill_v3_4mini",   "model": "gpt-4.1-mini", "persona": "v3",     "trial_dir": "text_simulation/text_simulation_output_skill_v3"},
    {"name": "skill_v4_4mini",   "model": "gpt-4.1-mini", "persona": "v4",     "trial_dir": "text_simulation/text_simulation_output_skill_v4"},
    {"name": "demographic_o4mini","model": "o4-mini",      "persona": "demog",  "trial_dir": "text_simulation/text_simulation_output_demographic_o4mini"},
    {"name": "original_o4mini",  "model": "o4-mini",      "persona": "full",   "trial_dir": "text_simulation/text_simulation_output_o4mini"},
    {"name": "skill_v1_o4mini",  "model": "o4-mini",      "persona": "v1",     "trial_dir": "text_simulation/text_simulation_output_skill_v1_o4mini"},
    {"name": "skill_v2_o4mini",  "model": "o4-mini",      "persona": "v2",     "trial_dir": "text_simulation/text_simulation_output_skill_v2_o4mini"},
    {"name": "skill_v3_o4mini",  "model": "o4-mini",      "persona": "v3",     "trial_dir": "text_simulation/text_simulation_output_skill_v3_o4mini"},
    {"name": "skill_v4_o4mini",  "model": "o4-mini",      "persona": "v4",     "trial_dir": "text_simulation/text_simulation_output_skill_v4_o4mini"},
    {"name": "demographic_gpt4o","model": "gpt-4o",       "persona": "demog",  "trial_dir": "text_simulation/text_simulation_output_demographic_gpt4o"},
    {"name": "original_gpt4o",   "model": "gpt-4o",       "persona": "full",   "trial_dir": "text_simulation/text_simulation_output_gpt4o"},
    {"name": "skill_v1_gpt4o",   "model": "gpt-4o",       "persona": "v1",     "trial_dir": "text_simulation/text_simulation_output_skill_v1_gpt4o"},
    {"name": "skill_v2_gpt4o",   "model": "gpt-4o",       "persona": "v2",     "trial_dir": "text_simulation/text_simulation_output_skill_v2_gpt4o"},
    {"name": "skill_v3_gpt4o",   "model": "gpt-4o",       "persona": "v3",     "trial_dir": "text_simulation/text_simulation_output_skill_v3_gpt4o"},
    {"name": "skill_v4_gpt4o",   "model": "gpt-4o",       "persona": "v4",     "trial_dir": "text_simulation/text_simulation_output_skill_v4_gpt4o"},
]


def make_eval_config(exp: dict, base_config: dict) -> dict:
    cfg = base_config.copy()
    cfg["trial_dir"] = exp["trial_dir"]
    cfg["model_name"] = exp["model"]
    cfg["waves"] = {
        "wave1_3": base_config["waves"]["wave1_3"],
        "wave4": base_config["waves"]["wave4"],
        "llm_imputed": {
            "input_pattern": f"{exp['trial_dir']}/answer_blocks_llm_imputed/pid_{{pid}}_wave4_Q_wave4_A.json",
            "output_csv": f"{exp['trial_dir']}/csv_comparison/responses_llm_imputed.csv",
            "output_csv_formatted": f"{exp['trial_dir']}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv",
            "output_csv_labeled": f"{exp['trial_dir']}/csv_comparison/csv_formatted_label/responses_llm_imputed_label_formatted.csv",
        }
    }
    cfg["evaluation"] = {
        "output_dir": f"{exp['trial_dir']}/accuracy_evaluation",
        "model_name": exp["model"],
        "mad_plot_title": f"Digital Twin - {exp['model']} - {exp['persona']}",
    }
    return cfg


def run_step(script: str, config_path: str, label: str) -> bool:
    result = subprocess.run(
        [sys.executable, script, "--config", config_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING [{label}]: {result.stderr[-300:].strip()}")
        return False
    return True


def extract_mad_summary(trial_dir: str) -> dict | None:
    xlsx_path = Path(trial_dir) / "accuracy_evaluation" / "mad_accuracy_summary.xlsx"
    if not xlsx_path.exists():
        return None
    try:
        df = pd.read_excel(xlsx_path)
        # Target the "llm vs. wave1_3" row — that's the LLM prediction accuracy
        llm_row = df[df.iloc[:, 0].astype(str).str.contains("llm", case=False, na=False)]
        random_row = df[df.iloc[:, 0].astype(str).str.contains("random", case=False, na=False)]
        human_row = df[df.iloc[:, 0].astype(str).str.contains("wave4", case=False, na=False)]
        acc_col = "Mean Accuracy"
        result = {}
        if not llm_row.empty and acc_col in df.columns:
            result["LLM_Accuracy"] = round(float(llm_row[acc_col].values[0]), 4)
            result["LLM_CI_low"] = round(float(llm_row["Accuracy 95% CI Lower"].values[0]), 4)
            result["LLM_CI_high"] = round(float(llm_row["Accuracy 95% CI Upper"].values[0]), 4)
        if not random_row.empty and acc_col in df.columns:
            result["Random_Baseline"] = round(float(random_row[acc_col].values[0]), 4)
        if not human_row.empty and acc_col in df.columns:
            result["Human_Ceiling"] = round(float(human_row[acc_col].values[0]), 4)
        n_col = "TWIN_IDs"
        if n_col in df.columns and not llm_row.empty:
            result["N_personas"] = int(llm_row[n_col].values[0])
        return result if result else None
    except Exception as e:
        print(f"  Could not parse {xlsx_path}: {e}")
        return None


def main():
    with open(BASE_EVAL) as f:
        base_config = yaml.safe_load(f)

    results = []

    for exp in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"Evaluating: {exp['name']}  ({exp['model']} / persona={exp['persona']})")

        # Write temporary config
        tmp_cfg = make_eval_config(exp, base_config)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(tmp_cfg, f)
            tmp_path = f.name

        try:
            # Step 1: json2csv
            print("  Step 1: json2csv...")
            run_step("evaluation/json2csv.py", tmp_path, exp["name"])

            # Step 2: MAD accuracy
            print("  Step 2: MAD accuracy...")
            run_step("evaluation/mad_accuracy_evaluation.py", tmp_path, exp["name"])

        finally:
            os.unlink(tmp_path)

        # Extract summary metrics
        metrics = extract_mad_summary(exp["trial_dir"])
        row = {"experiment": exp["name"], "model": exp["model"], "persona": exp["persona"]}
        if metrics:
            row.update(metrics)
            print(f"  Result: {metrics}")
        else:
            print(f"  Could not extract metrics from {exp['trial_dir']}/accuracy_evaluation/")
        results.append(row)

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Save
    out_path = Path("evaluation/experiment_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
