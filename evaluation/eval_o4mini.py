"""
eval_o4mini.py — Evaluate all 10 o4-mini runs (one per persona setting).

Usage (from Digital-Twin-Simulation/):
    python evaluation/eval_o4mini.py
    python evaluation/eval_o4mini.py --out evaluation/o4mini_results.csv
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
BASE_OUTPUT = Path("text_simulation/text_simulation_output_o4mini")
MODEL = "o4-mini"

SETTINGS = [
    "skill_v1", "skill_v2", "skill_v3",
    "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]


def make_eval_config(trial_dir: str, base_config: dict) -> dict:
    cfg = base_config.copy()
    cfg["trial_dir"] = trial_dir
    cfg["model_name"] = MODEL
    cfg["waves"] = {
        "wave1_3": base_config["waves"]["wave1_3"],
        "wave4":   base_config["waves"]["wave4"],
        "llm_imputed": {
            "input_pattern":        f"{trial_dir}/answer_blocks_llm_imputed/pid_{{pid}}_wave4_Q_wave4_A.json",
            "output_csv":           f"{trial_dir}/csv_comparison/responses_llm_imputed.csv",
            "output_csv_formatted": f"{trial_dir}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv",
            "output_csv_labeled":   f"{trial_dir}/csv_comparison/csv_formatted_label/responses_llm_imputed_label_formatted.csv",
        },
    }
    cfg["evaluation"] = {
        "output_dir":     f"{trial_dir}/accuracy_evaluation",
        "model_name":     MODEL,
        "mad_plot_title": f"o4-mini | {Path(trial_dir).name}",
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
            out["LLM_Accuracy"] = round(float(llm_row[acc_col].values[0]), 4)
            out["LLM_CI_low"]   = round(float(llm_row["Accuracy 95% CI Lower"].values[0]), 4)
            out["LLM_CI_high"]  = round(float(llm_row["Accuracy 95% CI Upper"].values[0]), 4)
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
    parser.add_argument("--out", default="evaluation/o4mini_results.csv")
    args = parser.parse_args()

    with open(BASE_EVAL) as f:
        base_config = yaml.safe_load(f)

    rows = []
    for i, setting in enumerate(SETTINGS, 1):
        trial_dir = str(BASE_OUTPUT / setting)
        print(f"\n[{i}/{len(SETTINGS)}] {setting}  →  {trial_dir}")

        if not Path(trial_dir).exists():
            print(f"    SKIP: directory not found")
            rows.append({"setting": setting, "status": "missing"})
            continue

        imputed_dir = Path(trial_dir) / "answer_blocks_llm_imputed"
        if not imputed_dir.exists() or not list(imputed_dir.glob("*.json")):
            print(f"    SKIP: no imputed answers")
            rows.append({"setting": setting, "status": "no_imputed"})
            continue

        cfg = make_eval_config(trial_dir, base_config)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp_path = f.name

        try:
            ok1 = run_step("evaluation/json2csv.py",               tmp_path, setting)
            ok2 = run_step("evaluation/mad_accuracy_evaluation.py", tmp_path, setting)
        finally:
            os.unlink(tmp_path)

        metrics = extract_metrics(trial_dir)
        row = {"setting": setting, "status": "ok" if (ok1 and ok2) else "partial"}
        row.update(metrics)
        if metrics:
            print(f"    Accuracy: {metrics.get('LLM_Accuracy','N/A')}  "
                  f"(CI [{metrics.get('LLM_CI_low','?')}, {metrics.get('LLM_CI_high','?')}])  "
                  f"| Random: {metrics.get('Random_Baseline','N/A')}  "
                  f"| Human: {metrics.get('Human_Ceiling','N/A')}")
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print("O4-MINI RESULTS")
    print(f"{'='*60}")
    if "LLM_Accuracy" in df.columns:
        display = df[["setting", "LLM_Accuracy", "LLM_CI_low", "LLM_CI_high",
                       "Random_Baseline", "Human_Ceiling"]].copy()
        display = display.set_index("setting").reindex(SETTINGS)
        display = display.sort_values("LLM_Accuracy", ascending=False)
        print(display.to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
