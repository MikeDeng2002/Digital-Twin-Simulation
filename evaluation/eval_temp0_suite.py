"""
eval_temp0_suite.py — Evaluate nano_temp0 or mini_temp0 experiment results.

Usage (from Digital-Twin-Simulation/):
    python evaluation/eval_temp0_suite.py --suite nano_temp0
    python evaluation/eval_temp0_suite.py --suite mini_temp0
"""

import os, sys, yaml, tempfile, subprocess, argparse
import pandas as pd
from pathlib import Path

BASE_EVAL = Path("evaluation/evaluation_basic.yaml")

SETTINGS_DEFAULT = [
    "skill_v1", "skill_v2", "skill_v3",
    "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]
SETTINGS_V2 = ["skill_v2_v1", "skill_v2_v2", "skill_v2_v3"]
SETTINGS_ABLATION = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]
SETTINGS_SKILLS_EP = ["bg", "bg_dp", "bg_tools", "bg_ep", "bg_dp_tools", "bg_dp_ep", "bg_dp_tools_ep"]
SETTINGS = SETTINGS_DEFAULT  # kept for backward compat
REASONING_LEVELS = ["none", "low", "medium", "high"]

SUITES = {
    "nano_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_temp0_20p_results.csv",
        "settings":    SETTINGS_DEFAULT,
    },
    "mini_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_mini_temp0"),
        "model":       "gpt-5.4-mini",
        "out_csv":     "evaluation/mini_temp0_20p_results.csv",
        "settings":    SETTINGS_DEFAULT,
    },
    "nano_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_temp0_20p_results.csv",
        "settings":    SETTINGS_V2,
    },
    "nano_v2_ablation_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_temp0_20p_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
    "nano_v2_ablation_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_v2_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_v2_temp0_20p_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
    "nano_v2_ablation_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_v3_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_v3_temp0_20p_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
    "nano_skills_ep_ablation_v1_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v1_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_skills_ep_ablation_v1_temp0_20p_results.csv",
        "settings":    SETTINGS_SKILLS_EP,
    },
    "nano_skills_ep_ablation_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v2_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_skills_ep_ablation_v2_temp0_20p_results.csv",
        "settings":    SETTINGS_SKILLS_EP,
    },
    "nano_skills_ep_ablation_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v3_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_skills_ep_ablation_v3_temp0_20p_results.csv",
        "settings":    SETTINGS_SKILLS_EP,
    },
    "nano_v2_ablation_fixed_v1_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v1_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_fixed_v1_temp0_20p_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
    "nano_v2_ablation_fixed_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v2_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_fixed_v2_temp0_20p_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
    "nano_v2_ablation_fixed_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v3_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_fixed_v3_temp0_20p_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
    "nano_v2_ablation_50p_v2_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_50p_v2_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_50p_v2_temp0_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
    "nano_v2_ablation_50p_v3_temp0": {
        "base_output": Path("text_simulation/text_simulation_output_nano_v2_ablation_50p_v3_temp0"),
        "model":       "gpt-5.4-nano",
        "out_csv":     "evaluation/nano_v2_ablation_50p_v3_temp0_results.csv",
        "settings":    SETTINGS_ABLATION,
    },
}


def make_eval_config(trial_dir: str, model: str, base_config: dict) -> dict:
    cfg = base_config.copy()
    cfg["trial_dir"]  = trial_dir
    cfg["model_name"] = model
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
        "model_name":     model,
        "mad_plot_title": f"{model} | {Path(trial_dir).parent.name} | {Path(trial_dir).name}",
    }
    return cfg


def run_step(script, config_path, label):
    result = subprocess.run([sys.executable, script, "--config", config_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    WARNING [{label}]: {result.stderr[-300:].strip()}")
        return False
    return True


def extract_metrics(trial_dir):
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
    parser.add_argument("--suite",     required=True, choices=list(SUITES.keys()))
    parser.add_argument("--setting",   default=None)
    parser.add_argument("--reasoning", default=None)
    args = parser.parse_args()

    suite      = SUITES[args.suite]
    base_out   = suite["base_output"]
    model      = suite["model"]
    out_csv    = Path(suite["out_csv"])

    default_settings = suite.get("settings", SETTINGS_DEFAULT)
    settings   = [args.setting]   if args.setting   else default_settings
    reasonings = [args.reasoning] if args.reasoning else REASONING_LEVELS

    with open(BASE_EVAL) as f:
        base_config = yaml.safe_load(f)

    rows  = []
    total = len(settings) * len(reasonings)
    done  = 0

    for setting in settings:
        for reasoning in reasonings:
            done += 1
            trial_dir = str(base_out / setting / reasoning)
            label     = f"{setting}__{reasoning}"
            print(f"[{done}/{total}] {label}")

            if not Path(trial_dir).exists():
                print(f"    SKIP: directory not found")
                rows.append({"setting": setting, "reasoning": reasoning, "status": "missing"})
                continue

            imputed = Path(trial_dir) / "answer_blocks_llm_imputed"
            if not imputed.exists() or not list(imputed.glob("*.json")):
                print(f"    SKIP: no imputed answers")
                rows.append({"setting": setting, "reasoning": reasoning, "status": "no_imputed"})
                continue

            cfg = make_eval_config(trial_dir, model, base_config)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(cfg, f)
                tmp = f.name

            try:
                ok1 = run_step("evaluation/json2csv.py",                tmp, label)
                _   = run_step("evaluation/clip_imputed_to_range.py",   tmp, label)
                ok2 = run_step("evaluation/mad_accuracy_evaluation.py", tmp, label)
            finally:
                os.unlink(tmp)

            metrics = extract_metrics(trial_dir)
            row = {"setting": setting, "reasoning": reasoning,
                   "status": "ok" if (ok1 and ok2) else "partial"}
            row.update(metrics)
            if metrics:
                acc = metrics.get("LLM_Accuracy", "N/A")
                print(f"    Accuracy: {acc}  "
                      f"(CI [{metrics.get('LLM_CI_low','?')}, {metrics.get('LLM_CI_high','?')}])  "
                      f"| Random: {metrics.get('Random_Baseline','N/A')}  "
                      f"| Human: {metrics.get('Human_Ceiling','N/A')}")
            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print(f"{args.suite.upper()} — {model} — 20 personas")
    print(f"{'='*70}")

    if "LLM_Accuracy" in df.columns:
        pivot = df.pivot_table(index="setting", columns="reasoning",
                               values="LLM_Accuracy", aggfunc="first")
        col_order = [c for c in REASONING_LEVELS if c in pivot.columns]
        pivot     = pivot[col_order]
        pivot     = pivot.reindex([s for s in default_settings if s in pivot.index])
        pivot["avg"] = pivot.mean(axis=1).round(3)
        print(pivot.round(3).to_string())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
