"""
run_postprocess.py — Run postprocess_simulation_outputs_with_pid for all
personas in a batch-completed output directory tree.

Usage (from Digital-Twin-Simulation/):
    python text_simulation/run_postprocess.py --suite nano_temp0
    python text_simulation/run_postprocess.py --suite mini_temp0
"""

import os, re, argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path("text_simulation")))
from postprocess_responses import postprocess_simulation_outputs_with_pid

SUITES = {
    "nano_temp0": Path("text_simulation/text_simulation_output_nano_temp0"),
    "mini_temp0": Path("text_simulation/text_simulation_output_mini_temp0"),
    "nano_v2_temp0": Path("text_simulation/text_simulation_output_nano_v2_temp0"),
    "nano_v2_ablation_temp0": Path("text_simulation/text_simulation_output_nano_v2_ablation_temp0"),
    "nano_v2_ablation_v2_temp0": Path("text_simulation/text_simulation_output_nano_v2_ablation_v2_temp0"),
    "nano_v2_ablation_v3_temp0": Path("text_simulation/text_simulation_output_nano_v2_ablation_v3_temp0"),
    "nano_skills_ep_ablation_v1_temp0": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v1_temp0"),
    "nano_skills_ep_ablation_v2_temp0": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v2_temp0"),
    "nano_skills_ep_ablation_v3_temp0": Path("text_simulation/text_simulation_output_nano_skills_ep_ablation_v3_temp0"),
    "nano_v2_ablation_fixed_v1_temp0": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v1_temp0"),
    "nano_v2_ablation_fixed_v2_temp0": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v2_temp0"),
    "nano_v2_ablation_fixed_v3_temp0": Path("text_simulation/text_simulation_output_nano_v2_ablation_fixed_v3_temp0"),
    "nano_v2_ablation_50p_v2_temp0":   Path("text_simulation/text_simulation_output_nano_v2_ablation_50p_v2_temp0"),
    "nano_v2_ablation_50p_v3_temp0":   Path("text_simulation/text_simulation_output_nano_v2_ablation_50p_v3_temp0"),
}

QUESTION_JSON_DIR = "./data/mega_persona_json/answer_blocks"

SETTINGS_DEFAULT = [
    "skill_v1", "skill_v2", "skill_v3", "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]

SETTINGS_V2 = ["skill_v2_v1", "skill_v2_v2", "skill_v2_v3"]

SETTINGS_ABLATION = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]

SETTINGS_SKILLS_EP = ["bg", "bg_dp", "bg_tools", "bg_ep", "bg_dp_tools", "bg_dp_ep", "bg_dp_tools_ep"]

SUITE_SETTINGS = {
    "nano_v2_temp0":                    SETTINGS_V2,
    "nano_v2_ablation_temp0":           SETTINGS_ABLATION,
    "nano_v2_ablation_v2_temp0":        SETTINGS_ABLATION,
    "nano_v2_ablation_v3_temp0":        SETTINGS_ABLATION,
    "nano_skills_ep_ablation_v1_temp0": SETTINGS_SKILLS_EP,
    "nano_skills_ep_ablation_v2_temp0": SETTINGS_SKILLS_EP,
    "nano_skills_ep_ablation_v3_temp0": SETTINGS_SKILLS_EP,
    "nano_v2_ablation_fixed_v1_temp0":  SETTINGS_ABLATION,
    "nano_v2_ablation_fixed_v2_temp0":  SETTINGS_ABLATION,
    "nano_v2_ablation_fixed_v3_temp0":  SETTINGS_ABLATION,
    "nano_v2_ablation_50p_v2_temp0":    SETTINGS_ABLATION,
    "nano_v2_ablation_50p_v3_temp0":    SETTINGS_ABLATION,
}

SETTINGS = SETTINGS_DEFAULT  # kept for backward compat reference
REASONING_LEVELS = ["none", "low", "medium", "high"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=list(SUITES.keys()), metavar="SUITE")
    parser.add_argument("--setting",   default=None)
    parser.add_argument("--reasoning", default=None)
    args = parser.parse_args()

    base_dir = SUITES[args.suite]
    default_settings = SUITE_SETTINGS.get(args.suite, SETTINGS_DEFAULT)
    settings   = [args.setting]   if args.setting   else default_settings
    reasonings = [args.reasoning] if args.reasoning else REASONING_LEVELS

    total_ok = total_fail = total_skip = 0
    configs = [(s, r) for s in settings for r in reasonings]

    for i, (setting, reasoning) in enumerate(configs, 1):
        trial_dir = base_dir / setting / reasoning
        imputed_dir = trial_dir / "answer_blocks_llm_imputed"

        if not trial_dir.exists():
            print(f"[{i}/{len(configs)}] {setting}__{reasoning} — SKIP (dir not found)")
            total_skip += 1
            continue

        # Find all response JSON files
        response_files = sorted(
            trial_dir.rglob("pid_*_response.json"),
            key=lambda f: int(re.search(r"pid_(\d+)", f.name).group(1))
        )
        if not response_files:
            print(f"[{i}/{len(configs)}] {setting}__{reasoning} — SKIP (no responses)")
            total_skip += 1
            continue

        ok = fail = 0
        for resp_file in response_files:
            pid_match = re.search(r"pid_(\d+)", resp_file.name)
            if not pid_match:
                continue
            pid = f"pid_{pid_match.group(1)}"
            try:
                result = postprocess_simulation_outputs_with_pid(
                    pid,
                    str(trial_dir),
                    QUESTION_JSON_DIR,
                    str(imputed_dir),
                )
                if result:
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1

        total_ok   += ok
        total_fail += fail
        status = "✓" if fail == 0 else f"⚠ {fail} failed"
        print(f"[{i}/{len(configs)}] {setting}__{reasoning} — {ok}/{ok+fail} ok  {status}")

    print(f"\nDone: {total_ok} ok, {total_fail} failed, {total_skip} skipped configs")


if __name__ == "__main__":
    main()
