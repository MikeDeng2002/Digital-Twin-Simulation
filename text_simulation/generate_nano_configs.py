"""
generate_nano_configs.py — Generate all YAML configs for the nano experiment.

Creates 40 config files (10 persona settings × 4 reasoning strengths) in
text_simulation/configs/nano/.

Each config specifies:
  - model: gpt-4.1-nano-2025-04-14
  - temperature: 1.0 (fixed for all configs)
  - input_folder_dir: the persona input folder
  - output_folder_dir: separate output per (setting, reasoning, rep)
  - system_instruction: varies by reasoning strength

Repetitions (3 per cell) are handled at runtime by run_nano_experiment.sh,
which runs each config 3 times writing to rep_1, rep_2, rep_3 subfolders.
All repetitions use temperature=1.0 so variance comes from sampling randomness.

Usage (from Digital-Twin-Simulation/):
    python text_simulation/generate_nano_configs.py
"""

from pathlib import Path
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--temp0", action="store_true",
                    help="Generate temperature=0.0 configs (no reps, deterministic)")
gen_args = parser.parse_args()

if gen_args.temp0:
    OUT_DIR     = Path("text_simulation/configs/nano_temp0")
    TEMPERATURE = 0.0
    N_REPS      = 1   # deterministic — no need for repetitions
else:
    OUT_DIR     = Path("text_simulation/configs/nano")
    TEMPERATURE = 1.0
    N_REPS      = 3   # sample randomness across 3 reps

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove any old configs first
for old in OUT_DIR.glob("*.yaml"):
    old.unlink()

MODEL = "gpt-5.4-nano"

# 10 persona settings: (config_name, input_folder_dir)
SETTINGS = [
    ("skill_v1",         "text_simulation_input_skill_v1_direct"),
    ("skill_v2",         "text_simulation_input_skill_v2_inferred"),
    ("skill_v3",         "text_simulation_input_skill_v3_maximum"),
    ("raw",              "text_simulation_input"),
    ("raw_start_v1",     "text_simulation_input_raw_start_skill_v1"),
    ("raw_start_v2",     "text_simulation_input_raw_start_skill_v2"),
    ("raw_start_v3",     "text_simulation_input_raw_start_skill_v3"),
    ("skill_v1_raw_end", "text_simulation_input_skill_v1_raw_end"),
    ("skill_v2_raw_end", "text_simulation_input_skill_v2_raw_end"),
    ("skill_v3_raw_end", "text_simulation_input_skill_v3_raw_end"),
]

# Shared system instruction (same for all reasoning levels — effort is set via API parameter)
SYSTEM_INSTRUCTION = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' "
    "as if you are the person described in the 'Persona Profile' above. "
    "Adhere to the persona and follow all formatting instructions carefully."
)

# 4 reasoning effort levels (passed as reasoning={"effort": ...} in responses API)
REASONING_LEVELS = ["none", "low", "medium", "high"]

configs_written = []

for setting_name, input_dir in SETTINGS:
    for reasoning_name in REASONING_LEVELS:
        config_name = f"{setting_name}__{reasoning_name}"
        # output_folder_dir uses {rep} placeholder; runner fills it in per repetition
        output_dir_template = f"text_simulation_output_nano/{setting_name}/{reasoning_name}/{{rep}}"

        # temp0: single run, no {rep} placeholder needed
        if TEMPERATURE == 0.0:
            output_dir_val = f"text_simulation_output_nano_temp0/{setting_name}/{reasoning_name}"
        else:
            output_dir_val = output_dir_template

        config = {
            "provider": "openai",
            "model_name": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": 16384,
            "max_retries": 10,
            "num_workers": 50,
            "force_regenerate": False,
            "max_personas": 5,
            "input_folder_dir":   input_dir,
            "output_folder_dir":  output_dir_val,
            "system_instruction": SYSTEM_INSTRUCTION,
            "reasoning_effort":   reasoning_name,
        }

        out_path = OUT_DIR / f"{config_name}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=120)

        configs_written.append(config_name)

print(f"Written {len(configs_written)} configs to {OUT_DIR}/")
print(f"  Settings:         {len(SETTINGS)}")
print(f"  Reasoning levels: {len(REASONING_LEVELS)} (none, low, medium, high) — via responses API reasoning param")
print(f"  Temperature:      {TEMPERATURE}")
print(f"  Repetitions:      {N_REPS} per config")
print(f"  Total runs:       {len(SETTINGS)} × {len(REASONING_LEVELS)} × {N_REPS} = {len(configs_written) * N_REPS}")
