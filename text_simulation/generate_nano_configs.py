"""
generate_nano_configs.py — Generate all YAML configs for the nano experiment.

Creates 40 config files (10 persona settings × 4 reasoning strengths) in
text_simulation/configs/nano/.

Each config specifies:
  - model: gpt-4.1-nano-2025-04-14
  - input_folder_dir: the persona input folder
  - output_folder_dir: separate output per (setting, reasoning)
  - system_instruction: varies by reasoning strength

Repetitions (3 per cell) are handled at runtime by run_nano_experiment.sh,
which runs each config with temperatures 0.0, 0.5, and 1.0, writing to
separate output folders (rep_1, rep_2, rep_3).

Usage (from Digital-Twin-Simulation/):
    python text_simulation/generate_nano_configs.py
"""

from pathlib import Path
import yaml

OUT_DIR = Path("text_simulation/configs/nano")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4.1-nano-2025-04-14"

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

# 4 reasoning strengths: (name, system_instruction)
REASONING_LEVELS = {
    "none": (
        "You are an AI assistant. Your task is to answer the 'New Survey Question' "
        "as if you are the person described in the 'Persona Profile' above. "
        "Adhere to the persona and follow all formatting instructions carefully."
    ),
    "low": (
        "You are an AI assistant. Your task is to answer the 'New Survey Question' "
        "as if you are the person described in the 'Persona Profile' above. "
        "Before giving your final answer, write one sentence noting the single most "
        "relevant trait or belief from the persona that drives your answer. "
        "Then provide the formatted answer."
    ),
    "medium": (
        "You are an AI assistant. Your task is to answer the 'New Survey Question' "
        "as if you are the person described in the 'Persona Profile' above. "
        "Before giving your final answer, briefly reason in 2-3 sentences: "
        "(1) what key background or values are most relevant? "
        "(2) what does this suggest their likely answer would be? "
        "Then provide the formatted answer."
    ),
    "high": (
        "You are an AI assistant. Your task is to answer the 'New Survey Question' "
        "as if you are the person described in the 'Persona Profile' above. "
        "Before giving your final answer, reason step by step: "
        "(1) What does the persona's background, education, and demographics suggest? "
        "(2) Which personality traits or values are most relevant to this question? "
        "(3) How would this person's decision-making style shape their answer? "
        "(4) What is the most consistent answer given all of the above? "
        "Then provide the formatted answer."
    ),
}

TEMPERATURES = {
    "rep_1": 0.0,
    "rep_2": 0.5,
    "rep_3": 1.0,
}

configs_written = []

for setting_name, input_dir in SETTINGS:
    for reasoning_name, system_instruction in REASONING_LEVELS.items():
        for rep_name, temperature in TEMPERATURES.items():
            config_name = f"{setting_name}__{reasoning_name}__{rep_name}"
            output_dir  = f"text_simulation_output_nano/{setting_name}/{reasoning_name}/{rep_name}"

            config = {
                "provider": "openai",
                "model_name": MODEL,
                "temperature": temperature,
                "max_tokens": 16384,
                "max_retries": 10,
                "num_workers": 50,
                "force_regenerate": False,
                "max_personas": 100,
                "input_folder_dir":  input_dir,
                "output_folder_dir": output_dir,
                "system_instruction": system_instruction,
            }

            out_path = OUT_DIR / f"{config_name}.yaml"
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                          sort_keys=False, width=120)

            configs_written.append(config_name)

print(f"Written {len(configs_written)} configs to {OUT_DIR}/")
print(f"  Settings:          {len(SETTINGS)}")
print(f"  Reasoning levels:  {len(REASONING_LEVELS)} (none, low, medium, high)")
print(f"  Repetitions:       {len(TEMPERATURES)} (rep_1=temp0.0, rep_2=temp0.5, rep_3=temp1.0)")
print(f"  Total:             {len(SETTINGS)} × {len(REASONING_LEVELS)} × {len(TEMPERATURES)} = {len(configs_written)}")
