"""
generate_o4mini_configs.py — Generate YAML configs for the o4-mini experiment.

Creates 10 config files (one per persona setting) in
text_simulation/configs/o4mini/.

Each config:
  - model: o4-mini
  - provider: openai
  - no reasoning_effort (model reasons internally)
  - no temperature (o4-mini ignores it)
  - max_personas: 5

Usage (from Digital-Twin-Simulation/):
    python text_simulation/generate_o4mini_configs.py
"""

from pathlib import Path
import yaml

OUT_DIR = Path("text_simulation/configs/o4mini")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove old configs
for old in OUT_DIR.glob("*.yaml"):
    old.unlink()

MODEL = "o4-mini"

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

SYSTEM_INSTRUCTION = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' "
    "as if you are the person described in the 'Persona Profile' above. "
    "Adhere to the persona and follow all formatting instructions carefully."
)

configs_written = []

for setting_name, input_dir in SETTINGS:
    config = {
        "provider":           "openai",
        "model_name":         MODEL,
        "max_tokens":         16384,
        "max_retries":        10,
        "num_workers":        20,
        "force_regenerate":   False,
        "max_personas":       5,
        "input_folder_dir":   input_dir,
        "output_folder_dir":  f"text_simulation_output_o4mini/{setting_name}",
        "system_instruction": SYSTEM_INSTRUCTION,
    }

    out_path = OUT_DIR / f"{setting_name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120)
    configs_written.append(setting_name)

print(f"Written {len(configs_written)} configs to {OUT_DIR}/")
for name in configs_written:
    print(f"  {name}")
