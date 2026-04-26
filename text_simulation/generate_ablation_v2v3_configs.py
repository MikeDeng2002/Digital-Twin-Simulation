"""
generate_ablation_v2v3_configs.py — Generate YAML configs for v2_inferred and v3_maximum ablation experiments.

Creates configs/nano_v2_ablation_v2_temp0/ and configs/nano_v2_ablation_v3_temp0/.
Each has 4 conditions × 4 reasoning levels = 16 configs per version.

Conditions:
  bg, bg_dp, bg_ep, bg_dp_ep

bg_dp_ep uses the full skill dirs (text_simulation_input_skill_v2_v2 / v3).
Others use the new ablation dirs built by create_ablation_inputs_v2v3.py.

Usage (from Digital-Twin-Simulation/):
    python text_simulation/generate_ablation_v2v3_configs.py
"""

from pathlib import Path

REASONING_LEVELS = ["none", "low", "medium", "high"]
MAX_TOKENS = {"none": 16384, "low": 16384, "medium": 32768, "high": 32768}

CONDITIONS = {
    "v2": {
        "bg":       "text_simulation_input_skill_v2_v2_bg",
        "bg_dp":    "text_simulation_input_skill_v2_v2_bg_dp",
        "bg_ep":    "text_simulation_input_skill_v2_v2_bg_ep",
        "bg_dp_ep": "text_simulation_input_skill_v2_v2",
    },
    "v3": {
        "bg":       "text_simulation_input_skill_v2_v3_bg",
        "bg_dp":    "text_simulation_input_skill_v2_v3_bg_dp",
        "bg_ep":    "text_simulation_input_skill_v2_v3_bg_ep",
        "bg_dp_ep": "text_simulation_input_skill_v2_v3",
    },
}

SYSTEM_INSTRUCTION = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described\n"
    "  in the 'Persona Skill Profile' above. Adhere to the persona and follow all formatting instructions carefully."
)

CONFIG_DIR_BASE = Path("text_simulation/configs")


def write_config(config_dir: Path, condition: str, reasoning: str, input_dir: str, output_base: str):
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{condition}__{reasoning}.yaml"
    content = f"""provider: openai
model_name: gpt-5.4-nano
temperature: 0.0
max_tokens: {MAX_TOKENS[reasoning]}
max_retries: 10
num_workers: 50
force_regenerate: false
max_personas: 20
input_folder_dir: {input_dir}
output_folder_dir: {output_base}/{condition}/{reasoning}
system_instruction: {SYSTEM_INSTRUCTION}
reasoning_effort: {reasoning}
"""
    (config_dir / fname).write_text(content, encoding="utf-8")


def main():
    total = 0
    for version_short, cond_map in CONDITIONS.items():
        suite_name = f"nano_v2_ablation_{version_short}_temp0"
        config_dir = CONFIG_DIR_BASE / suite_name
        output_base = f"text_simulation_output_{suite_name}"

        for condition, input_dir in cond_map.items():
            for reasoning in REASONING_LEVELS:
                write_config(config_dir, condition, reasoning, input_dir, output_base)
                total += 1

        print(f"  [{suite_name}] → {config_dir}  (16 configs)")

    print(f"\nTotal: {total} config files written.")


if __name__ == "__main__":
    main()
