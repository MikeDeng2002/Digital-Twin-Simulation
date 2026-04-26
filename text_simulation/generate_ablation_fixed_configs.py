"""
generate_ablation_fixed_configs.py — Generate configs for the fixed ablation experiment.

4 conditions × 3 versions × high reasoning only = 12 configs.

Conditions (skills_v2 source, correct question files):
  bg        — background only
  bg_dp     — background + decision_procedure
  bg_ep     — background + evaluation_profile
  bg_dp_ep  — full (bg + dp + ep)

Versions: v1_direct, v2_inferred, v3_maximum

Suite name: nano_v2_ablation_fixed_{version}_temp0

Usage (from Digital-Twin-Simulation/):
    python text_simulation/generate_ablation_fixed_configs.py
"""

from pathlib import Path

CONDITIONS = {
    "v1": {
        "bg":       "text_simulation_input_skill_v2_bg",
        "bg_dp":    "text_simulation_input_skill_v2_bg_dp",
        "bg_ep":    "text_simulation_input_skill_v2_bg_ep",
        "bg_dp_ep": "text_simulation_input_skill_v2_v1",
    },
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


def main():
    total = 0
    for version, cond_map in CONDITIONS.items():
        suite      = f"nano_v2_ablation_fixed_{version}_temp0"
        config_dir = CONFIG_DIR_BASE / suite
        config_dir.mkdir(parents=True, exist_ok=True)

        for condition, input_dir in cond_map.items():
            output_dir = f"text_simulation_output_{suite}/{condition}/high"
            content = f"""provider: openai
model_name: gpt-5.4-nano
temperature: 0.0
max_tokens: 32768
max_retries: 10
num_workers: 50
force_regenerate: false
max_personas: 20
input_folder_dir: {input_dir}
output_folder_dir: {output_dir}
system_instruction: {SYSTEM_INSTRUCTION}
reasoning_effort: high
"""
            (config_dir / f"{condition}__high.yaml").write_text(content, encoding="utf-8")
            total += 1

        print(f"  [{suite}] → {config_dir}  (4 configs)")

    print(f"\nTotal: {total} config files.")


if __name__ == "__main__":
    main()
