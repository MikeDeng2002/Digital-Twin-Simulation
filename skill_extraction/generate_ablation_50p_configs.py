"""
generate_ablation_50p_configs.py — Configs for 50-persona ablation (v2+v3, high reasoning only).

4 conditions × 2 versions × 1 reasoning = 8 configs total.
Suite: nano_v2_ablation_50p_{version}_temp0
"""

from pathlib import Path

# Input dirs contain ONLY pids 21-50 (created by create_ablation_50p_inputs.py)
CONDITIONS = {
    "v2": {
        "bg":       "text_simulation_input_skill_v2_v2_bg_p21_50",
        "bg_dp":    "text_simulation_input_skill_v2_v2_bg_dp_p21_50",
        "bg_ep":    "text_simulation_input_skill_v2_v2_bg_ep_p21_50",
        "bg_dp_ep": "text_simulation_input_skill_v2_v2_p21_50",
    },
    "v3": {
        "bg":       "text_simulation_input_skill_v2_v3_bg_p21_50",
        "bg_dp":    "text_simulation_input_skill_v2_v3_bg_dp_p21_50",
        "bg_ep":    "text_simulation_input_skill_v2_v3_bg_ep_p21_50",
        "bg_dp_ep": "text_simulation_input_skill_v2_v3_p21_50",
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
        suite      = f"nano_v2_ablation_50p_{version}_temp0"
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
max_personas: 30
input_folder_dir: {input_dir}
output_folder_dir: {output_dir}
system_instruction: {SYSTEM_INSTRUCTION}
reasoning_effort: high
"""
            (config_dir / f"{condition}__high.yaml").write_text(content, encoding="utf-8")
            total += 1

        print(f"  [{suite}] — 4 configs")

    print(f"\nTotal: {total} configs")


if __name__ == "__main__":
    main()
