"""
generate_skills_ep_configs.py — Generate YAML configs for the skills+ep ablation experiment.

Creates 3 config suites (one per skill version), each with 7 conditions × 4 reasoning levels = 28 configs.
Total: 84 configs across 3 suites.

Suites:
  configs/nano_skills_ep_ablation_v1_temp0/  — v1_direct
  configs/nano_skills_ep_ablation_v2_temp0/  — v2_inferred
  configs/nano_skills_ep_ablation_v3_temp0/  — v3_maximum

Usage (from Digital-Twin-Simulation/):
    python text_simulation/generate_skills_ep_configs.py
"""

from pathlib import Path

REASONING_LEVELS = ["none", "low", "medium", "high"]
MAX_TOKENS = {"none": 16384, "low": 16384, "medium": 32768, "high": 32768}

CONDITIONS = ["bg", "bg_dp", "bg_tools", "bg_ep", "bg_dp_tools", "bg_dp_ep", "bg_dp_tools_ep"]

VERSIONS = ["v1", "v2", "v3"]

SYSTEM_INSTRUCTION = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described\n"
    "  in the 'Persona Skill Profile' above. Adhere to the persona and follow all formatting instructions carefully."
)

CONFIG_DIR_BASE = Path("text_simulation/configs")


def write_config(config_dir: Path, condition: str, reasoning: str, version: str):
    config_dir.mkdir(parents=True, exist_ok=True)
    input_dir  = f"text_simulation_input_skills_{version}_{condition}"
    suite_name = f"nano_skills_ep_ablation_{version}_temp0"
    output_dir = f"text_simulation_output_{suite_name}/{condition}/{reasoning}"

    content = f"""provider: openai
model_name: gpt-5.4-nano
temperature: 0.0
max_tokens: {MAX_TOKENS[reasoning]}
max_retries: 10
num_workers: 50
force_regenerate: false
max_personas: 20
input_folder_dir: {input_dir}
output_folder_dir: {output_dir}
system_instruction: {SYSTEM_INSTRUCTION}
reasoning_effort: {reasoning}
"""
    (config_dir / f"{condition}__{reasoning}.yaml").write_text(content, encoding="utf-8")


def main():
    total = 0
    for version in VERSIONS:
        suite_name = f"nano_skills_ep_ablation_{version}_temp0"
        config_dir = CONFIG_DIR_BASE / suite_name

        for condition in CONDITIONS:
            for reasoning in REASONING_LEVELS:
                write_config(config_dir, condition, reasoning, version)
                total += 1

        print(f"  [{suite_name}] → {config_dir}  ({len(CONDITIONS) * len(REASONING_LEVELS)} configs)")

    print(f"\nTotal: {total} config files written.")


if __name__ == "__main__":
    main()
