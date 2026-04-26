"""
create_order_experiment_inputs.py — Build inputs testing section order effects.

All conditions use v2_inferred files from skills/pid_1/:
  bg = background.txt
  dp = decision_procedure.txt
  tools = tools.txt

Conditions (section order variants):
  order_bg_dp_tools   — bg → dp → tools  (current new format)
  order_dp_bg_tools   — dp → bg → tools  (dp first)
  order_bg_tools_dp   — bg → tools → dp  (old format order — control)
  order_bg_dp         — bg → dp only
  order_dp_bg         — dp → bg only

Only pid_1, version v2_inferred.

Usage (from Digital-Twin-Simulation/):
    python text_simulation/create_order_experiment_inputs.py
"""

from pathlib import Path

SKILLS_DIR   = Path("text_simulation/skills/pid_1/v2_inferred")
QUESTION_FILE = Path("text_simulation/text_questions/pid_1.txt")
OUTPUT_BASE  = Path("text_simulation")

PERSONA_HEADER    = "## Persona Skill Profile:\n"
SECTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"

SECTION_HEADERS = {
    "background":          "### Background",
    "decision_procedure":  "### Decision Procedure",
    "tools":               "### Tools",
}

# (condition_name, ordered list of file keys)
CONDITIONS = [
    ("order_bg_dp_tools",  ["background", "decision_procedure", "tools"]),
    ("order_dp_bg_tools",  ["decision_procedure", "background", "tools"]),
    ("order_bg_tools_dp",  ["background", "tools", "decision_procedure"]),
    ("order_bg_dp",        ["background", "decision_procedure"]),
    ("order_dp_bg",        ["decision_procedure", "background"]),
]


def build_persona_text(file_keys):
    parts = []
    for key in file_keys:
        header  = SECTION_HEADERS[key]
        content = (SKILLS_DIR / f"{key}.txt").read_text(encoding="utf-8").strip()
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


def main():
    question_text = QUESTION_FILE.read_text(encoding="utf-8")

    for condition, file_keys in CONDITIONS:
        out_dir = OUTPUT_BASE / f"text_simulation_input_{condition}"
        out_dir.mkdir(parents=True, exist_ok=True)

        persona_text = build_persona_text(file_keys)
        combined = PERSONA_HEADER + persona_text + SECTION_SEPARATOR + question_text

        out_path = out_dir / "pid_1_prompt.txt"
        out_path.write_text(combined, encoding="utf-8")
        print(f"  [{condition}] → {out_path}  (order: {' → '.join(file_keys)})")

    print(f"\nDone: {len(CONDITIONS)} input dirs created for pid_1.")


if __name__ == "__main__":
    main()
