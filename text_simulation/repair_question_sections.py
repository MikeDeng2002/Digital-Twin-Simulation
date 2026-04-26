"""
repair_question_sections.py — Fix question sections in any broken input dir.

For each pid_{N}_prompt.txt in a directory, replaces whatever question section
is currently present with the correct content from text_questions/pid_{N}.txt.

Works on any prompt format (raw persona, skill v2, minimal, etc.) as long as
the separator line is present.

Usage (from Digital-Twin-Simulation/):
    # Fix specific dirs
    python text_simulation/repair_question_sections.py \
        text_simulation_input_simple \
        text_simulation_input_demographic \
        text_simulation_input_minimal_demographics_only \
        text_simulation_input_minimal_with_signals \
        text_simulation_input_skill_v2_fix_A

    # Fix all dirs that fail verification
    python text_simulation/repair_question_sections.py --all
"""

import re
import argparse
from pathlib import Path

QUESTION_DIR = Path("text_simulation/text_questions")
INPUT_BASE   = Path("text_simulation")

SEPARATOR = "---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"


def repair_dir(input_dir: Path) -> tuple[int, int, int]:
    fixed = skipped = failed = 0
    for prompt_file in sorted(input_dir.glob("pid_*_prompt.txt"),
                               key=lambda f: int(re.search(r"pid_(\d+)", f.name).group(1))):
        pid = re.search(r"pid_(\d+)", prompt_file.name).group(1)
        correct_q_file = QUESTION_DIR / f"pid_{pid}.txt"

        if not correct_q_file.exists():
            print(f"  pid_{pid}: SKIP — pid_{pid}.txt not in {QUESTION_DIR}")
            skipped += 1
            continue

        prompt_text = prompt_file.read_text(encoding="utf-8")
        correct_q   = correct_q_file.read_text(encoding="utf-8")

        if SEPARATOR not in prompt_text:
            print(f"  pid_{pid}: SKIP — separator not found in prompt")
            skipped += 1
            continue

        persona_section = prompt_text.split(SEPARATOR, 1)[0]
        current_q       = prompt_text.split(SEPARATOR, 1)[1]

        if current_q.strip() == correct_q.strip():
            skipped += 1
            continue

        # Assert the pid in filename matches the correct question file
        assert (QUESTION_DIR / f"pid_{pid}.txt").stem == f"pid_{pid}"

        repaired = persona_section + SEPARATOR + correct_q
        prompt_file.write_text(repaired, encoding="utf-8")
        fixed += 1

    return fixed, skipped, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", help="Input dir names to repair (relative to text_simulation/)")
    parser.add_argument("--all", action="store_true", help="Repair all text_simulation_input_* dirs")
    args = parser.parse_args()

    if args.all:
        dirs = sorted(d for d in INPUT_BASE.glob("text_simulation_input_*") if d.is_dir())
    else:
        dirs = [INPUT_BASE / d for d in args.dirs]

    total_fixed = total_skipped = 0
    for d in dirs:
        if not d.exists():
            print(f"SKIP {d.name} — directory not found")
            continue
        fixed, skipped, _ = repair_dir(d)
        total_fixed   += fixed
        total_skipped += skipped
        status = f"fixed {fixed}" if fixed else "already OK"
        print(f"  {d.name}: {status}  ({skipped} unchanged)")

    print(f"\nTotal: {total_fixed} files repaired, {total_skipped} unchanged.")


if __name__ == "__main__":
    main()
