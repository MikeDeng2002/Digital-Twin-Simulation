"""
create_skill_inputs.py — Build simulation input folders for skill-based experiments.

For each persona and each skill version, combines:
  skill files (background + tools + decision_procedure)
  + wave-4 question text (same as original experiment)
→ text_simulation/text_simulation_input_skill_{version}/pid_{pid}_prompt.txt

Usage (run from Digital-Twin-Simulation/):
    poetry run python skill_extraction/create_skill_inputs.py
    poetry run python skill_extraction/create_skill_inputs.py --pids 1,2,3,4,5,6,7,8,9,10,11
"""

import re
import argparse
from pathlib import Path

SKILLS_DIR = Path("text_simulation/skills")
QUESTIONS_DIR = Path("text_simulation/text_questions")
VERSIONS = ["v1_direct", "v2_inferred", "v3_maximum", "v4_chained"]

SKILL_HEADER = "## Persona Skill Profile:\n"
QUESTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"


def get_pids_with_skills(versions: list) -> list[str]:
    """Return PIDs that have all skill files for all requested versions."""
    pids = []
    for pid_dir in sorted(SKILLS_DIR.iterdir(), key=lambda p: int(p.name.replace("pid_", ""))):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name.replace("pid_", "")
        complete = all(
            (pid_dir / version / fname).exists()
            for version in versions
            for fname in ["background.txt", "tools.txt", "decision_procedure.txt"]
        )
        if complete:
            pids.append(pid)
    return pids


def load_skill_text(pid: str, version: str) -> str:
    base = SKILLS_DIR / f"pid_{pid}" / version
    background = (base / "background.txt").read_text(encoding="utf-8").strip()
    tools = (base / "tools.txt").read_text(encoding="utf-8").strip()
    decision = (base / "decision_procedure.txt").read_text(encoding="utf-8").strip()
    return f"### Background\n{background}\n\n### Information Sources\n{tools}\n\n### Decision Procedure\n{decision}"


def load_question_text(pid: str) -> str | None:
    for name in [f"pid_{pid}.txt", f"pid_{pid}_mega_persona.txt"]:
        path = QUESTIONS_DIR / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def create_inputs_for_version(pids: list[str], version: str):
    output_dir = Path(f"text_simulation/text_simulation_input_skill_{version}")
    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for pid in pids:
        out_path = output_dir / f"pid_{pid}_prompt.txt"
        if out_path.exists():
            skipped += 1
            continue

        question_text = load_question_text(pid)
        if question_text is None:
            print(f"  Warning: no question file for pid_{pid}, skipping")
            continue

        skill_text = load_skill_text(pid, version)
        combined = SKILL_HEADER + skill_text + QUESTION_SEPARATOR + question_text
        out_path.write_text(combined, encoding="utf-8")
        created += 1

    print(f"  [{version}] {created} created, {skipped} already existed → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create skill-based simulation input files.")
    parser.add_argument("--pids", type=str, default=None,
                        help="Comma-separated PIDs to process (default: all with complete skills)")
    parser.add_argument("--version", choices=VERSIONS + ["all"], default="all")
    args = parser.parse_args()

    versions = VERSIONS if args.version == "all" else [args.version]

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",")]
        print(f"Using {len(pids)} specified PIDs")
    else:
        pids = get_pids_with_skills(versions)
        print(f"Found {len(pids)} PIDs with complete skills: {pids}")

    for version in versions:
        create_inputs_for_version(pids, version)

    print("\nDone. Skill input folders created:")
    for version in versions:
        print(f"  text_simulation/text_simulation_input_skill_{version}/")


if __name__ == "__main__":
    main()
