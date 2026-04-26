"""
create_skills_v4_inputs.py — Build combined LLM prompt files from v4 skill profiles.

Reads three skill files (background, decision_profile, evaluation_profile) from
text_simulation/skills_v4/pid_{pid}/ (note: v4 uses decision_profile.txt, not
decision_procedure.txt as v2 did) and combines each with the per-persona survey
question file from text_simulation/text_questions/pid_{pid}.txt to produce
text_simulation/text_simulation_input_skills_v4/pid_{pid}_prompt.txt.

Persona-safety: every pid is matched by EXACT filename (pid_{N}.txt), never by
regex prefix, and the survey file is required to exist for that pid. A previous
version of this pipeline had a mis-alignment bug where persona N was merged
with persona M's questions; here we assert exact-match to prevent recurrence.

Usage (from Digital-Twin-Simulation/):
    python text_simulation/create_skills_v4_inputs.py
    python text_simulation/create_skills_v4_inputs.py --pids 1,2,3
"""

import argparse
from pathlib import Path

SKILLS_V4_DIR = Path("text_simulation/skills_v4")
QUESTION_DIR  = Path("text_simulation/text_questions")
OUTPUT_DIR    = Path("text_simulation/text_simulation_input_skills_v4")

PERSONA_HEADER    = "## Persona Skill Profile:\n"
SECTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"

SKILL_FILES = [
    ("background.txt",         "### Background"),
    ("decision_profile.txt",   "### Decision Profile"),
    ("evaluation_profile.txt", "### Evaluation Profile"),
]


def build_persona_text(skill_dir: Path) -> str:
    parts = []
    for fname, header in SKILL_FILES:
        body = (skill_dir / fname).read_text(encoding="utf-8").strip()
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def build_one(pid: str) -> str:
    """Return 'ok' | 'skip: <reason>'."""
    skill_dir = SKILLS_V4_DIR / f"pid_{pid}"
    if not skill_dir.is_dir():
        return f"skip: skills_v4/pid_{pid}/ not found"

    missing = [f for f, _ in SKILL_FILES if not (skill_dir / f).exists()]
    if missing:
        return f"skip: missing {missing}"

    q_file = QUESTION_DIR / f"pid_{pid}.txt"
    if not q_file.exists():
        return f"skip: {q_file} not found"

    # Persona-safety: exact filename match, never a prefix/regex collision.
    assert q_file.name == f"pid_{pid}.txt", (
        f"Unexpected question filename for pid_{pid}: {q_file.name}"
    )

    persona_text  = build_persona_text(skill_dir)
    question_text = q_file.read_text(encoding="utf-8")
    combined      = PERSONA_HEADER + persona_text + SECTION_SEPARATOR + question_text

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"pid_{pid}_prompt.txt"
    out_path.write_text(combined, encoding="utf-8")
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", default=None,
                        help="Comma-separated pid list (default: all in skills_v4/)")
    args = parser.parse_args()

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",")]
    else:
        pids = sorted(
            (d.name.replace("pid_", "") for d in SKILLS_V4_DIR.glob("pid_*") if d.is_dir()),
            key=lambda x: int(x),
        )
        print(f"Found {len(pids)} personas in {SKILLS_V4_DIR}: {pids}")

    written = 0
    for pid in pids:
        result = build_one(pid)
        print(f"  pid_{pid}: {result}")
        if result == "ok":
            written += 1

    print(f"\nTotal: {written}/{len(pids)} prompt files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
