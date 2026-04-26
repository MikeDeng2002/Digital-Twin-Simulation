"""
create_skill_v2_inputs.py — Build combined LLM prompt files from v2 skill profiles.

Reads three skill files (background, decision_procedure, evaluation_profile) from
text_simulation/skills_v2/pid_{pid}/{version}/ and combines each with the question
prompt to produce text_simulation/text_simulation_input_skill_v2_{version}/pid_{pid}_prompt.txt

Usage (from Digital-Twin-Simulation/):
    # All 3 versions (default)
    python text_simulation/create_skill_v2_inputs.py

    # Single version
    python text_simulation/create_skill_v2_inputs.py --version v1_direct

    # Custom pid list
    python text_simulation/create_skill_v2_inputs.py --pids 1,2,3
"""

import re
import argparse
from pathlib import Path

SKILLS_V2_DIR    = Path("text_simulation/skills_v2")
QUESTION_DIR     = Path("text_simulation/text_questions")
OUTPUT_BASE      = Path("text_simulation")

VERSIONS = ["v1_direct", "v2_inferred", "v3_maximum"]

# Input folders names (must match what configs reference)
VERSION_TO_INPUT_DIR = {
    "v1_direct":  "text_simulation_input_skill_v2_v1",
    "v2_inferred": "text_simulation_input_skill_v2_v2",
    "v3_maximum": "text_simulation_input_skill_v2_v3",
}

PERSONA_HEADER    = "## Persona Skill Profile:\n"
SECTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"

# Build exact pid→file map once (same approach as create_text_simulation_input.py)
_QUESTION_FILES: dict[str, Path] = {}
for _f in QUESTION_DIR.iterdir():
    _m = re.search(r"(pid_\d+)", _f.name)
    if _m and _m.group(1) not in _QUESTION_FILES:
        _QUESTION_FILES[_m.group(1)] = _f


def build_persona_text(skill_dir: Path) -> str:
    """Concatenate the 3 v2 skill files into a single persona text block."""
    parts = []

    bg = (skill_dir / "background.txt").read_text(encoding="utf-8").strip()
    parts.append(f"### Background\n{bg}")

    dp = (skill_dir / "decision_procedure.txt").read_text(encoding="utf-8").strip()
    parts.append(f"### Decision Procedure\n{dp}")

    ep = (skill_dir / "evaluation_profile.txt").read_text(encoding="utf-8").strip()
    parts.append(f"### Evaluation Profile\n{ep}")

    return "\n\n".join(parts)


def find_question_file(pid: str) -> Path | None:
    """Find the question prompt file for this pid."""
    return _QUESTION_FILES.get(f"pid_{pid}")
    return None


def build_inputs_for_version(version: str, pids: list[str]) -> int:
    out_dir = OUTPUT_BASE / VERSION_TO_INPUT_DIR[version]
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    for pid in pids:
        skill_dir = SKILLS_V2_DIR / f"pid_{pid}" / version

        # Check all 3 files exist
        files_needed = ["background.txt", "decision_procedure.txt", "evaluation_profile.txt"]
        missing = [f for f in files_needed if not (skill_dir / f).exists()]
        if missing:
            print(f"  pid_{pid} [{version}] — SKIP (missing: {missing})")
            skipped += 1
            continue

        q_file = find_question_file(pid)
        if q_file is None:
            print(f"  pid_{pid} [{version}] — SKIP (no question file found in {QUESTION_DIR})")
            skipped += 1
            continue

        assert q_file.stem == f"pid_{pid}", (
            f"Question file mismatch for pid_{pid}: got {q_file.name}, expected pid_{pid}.txt"
        )

        persona_text = build_persona_text(skill_dir)
        question_text = q_file.read_text(encoding="utf-8")
        combined = PERSONA_HEADER + persona_text + SECTION_SEPARATOR + question_text

        out_path = out_dir / f"pid_{pid}_prompt.txt"
        out_path.write_text(combined, encoding="utf-8")
        done += 1

    print(f"  [{version}] → {out_dir}  ({done} written, {skipped} skipped)")
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=VERSIONS + ["all"], default="all",
                        help="Which skill version to build inputs for (default: all)")
    parser.add_argument("--pids", default=None,
                        help="Comma-separated pid list (default: all available in skills_v2/)")
    args = parser.parse_args()

    # Discover pids
    if args.pids:
        pids = [p.strip() for p in args.pids.split(",")]
    else:
        pids = sorted(
            {d.name.replace("pid_", "") for d in SKILLS_V2_DIR.glob("pid_*") if d.is_dir()},
            key=lambda x: int(x)
        )
        print(f"Found {len(pids)} personas in {SKILLS_V2_DIR}")

    versions = VERSIONS if args.version == "all" else [args.version]

    total = 0
    for version in versions:
        total += build_inputs_for_version(version, pids)

    print(f"\nTotal: {total} prompt files written across {len(versions)} version(s).")


if __name__ == "__main__":
    main()
