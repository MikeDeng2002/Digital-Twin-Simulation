"""
create_ablation_50p_inputs.py — Build input dirs for pids 21-50 only (v2+v3 ablation).

Produces 8 dirs (4 conditions × 2 versions), each containing pid_21 through pid_50.
Named with _p21_50 suffix to distinguish from the existing 1-20 dirs.

Usage (from Digital-Twin-Simulation/):
    python skill_extraction/create_ablation_50p_inputs.py
"""

import re
from pathlib import Path

SKILLS_V2_DIR = Path("text_simulation/skills_v2")
QUESTION_DIR  = Path("text_simulation/text_questions")
OUTPUT_BASE   = Path("text_simulation")

PERSONA_HEADER    = "## Persona Skill Profile:\n"
SECTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"

ABLATION_CONFIGS = [
    ("v2_inferred", "bg",       ["background.txt"]),
    ("v2_inferred", "bg_dp",    ["background.txt", "decision_procedure.txt"]),
    ("v2_inferred", "bg_ep",    ["background.txt", "evaluation_profile.txt"]),
    ("v2_inferred", "bg_dp_ep", ["background.txt", "decision_procedure.txt", "evaluation_profile.txt"]),
    ("v3_maximum",  "bg",       ["background.txt"]),
    ("v3_maximum",  "bg_dp",    ["background.txt", "decision_procedure.txt"]),
    ("v3_maximum",  "bg_ep",    ["background.txt", "evaluation_profile.txt"]),
    ("v3_maximum",  "bg_dp_ep", ["background.txt", "decision_procedure.txt", "evaluation_profile.txt"]),
]

SECTION_HEADERS = {
    "background.txt":          "### Background",
    "decision_procedure.txt":  "### Decision Procedure",
    "evaluation_profile.txt":  "### Evaluation Profile",
}

VERSION_SHORT = {"v2_inferred": "v2", "v3_maximum": "v3"}

# Build exact pid→file map (same safe approach as create scripts)
_QUESTION_FILES: dict[str, Path] = {}
for _f in QUESTION_DIR.iterdir():
    _m = re.search(r"(pid_\d+)", _f.name)
    if _m and _m.group(1) not in _QUESTION_FILES:
        _QUESTION_FILES[_m.group(1)] = _f


def find_question_file(pid: str) -> Path | None:
    return _QUESTION_FILES.get(f"pid_{pid}")


def build_persona_text(skill_dir: Path, files: list[str]) -> str:
    parts = []
    for fname in files:
        content = (skill_dir / fname).read_text(encoding="utf-8").strip()
        parts.append(f"{SECTION_HEADERS[fname]}\n{content}")
    return "\n\n".join(parts)


def main():
    pids = [str(i) for i in range(21, 51)]
    total = 0

    for version, condition, files in ABLATION_CONFIGS:
        vshort  = VERSION_SHORT[version]
        # Naming: same as existing but with _p21_50 suffix
        if condition == "bg_dp_ep":
            out_dir = OUTPUT_BASE / f"text_simulation_input_skill_v2_{vshort}_p21_50"
        else:
            out_dir = OUTPUT_BASE / f"text_simulation_input_skill_v2_{vshort}_{condition}_p21_50"
        out_dir.mkdir(parents=True, exist_ok=True)

        done = skipped = 0
        for pid in pids:
            skill_dir = SKILLS_V2_DIR / f"pid_{pid}" / version
            missing   = [f for f in files if not (skill_dir / f).exists()]
            if missing:
                print(f"  pid_{pid} [{version}/{condition}] — SKIP (missing: {missing})")
                skipped += 1
                continue

            q_file = find_question_file(pid)
            if q_file is None:
                print(f"  pid_{pid} — SKIP (no question file)")
                skipped += 1
                continue

            assert q_file.stem == f"pid_{pid}", \
                f"Question file mismatch for pid_{pid}: got {q_file.name}"

            persona_text  = build_persona_text(skill_dir, files)
            question_text = q_file.read_text(encoding="utf-8")
            combined      = PERSONA_HEADER + persona_text + SECTION_SEPARATOR + question_text
            (out_dir / f"pid_{pid}_prompt.txt").write_text(combined, encoding="utf-8")
            done += 1

        total += done
        print(f"  [{vshort}/{condition}] → {out_dir.name}  ({done} written, {skipped} skipped)")

    print(f"\nTotal: {total} prompt files across {len(ABLATION_CONFIGS)} dirs.")


if __name__ == "__main__":
    main()
