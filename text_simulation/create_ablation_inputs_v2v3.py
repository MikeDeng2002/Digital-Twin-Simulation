"""
create_ablation_inputs_v2v3.py — Build ablation input directories for v2_inferred and v3_maximum.

Produces 6 new directories (3 ablation conditions × 2 versions):
  text_simulation_input_skill_v2_v2_bg       — v2_inferred background only
  text_simulation_input_skill_v2_v2_bg_dp    — v2_inferred bg + decision_procedure
  text_simulation_input_skill_v2_v2_bg_ep    — v2_inferred bg + evaluation_profile
  text_simulation_input_skill_v2_v3_bg       — v3_maximum background only
  text_simulation_input_skill_v2_v3_bg_dp    — v3_maximum bg + decision_procedure
  text_simulation_input_skill_v2_v3_bg_ep    — v3_maximum bg + evaluation_profile

bg_dp_ep (all 3 files) already exists:
  text_simulation_input_skill_v2_v2   — v2_inferred full
  text_simulation_input_skill_v2_v3   — v3_maximum full

Usage (from Digital-Twin-Simulation/):
    python text_simulation/create_ablation_inputs_v2v3.py
"""

import re
from pathlib import Path

SKILLS_V2_DIR = Path("text_simulation/skills_v2")
QUESTION_DIR  = Path("text_simulation/text_questions")
OUTPUT_BASE   = Path("text_simulation")

# Build exact pid→file map once (same approach as create_text_simulation_input.py)
_QUESTION_FILES: dict[str, Path] = {}
for _f in QUESTION_DIR.iterdir():
    _m = re.search(r"(pid_\d+)", _f.name)
    if _m and _m.group(1) not in _QUESTION_FILES:
        _QUESTION_FILES[_m.group(1)] = _f

PERSONA_HEADER    = "## Persona Skill Profile:\n"
SECTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"

# (version, ablation_label, files_to_include)
ABLATION_CONFIGS = [
    ("v1_direct",  "bg",       ["background.txt"]),
    ("v1_direct",  "bg_dp",    ["background.txt", "decision_procedure.txt"]),
    ("v1_direct",  "bg_ep",    ["background.txt", "evaluation_profile.txt"]),
    ("v2_inferred", "bg",       ["background.txt"]),
    ("v2_inferred", "bg_dp",    ["background.txt", "decision_procedure.txt"]),
    ("v2_inferred", "bg_ep",    ["background.txt", "evaluation_profile.txt"]),
    ("v3_maximum",  "bg",       ["background.txt"]),
    ("v3_maximum",  "bg_dp",    ["background.txt", "decision_procedure.txt"]),
    ("v3_maximum",  "bg_ep",    ["background.txt", "evaluation_profile.txt"]),
]

SECTION_HEADERS = {
    "background.txt":           "### Background",
    "decision_procedure.txt":   "### Decision Procedure",
    "evaluation_profile.txt":   "### Evaluation Profile",
}

VERSION_SHORT = {
    "v1_direct":  "v1",
    "v2_inferred": "v2",
    "v3_maximum":  "v3",
}


def find_question_file(pid: str) -> Path | None:
    return _QUESTION_FILES.get(f"pid_{pid}")


def build_persona_text(skill_dir: Path, files: list[str]) -> str:
    parts = []
    for fname in files:
        header = SECTION_HEADERS[fname]
        content = (skill_dir / fname).read_text(encoding="utf-8").strip()
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


def build_ablation_dir(version: str, ablation: str, files: list[str], pids: list[str]):
    vshort = VERSION_SHORT[version]
    # v1_direct uses no version prefix (matches existing configs: skill_v2_bg not skill_v2_v1_bg)
    if vshort == "v1":
        out_dir = OUTPUT_BASE / f"text_simulation_input_skill_v2_{ablation}"
    else:
        out_dir = OUTPUT_BASE / f"text_simulation_input_skill_v2_{vshort}_{ablation}"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for pid in pids:
        skill_dir = SKILLS_V2_DIR / f"pid_{pid}" / version
        missing = [f for f in files if not (skill_dir / f).exists()]
        if missing:
            print(f"  pid_{pid} [{version}/{ablation}] — SKIP (missing: {missing})")
            skipped += 1
            continue

        q_file = find_question_file(pid)
        if q_file is None:
            print(f"  pid_{pid} [{version}/{ablation}] — SKIP (no question file)")
            skipped += 1
            continue

        assert q_file.stem == f"pid_{pid}", (
            f"Question file mismatch for pid_{pid}: got {q_file.name}, expected pid_{pid}.txt"
        )

        persona_text = build_persona_text(skill_dir, files)
        question_text = q_file.read_text(encoding="utf-8")
        combined = PERSONA_HEADER + persona_text + SECTION_SEPARATOR + question_text

        (out_dir / f"pid_{pid}_prompt.txt").write_text(combined, encoding="utf-8")
        done += 1

    print(f"  [{version}/{ablation}] → {out_dir}  ({done} written, {skipped} skipped)")
    return done


def main():
    pids = sorted(
        {d.name.replace("pid_", "") for d in SKILLS_V2_DIR.glob("pid_*") if d.is_dir()},
        key=lambda x: int(x)
    )
    print(f"Found {len(pids)} personas in {SKILLS_V2_DIR}")

    total = 0
    for version, ablation, files in ABLATION_CONFIGS:
        total += build_ablation_dir(version, ablation, files, pids)

    print(f"\nTotal: {total} prompt files written across {len(ABLATION_CONFIGS)} ablation configs.")


if __name__ == "__main__":
    main()
