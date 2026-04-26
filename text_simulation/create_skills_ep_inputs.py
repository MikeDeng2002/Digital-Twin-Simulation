"""
create_skills_ep_inputs.py — Build input directories combining skills/ (bg, dp, tools)
with evaluation_profile.txt from skills_v2/.

Produces 21 directories (7 ablation conditions × 3 skill versions):

  For each version (v1_direct → v1, v2_inferred → v2, v3_maximum → v3):
    text_simulation_input_skills_v{N}_bg             — background only
    text_simulation_input_skills_v{N}_bg_dp          — bg + decision_procedure
    text_simulation_input_skills_v{N}_bg_tools       — bg + tools
    text_simulation_input_skills_v{N}_bg_ep          — bg + evaluation_profile (from skills_v2)
    text_simulation_input_skills_v{N}_bg_dp_tools    — bg + dp + tools (full original skills)
    text_simulation_input_skills_v{N}_bg_dp_ep       — bg + dp + ep (no tools)
    text_simulation_input_skills_v{N}_bg_dp_tools_ep — all 4 files (new full)

Usage (from Digital-Twin-Simulation/):
    python text_simulation/create_skills_ep_inputs.py
    python text_simulation/create_skills_ep_inputs.py --pids 1,2,3
    python text_simulation/create_skills_ep_inputs.py --version v1
"""

import re
import argparse
from pathlib import Path

SKILLS_DIR    = Path("text_simulation/skills")
SKILLS_V2_DIR = Path("text_simulation/skills_v2")
QUESTION_DIR  = Path("text_simulation/text_questions")
OUTPUT_BASE   = Path("text_simulation")

# Mapping: short version label → folder name used in both skills/ and skills_v2/
VERSION_MAP = {
    "v1": "v1_direct",
    "v2": "v2_inferred",
    "v3": "v3_maximum",
}

# Ablation conditions: (label, [(source_dir_key, filename), ...])
# source_dir_key: "skills" = skills/, "skills_v2" = skills_v2/
ABLATION_CONFIGS = [
    ("bg",            [("skills",    "background.txt")]),
    ("bg_dp",         [("skills",    "background.txt"),
                       ("skills",    "decision_procedure.txt")]),
    ("bg_tools",      [("skills",    "background.txt"),
                       ("skills",    "tools.txt")]),
    ("bg_ep",         [("skills",    "background.txt"),
                       ("skills_v2", "evaluation_profile.txt")]),
    ("bg_dp_tools",   [("skills",    "background.txt"),
                       ("skills",    "decision_procedure.txt"),
                       ("skills",    "tools.txt")]),
    ("bg_dp_ep",      [("skills",    "background.txt"),
                       ("skills",    "decision_procedure.txt"),
                       ("skills_v2", "evaluation_profile.txt")]),
    ("bg_dp_tools_ep",[("skills",    "background.txt"),
                       ("skills",    "decision_procedure.txt"),
                       ("skills",    "tools.txt"),
                       ("skills_v2", "evaluation_profile.txt")]),
]

SECTION_HEADERS = {
    "background.txt":          "### Background",
    "decision_procedure.txt":  "### Decision Procedure",
    "tools.txt":               "### Tools",
    "evaluation_profile.txt":  "### Evaluation Profile",
}

PERSONA_HEADER    = "## Persona Skill Profile:\n"
SECTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"


# Build exact pid→file map once (same approach as create_text_simulation_input.py)
_QUESTION_FILES: dict[str, Path] = {}
for _f in QUESTION_DIR.iterdir():
    _m = re.search(r"(pid_\d+)", _f.name)
    if _m and _m.group(1) not in _QUESTION_FILES:
        _QUESTION_FILES[_m.group(1)] = _f


def find_question_file(pid: str) -> Path | None:
    return _QUESTION_FILES.get(f"pid_{pid}")


def get_source_dir(source_key: str, version_folder: str) -> Path:
    if source_key == "skills":
        return SKILLS_DIR / f"pid_{{pid}}" / version_folder
    else:
        return SKILLS_V2_DIR / f"pid_{{pid}}" / version_folder


def build_persona_text(pid: str, version_folder: str, file_specs: list) -> str:
    parts = []
    for source_key, fname in file_specs:
        if source_key == "skills":
            fpath = SKILLS_DIR / f"pid_{pid}" / version_folder / fname
        else:
            fpath = SKILLS_V2_DIR / f"pid_{pid}" / version_folder / fname
        header = SECTION_HEADERS[fname]
        content = fpath.read_text(encoding="utf-8").strip()
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


def check_files_exist(pid: str, version_folder: str, file_specs: list) -> list[str]:
    missing = []
    for source_key, fname in file_specs:
        if source_key == "skills":
            fpath = SKILLS_DIR / f"pid_{pid}" / version_folder / fname
        else:
            fpath = SKILLS_V2_DIR / f"pid_{pid}" / version_folder / fname
        if not fpath.exists():
            missing.append(f"{source_key}/{fname}")
    return missing


def build_ablation_dir(version_short: str, condition: str, file_specs: list, pids: list[str]):
    version_folder = VERSION_MAP[version_short]
    out_dir = OUTPUT_BASE / f"text_simulation_input_skills_{version_short}_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for pid in pids:
        missing = check_files_exist(pid, version_folder, file_specs)
        if missing:
            print(f"  pid_{pid} [{version_short}/{condition}] — SKIP (missing: {missing})")
            skipped += 1
            continue

        q_file = find_question_file(pid)
        if q_file is None:
            print(f"  pid_{pid} [{version_short}/{condition}] — SKIP (no question file)")
            skipped += 1
            continue

        # Sanity check: question file must be exactly pid_{pid}.txt
        expected_stem = f"pid_{pid}"
        assert q_file.stem == expected_stem, (
            f"Question file mismatch for pid_{pid}: got {q_file.name}, expected pid_{pid}.txt. "
            f"Fix find_question_file() — glob 'pid_{pid}*.txt' is ambiguous."
        )

        persona_text = build_persona_text(pid, version_folder, file_specs)
        question_text = q_file.read_text(encoding="utf-8")
        combined = PERSONA_HEADER + persona_text + SECTION_SEPARATOR + question_text

        (out_dir / f"pid_{pid}_prompt.txt").write_text(combined, encoding="utf-8")
        done += 1

    print(f"  [{version_short}/{condition}] → {out_dir}  ({done} written, {skipped} skipped)")
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2", "v3", "all"], default="all")
    parser.add_argument("--pids", default=None,
                        help="Comma-separated pid list (default: all 20 from skills_v2/)")
    args = parser.parse_args()

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",")]
    else:
        pids = sorted(
            {d.name.replace("pid_", "") for d in SKILLS_V2_DIR.glob("pid_*") if d.is_dir()},
            key=lambda x: int(x)
        )
        # Only use pids that also exist in skills/
        pids = [p for p in pids if (SKILLS_DIR / f"pid_{p}").is_dir()]
        print(f"Found {len(pids)} personas in both skills/ and skills_v2/")

    versions = ["v1", "v2", "v3"] if args.version == "all" else [args.version]

    total = 0
    for version_short in versions:
        for condition, file_specs in ABLATION_CONFIGS:
            total += build_ablation_dir(version_short, condition, file_specs, pids)

    print(f"\nTotal: {total} prompt files written across {len(versions)} versions × {len(ABLATION_CONFIGS)} conditions.")


if __name__ == "__main__":
    main()
