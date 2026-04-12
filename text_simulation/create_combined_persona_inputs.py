"""
create_combined_persona_inputs.py — Build combined persona input folders for the nano experiment.

Creates 6 new input folders that combine raw Q&A text with skill profiles:
  - raw_start_skill_v1/v2/v3  : raw text first, then skill profile
  - skill_v1/v2/v3_raw_end    : skill profile first, then raw text

The question block (survey question to answer) is always appended last,
identical to how create_text_simulation_input.py works.

Usage (from Digital-Twin-Simulation/):
    python text_simulation/create_combined_persona_inputs.py
    python text_simulation/create_combined_persona_inputs.py --pids 1,2,3
"""

import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm

BASE = Path("text_simulation")

# Source folders
RAW_DIR        = BASE / "text_simulation_input"            # full raw Q&A prompts (already has question appended)
SKILL_DIRS = {
    "v1": BASE / "text_simulation_input_skill_v1_direct",
    "v2": BASE / "text_simulation_input_skill_v2_inferred",
    "v3": BASE / "text_simulation_input_skill_v3_maximum",
}

# Output folders
OUTPUT_DIRS = {
    "raw_start_v1": BASE / "text_simulation_input_raw_start_skill_v1",
    "raw_start_v2": BASE / "text_simulation_input_raw_start_skill_v2",
    "raw_start_v3": BASE / "text_simulation_input_raw_start_skill_v3",
    "skill_v1_raw_end": BASE / "text_simulation_input_skill_v1_raw_end",
    "skill_v2_raw_end": BASE / "text_simulation_input_skill_v2_raw_end",
    "skill_v3_raw_end": BASE / "text_simulation_input_skill_v3_raw_end",
}

SEPARATOR = "\n\n---\n"
QUESTION_SEPARATOR = "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"


def extract_persona_block(prompt_text: str) -> str:
    """Extract just the persona section (before the question separator)."""
    if QUESTION_SEPARATOR in prompt_text:
        return prompt_text.split(QUESTION_SEPARATOR)[0]
    # Fallback: return everything
    return prompt_text


def extract_question_block(prompt_text: str) -> str:
    """Extract just the question section (after the question separator)."""
    if QUESTION_SEPARATOR in prompt_text:
        return prompt_text.split(QUESTION_SEPARATOR)[1]
    return ""


def extract_raw_persona_content(raw_prompt: str) -> str:
    """Strip the header from raw persona block."""
    header = "## Persona Profile (This individual's past survey responses):\n"
    persona_block = extract_persona_block(raw_prompt)
    if persona_block.startswith(header):
        return persona_block[len(header):]
    return persona_block


def extract_skill_persona_content(skill_prompt: str) -> str:
    """Strip the header from skill persona block."""
    header = "## Persona Skill Profile:\n"
    persona_block = extract_persona_block(skill_prompt)
    if persona_block.startswith(header):
        return persona_block[len(header):]
    return persona_block


def build_raw_start_prompt(raw_prompt: str, skill_prompt: str, version: str) -> str:
    """Raw Q&A first, then skill profile, then question."""
    raw_content = extract_raw_persona_content(raw_prompt)
    skill_content = extract_skill_persona_content(skill_prompt)
    question = extract_question_block(skill_prompt)

    persona_block = (
        f"## Persona Profile:\n"
        f"### Part 1: Full Survey Responses (Waves 1–3)\n"
        f"{raw_content}"
        f"{SEPARATOR}"
        f"### Part 2: Extracted Skill Profile (v{version[-1]} synthesis)\n"
        f"{skill_content}"
    )
    return persona_block + QUESTION_SEPARATOR + question


def build_skill_raw_end_prompt(raw_prompt: str, skill_prompt: str, version: str) -> str:
    """Skill profile first, then raw Q&A, then question."""
    raw_content = extract_raw_persona_content(raw_prompt)
    skill_content = extract_skill_persona_content(skill_prompt)
    question = extract_question_block(skill_prompt)

    persona_block = (
        f"## Persona Profile:\n"
        f"### Part 1: Extracted Skill Profile (v{version[-1]} synthesis)\n"
        f"{skill_content}"
        f"{SEPARATOR}"
        f"### Part 2: Full Survey Responses (Waves 1–3)\n"
        f"{raw_content}"
    )
    return persona_block + QUESTION_SEPARATOR + question


def get_pids(folder: Path) -> list[str]:
    pids = []
    for f in folder.iterdir():
        m = re.search(r"pid_(\d+)_prompt\.txt", f.name)
        if m:
            pids.append(m.group(1))
    return sorted(pids, key=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", type=str, default=None,
                        help="Comma-separated PIDs to process (default: all matching)")
    args = parser.parse_args()

    # Determine PIDs: intersection of raw and all skill folders
    raw_pids = set(get_pids(RAW_DIR))
    skill_pids = {v: set(get_pids(d)) for v, d in SKILL_DIRS.items()}
    all_skill_pids = skill_pids["v1"] & skill_pids["v2"] & skill_pids["v3"]
    common_pids = sorted(raw_pids & all_skill_pids, key=int)

    if args.pids:
        requested = set(args.pids.split(","))
        common_pids = [p for p in common_pids if p in requested]

    print(f"Building combined inputs for {len(common_pids)} personas...")

    for out_key, out_dir in OUTPUT_DIRS.items():
        out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("raw_start_v1",  "v1", build_raw_start_prompt),
        ("raw_start_v2",  "v2", build_raw_start_prompt),
        ("raw_start_v3",  "v3", build_raw_start_prompt),
        ("skill_v1_raw_end", "v1", build_skill_raw_end_prompt),
        ("skill_v2_raw_end", "v2", build_skill_raw_end_prompt),
        ("skill_v3_raw_end", "v3", build_skill_raw_end_prompt),
    ]

    for out_key, version, build_fn in configs:
        out_dir = OUTPUT_DIRS[out_key]
        skill_dir = SKILL_DIRS[version]
        print(f"\nBuilding {out_key} → {out_dir}")
        for pid in tqdm(common_pids, desc=out_key):
            raw_path   = RAW_DIR   / f"pid_{pid}_prompt.txt"
            skill_path = skill_dir / f"pid_{pid}_prompt.txt"
            if not raw_path.exists() or not skill_path.exists():
                continue
            raw_text   = raw_path.read_text(encoding="utf-8")
            skill_text = skill_path.read_text(encoding="utf-8")
            combined   = build_fn(raw_text, skill_text, version)
            out_path   = out_dir / f"pid_{pid}_prompt.txt"
            out_path.write_text(combined, encoding="utf-8")

    print(f"\nDone. Combined input folders:")
    for out_dir in OUTPUT_DIRS.values():
        n = len(list(out_dir.glob("pid_*_prompt.txt")))
        print(f"  {out_dir}  ({n} files)")


if __name__ == "__main__":
    main()
