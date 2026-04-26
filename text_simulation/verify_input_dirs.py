"""
verify_input_dirs.py — Sanity-check all text_simulation_input_* directories.

For every pid_{N}_prompt.txt, confirms that the question section inside the
prompt matches exactly the content of text_questions/pid_{N}.txt.

Catches the silent bug where glob('pid_{N}*.txt') returns a wrong file
(e.g. pid_1251.txt instead of pid_1.txt).

Usage (from Digital-Twin-Simulation/):
    python text_simulation/verify_input_dirs.py                  # all input dirs
    python text_simulation/verify_input_dirs.py --dir text_simulation_input_skills_v2_bg_dp_tools
"""

import re
import argparse
from pathlib import Path

QUESTION_DIR   = Path("text_simulation/text_questions")
INPUT_BASE     = Path("text_simulation")
SEPARATOR      = "---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"


def extract_question_section(prompt_text: str) -> str:
    if SEPARATOR in prompt_text:
        return prompt_text.split(SEPARATOR, 1)[1]
    return ""


def verify_dir(input_dir: Path) -> tuple[int, int, list[str], bool]:
    ok = fail = 0
    errors = []

    # Check if ANY file in this dir uses the separator format
    has_separator = any(
        SEPARATOR in f.read_text(encoding="utf-8")
        for f in list(input_dir.glob("pid_*_prompt.txt"))[:3]
    )
    if not has_separator:
        return 0, 0, ["[SKIP] No separator — uses different format (e.g. raw interaction)"], True

    for prompt_file in sorted(input_dir.glob("pid_*_prompt.txt"),
                               key=lambda f: int(re.search(r"pid_(\d+)", f.name).group(1))):
        pid = re.search(r"pid_(\d+)", prompt_file.name).group(1)
        expected_q_file = QUESTION_DIR / f"pid_{pid}.txt"

        if not expected_q_file.exists():
            errors.append(f"  pid_{pid}: question file pid_{pid}.txt not found in {QUESTION_DIR}")
            fail += 1
            continue

        prompt_text = prompt_file.read_text(encoding="utf-8")
        if SEPARATOR not in prompt_text:
            ok += 1  # raw-format file in a mixed dir — treat as not applicable
            continue
        actual_q    = extract_question_section(prompt_text)
        expected_q  = expected_q_file.read_text(encoding="utf-8")

        if actual_q.strip() != expected_q.strip():
            # Try to identify which wrong file was used
            wrong_file = "unknown"
            for candidate in QUESTION_DIR.glob(f"pid_{pid}*.txt"):
                if candidate.stem != f"pid_{pid}":
                    cand_text = candidate.read_text(encoding="utf-8")
                    if cand_text.strip() == actual_q.strip():
                        wrong_file = candidate.name
                        break
            errors.append(
                f"  pid_{pid}: WRONG question file used "
                f"(got content of '{wrong_file}', expected 'pid_{pid}.txt')"
            )
            fail += 1
        else:
            ok += 1

    return ok, fail, errors, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None,
                        help="Single input dir name to check (default: all text_simulation_input_* dirs)")
    args = parser.parse_args()

    if args.dir:
        dirs = [INPUT_BASE / args.dir]
    else:
        dirs = sorted(INPUT_BASE.glob("text_simulation_input_*"))
        dirs = [d for d in dirs if d.is_dir()]

    total_ok = total_fail = 0
    failed_dirs = []

    for d in dirs:
        ok, fail, errors, skipped_format = verify_dir(d)
        if skipped_format:
            print(f"  {'SKIP (no sep)':18s}  {d.name}")
            continue
        total_ok += ok
        total_fail += fail
        status = "OK" if fail == 0 else f"FAIL ({fail} wrong)"
        print(f"  {status:18s}  {d.name}  ({ok} correct)")
        if errors:
            for e in errors[:3]:
                print(e)
            if len(errors) > 3:
                print(f"    ... and {len(errors)-3} more")
            failed_dirs.append(d.name)

    print(f"\n{'='*60}")
    print(f"Total: {total_ok} correct, {total_fail} wrong")
    if failed_dirs:
        print(f"Failed dirs ({len(failed_dirs)}):")
        for d in failed_dirs:
            print(f"  {d}")
    else:
        print("All input directories verified OK.")


if __name__ == "__main__":
    main()
