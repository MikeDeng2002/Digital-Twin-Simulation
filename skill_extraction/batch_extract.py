"""
batch_extract.py — Run skill extraction for multiple personas in parallel.

Usage (run from Digital-Twin-Simulation/):
    # Extract all personas (all 3 versions)
    poetry run python skill_extraction/batch_extract.py

    # Test run: first 5 personas only
    poetry run python skill_extraction/batch_extract.py --max_personas 5

    # Only v1_direct for first 10 personas
    poetry run python skill_extraction/batch_extract.py --max_personas 10 --version v1_direct

    # Resume (skip already-completed, only run missing)
    poetry run python skill_extraction/batch_extract.py --resume
"""

import os
import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from extract_skills import extract_skills_for_persona, VERSIONS, PERSONA_DIR, OUTPUT_BASE

load_dotenv()


def get_all_pids() -> list[str]:
    """Discover all persona IDs from text_personas directory."""
    pids = []
    for f in sorted(PERSONA_DIR.glob("pid_*.txt")):
        m = re.search(r"pid_(\d+)", f.name)
        if m:
            pids.append(m.group(1))
    return sorted(pids, key=lambda x: int(x))


def is_complete(pid: str, versions: list) -> bool:
    """Return True if all skill files already exist for this persona."""
    for version in versions:
        out_dir = OUTPUT_BASE / f"pid_{pid}" / version
        for fname in ["background.txt", "tools.txt", "decision_procedure.txt"]:
            if not (out_dir / fname).exists():
                return False
    return True


def run_one(pid: str, versions: list, force: bool) -> tuple[str, bool, str]:
    """Worker function — returns (pid, success, error_message)."""
    try:
        extract_skills_for_persona(pid, versions, force=force)
        return pid, True, ""
    except Exception as e:
        return pid, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch skill extraction for all personas.")
    parser.add_argument("--max_personas", type=int, default=None,
                        help="Limit to first N personas (default: all)")
    parser.add_argument("--version", choices=VERSIONS + ["all"], default="all",
                        help="Which version(s) to generate (default: all)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel API threads (default: 5)")
    parser.add_argument("--pids", type=str, default=None,
                        help="Comma-separated list of specific PIDs (e.g. 1,2,3)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip personas that are already complete")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if output files already exist")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")

    versions_to_run = VERSIONS if args.version == "all" else [args.version]

    if args.pids:
        all_pids = [p.strip() for p in args.pids.split(",")]
        print(f"Using {len(all_pids)} specified PIDs: {all_pids}")
    else:
        all_pids = get_all_pids()
        print(f"Found {len(all_pids)} personas in {PERSONA_DIR}")

    if args.max_personas:
        all_pids = all_pids[: args.max_personas]
        print(f"Limited to first {args.max_personas} personas")

    if args.resume:
        pending = [p for p in all_pids if not is_complete(p, versions_to_run)]
        skipped = len(all_pids) - len(pending)
        print(f"Resume mode: {skipped} already complete, {len(pending)} to process")
        all_pids = pending

    if not all_pids:
        print("Nothing to do.")
        return

    print(f"Running {len(all_pids)} personas x {len(versions_to_run)} versions "
          f"with {args.workers} workers...\n")

    success_count = 0
    fail_count = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, pid, versions_to_run, args.force): pid
            for pid in all_pids
        }
        for future in as_completed(futures):
            pid, ok, err = future.result()
            if ok:
                success_count += 1
                print(f"[{success_count + fail_count}/{len(all_pids)}] pid_{pid} OK")
            else:
                fail_count += 1
                failures.append((pid, err))
                print(f"[{success_count + fail_count}/{len(all_pids)}] pid_{pid} FAILED: {err}")

    print(f"\nDone. {success_count} succeeded, {fail_count} failed.")
    if failures:
        print("Failed personas:")
        for pid, err in failures:
            print(f"  pid_{pid}: {err}")


if __name__ == "__main__":
    main()
