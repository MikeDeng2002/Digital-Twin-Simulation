"""
run_pipeline_v2.py — Fully automatic end-to-end pipeline for the v2 skill experiment.

Pipeline stages:
  1. Extract v2 skills for pids 1-20  (batch_extract_v2.py)
  2. Build LLM input prompts            (text_simulation/create_skill_v2_inputs.py)
  3. Run simulation via Batch API       (text_simulation/run_batch_experiment_parallel.py)
  4. Postprocess responses              (text_simulation/run_postprocess.py)
  5. Evaluate accuracy                  (evaluation/eval_temp0_suite.py)

All stages run automatically with no user prompts.
Resume-safe: stages that detect existing output will skip already-done work.

Usage (from Digital-Twin-Simulation/):
    # Full pipeline
    python run_pipeline_v2.py

    # Start from a specific stage (1–5)
    python run_pipeline_v2.py --start_stage 3

    # Force-regenerate skill extraction even if files exist
    python run_pipeline_v2.py --force_extract
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SUITE = "nano_v2_temp0"
PIDS  = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"


def run(cmd: list[str], label: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming stdout/stderr live."""
    print(f"\n{'='*70}")
    print(f"STAGE: {label}")
    print(f"CMD:   {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, check=check)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status}  ({elapsed:.0f}s)")
    return result


def stage_1_extract(force: bool):
    """Extract v2 skill profiles for pids 1-20 via OpenAI Batch API.

    Submits one batch job: 20 pids × 3 versions × 3 files = 180 requests.
    Polls until complete, then writes all output files.
    """
    cmd = [
        sys.executable,
        "skill_extraction/batch_extract_v2.py",
        "--pids", PIDS,
    ]
    if not force:
        cmd.append("--resume")
    else:
        cmd.append("--force")
    run(cmd, "1 — Extract v2 skills (OpenAI Batch API, gpt-4.1-mini, 180 requests)")


def stage_2_build_inputs():
    """Assemble skill files into combined LLM prompt files for all 3 versions."""
    cmd = [
        sys.executable,
        "text_simulation/create_skill_v2_inputs.py",
        "--pids", PIDS,
    ]
    run(cmd, "2 — Build simulation inputs (3 versions × 20 personas)")


def stage_3_simulate():
    """Submit all 12 configs to OpenAI Batch API and wait for completion.

    Uses run_batch_experiment_parallel.py which:
      - Uploads JSONL files to the Batch API
      - Polls until all batches are complete
      - Downloads results automatically
      - Retries incomplete/truncated responses with doubled max_tokens
    """
    cmd = [
        sys.executable,
        "text_simulation/run_batch_experiment_parallel.py",
        "--suite", SUITE,
    ]
    run(cmd, f"3 — Run simulation via OpenAI Batch API ({SUITE}, 12 configs × 20 personas)")


def stage_4_postprocess():
    """Parse raw LLM responses into structured answer JSON files."""
    cmd = [
        sys.executable,
        "text_simulation/run_postprocess.py",
        "--suite", SUITE,
    ]
    run(cmd, "4 — Postprocess responses")


def stage_5_evaluate():
    """Run MAD accuracy evaluation for all 12 configs."""
    cmd = [
        sys.executable,
        "evaluation/eval_temp0_suite.py",
        "--suite", SUITE,
    ]
    run(cmd, "5 — Evaluate accuracy (LLM vs wave1-3)")


def main():
    parser = argparse.ArgumentParser(
        description="Full v2 skill extraction + simulation + evaluation pipeline."
    )
    parser.add_argument(
        "--start_stage", type=int, default=1, choices=range(1, 6),
        help="Start from this stage (1=extract, 2=inputs, 3=simulate, 4=postprocess, 5=eval)"
    )
    parser.add_argument(
        "--force_extract", action="store_true",
        help="Re-run skill extraction even if output files already exist (stage 1)"
    )
    parser.add_argument(
        "--only_stage", type=int, default=None, choices=range(1, 6),
        help="Run only this stage (overrides --start_stage)"
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    start = args.only_stage if args.only_stage else args.start_stage
    end   = (args.only_stage + 1) if args.only_stage else 6

    stages = {
        1: lambda: stage_1_extract(args.force_extract),
        2: stage_2_build_inputs,
        3: stage_3_simulate,
        4: stage_4_postprocess,
        5: stage_5_evaluate,
    }

    t_total = time.time()
    for stage_num in range(start, end):
        stages[stage_num]()

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE  (total {elapsed/60:.1f} min)")
    print(f"Results: text_simulation/text_simulation_output_nano_v2_temp0/")
    print(f"Evaluation: evaluation/nano_v2_temp0_20p_results.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
