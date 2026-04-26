"""
batch_extract_skills_v2.py — Batch API extraction of skills_v2 profiles.

Submits all requests as a single OpenAI Batch job (50% cost vs real-time).
Each request: one file (background / decision_procedure / evaluation_profile)
for one persona at one version.

Total requests = n_pids × n_versions × 3 files

Usage (from Digital-Twin-Simulation/):
    # pids 21-50, v2_inferred + v3_maximum only
    python skill_extraction/batch_extract_skills_v2.py --pids 21-50 --versions v2_inferred,v3_maximum

    # Check/resume a running batch
    python skill_extraction/batch_extract_skills_v2.py --resume batch_XXXXX
"""

import os, io, json, re, time, argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL      = "gpt-4.1-mini-2025-04-14"
VERSIONS   = ["v1_direct", "v2_inferred", "v3_maximum"]
PERSONA_DIR = Path("text_simulation/text_personas")
OUTPUT_BASE = Path("text_simulation/skills_v2")

MAX_TOKENS = {"v1_direct": 1500, "v2_inferred": 2000, "v3_maximum": 2500}
POLL_INTERVAL = 30
MAX_WAIT      = 86400

# ── Import prompts from extract_skills_v2.py ─────────────────────────────────
import sys
sys.path.insert(0, str(Path("skill_extraction")))
from extract_skills_v2 import (
    BACKGROUND_PROMPTS, DECISION_PROMPTS, EVALUATION_PROMPTS,
    parse_section, load_transcript
)

FILES = [
    ("background",         "background.txt",        BACKGROUND_PROMPTS,  "---BACKGROUND---"),
    ("decision_procedure", "decision_procedure.txt", DECISION_PROMPTS,    "---DECISION_PROCEDURE---"),
    ("evaluation_profile", "evaluation_profile.txt", EVALUATION_PROMPTS,  "---EVALUATION_PROFILE---"),
]


def build_jsonl(pids: list[str], versions: list[str], force: bool = False) -> bytes:
    lines = []
    skipped = 0
    for pid in pids:
        transcript = load_transcript(pid)
        for version in versions:
            out_dir = OUTPUT_BASE / f"pid_{pid}" / version
            for file_key, filename, prompt_dict, marker in FILES:
                out_path = out_dir / filename
                if not force and out_path.exists():
                    skipped += 1
                    continue
                custom_id = f"pid_{pid}__{version}__{file_key}"
                prompt    = prompt_dict[version].format(transcript=transcript)
                body = {
                    "model":       MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens":  MAX_TOKENS[version],
                }
                lines.append(json.dumps({
                    "custom_id": custom_id,
                    "method":    "POST",
                    "url":       "/v1/chat/completions",
                    "body":      body,
                }))

    print(f"  Requests to submit: {len(lines)}  (skipped {skipped} already done)")
    return "\n".join(lines).encode("utf-8")


def poll_batch(client: OpenAI, batch_id: str):
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        c = counts.completed if counts else "?"
        t = counts.total     if counts else "?"
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {batch.status}  {c}/{t}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        if time.time() - start > MAX_WAIT:
            raise TimeoutError(f"Batch {batch_id} timed out")
        time.sleep(POLL_INTERVAL)


def save_results(client: OpenAI, batch, pids: list[str], versions: list[str]):
    content = client.files.content(batch.output_file_id).text
    saved = errors = 0

    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]   # pid_21__v2_inferred__background

        if obj.get("error"):
            print(f"  ERROR {custom_id}: {obj['error']}")
            errors += 1
            continue

        # Parse custom_id
        parts = custom_id.split("__")
        pid, version, file_key = parts[0].replace("pid_",""), parts[1], parts[2]

        # Find matching file spec
        file_spec = next((f for f in FILES if f[0] == file_key), None)
        if not file_spec:
            continue
        _, filename, _, marker = file_spec

        # Extract text
        response_body = obj.get("response", {}).get("body", obj.get("response", {}))
        choices = response_body.get("choices", [])
        if not choices:
            print(f"  EMPTY {custom_id}")
            errors += 1
            continue
        raw     = choices[0].get("message", {}).get("content", "")
        content_text = parse_section(raw, marker)

        # Retry detection: Summary missing → log warning (can't retry in batch mode)
        if "## Summary" not in content_text:
            print(f"  WARNING {custom_id}: Summary section missing (may be truncated)")

        # Save
        out_dir = OUTPUT_BASE / f"pid_{pid}" / version
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / filename).write_text(content_text, encoding="utf-8")
        saved += 1

    print(f"\n  Saved: {saved}  Errors: {errors}")
    return saved, errors


def parse_pid_range(s: str) -> list[str]:
    """Parse '21-50' or '21,22,23' into list of pid strings."""
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return [str(i) for i in range(int(start), int(end) + 1)]
    return [p.strip() for p in s.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids",     default="21-50",
                        help="PID range e.g. '21-50' or '21,22,23'")
    parser.add_argument("--versions", default="v2_inferred,v3_maximum",
                        help="Comma-separated versions (default: v2_inferred,v3_maximum)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-extract even if output already exists")
    parser.add_argument("--resume",   default=None,
                        help="Batch ID to resume (skip submission, go straight to poll+save)")
    args = parser.parse_args()

    pids     = parse_pid_range(args.pids)
    versions = [v.strip() for v in args.versions.split(",")]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if args.resume:
        batch_id = args.resume
        print(f"Resuming batch {batch_id}...")
    else:
        # Build and submit
        print(f"Building JSONL for {len(pids)} pids × {versions}...")
        jsonl = build_jsonl(pids, versions, force=args.force)

        if not jsonl.strip():
            print("Nothing to submit — all files already exist. Use --force to re-extract.")
            return

        print("Uploading batch file...")
        batch_file = client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl), "application/jsonl"),
            purpose="batch",
        )
        print(f"  File: {batch_file.id}")

        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"script": "batch_extract_skills_v2", "pids": args.pids, "versions": args.versions},
        )
        batch_id = batch.id
        print(f"  Batch: {batch_id}\n")

    # Poll
    print("Polling...")
    batch = poll_batch(client, batch_id)

    if batch.status != "completed":
        print(f"Batch ended with status: {batch.status}")
        return

    # Save
    print("\nSaving results...")
    save_results(client, batch, pids, versions)
    print("Done.")


if __name__ == "__main__":
    main()
