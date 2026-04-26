"""
batch_extract_skills_v4.py — Batch API extraction of v4 skill profiles.

Submits all requests as a single OpenAI Batch job (50% cost vs real-time).
Each request produces ONE file (background / decision_profile / evaluation_profile)
for ONE persona.

Total requests = n_pids × 3 files

Output:
  text_simulation/skills_v4/pid_{pid}/
  ├── background.txt          (max_tokens=2000)
  ├── decision_profile.txt    (max_tokens=1800)
  └── evaluation_profile.txt  (max_tokens=2200)

Usage (from Digital-Twin-Simulation/):
    # Extract pids 1-20 (all 3 files)
    python skill_extraction/batch_extract_skills_v4.py --pids 1-20

    # Extract specific pids, single file type
    python skill_extraction/batch_extract_skills_v4.py --pids 1,5,10 --file background

    # Force re-extract even if files exist
    python skill_extraction/batch_extract_skills_v4.py --pids 1-20 --force

    # Resume a running or completed batch
    python skill_extraction/batch_extract_skills_v4.py --resume batch_XXXXX --pids 1-20

    # Dry run: show what would be submitted
    python skill_extraction/batch_extract_skills_v4.py --pids 1-20 --dry-run
"""

import os, io, json, re, time, argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL        = "gpt-4.1-mini-2025-04-14"
POLL_INTERVAL = 30    # seconds between status polls
MAX_WAIT      = 86400  # 24h timeout

# ── Import prompts and utilities from extract_skills_v4 ──────────────────────
import sys
sys.path.insert(0, str(Path("skill_extraction")))
from extract_skills_v4 import (
    FILES, MAX_TOKENS,
    load_transcript, parse_section,
    OUTPUT_BASE,
)

# Validate FILES structure: (file_key, filename, prompt_template, marker)
assert len(FILES) == 3, f"Expected 3 file specs, got {len(FILES)}"


# ── JSONL builder ─────────────────────────────────────────────────────────────

def build_jsonl(
    pids: list[str],
    files_filter: list[str] | None,
    force: bool,
) -> tuple[bytes, int]:
    """Build JSONL payload. Returns (bytes, n_requests)."""
    lines   = []
    skipped = 0
    missing_transcripts = []

    for pid in pids:
        try:
            transcript = load_transcript(pid)
        except FileNotFoundError:
            missing_transcripts.append(pid)
            continue

        out_dir = OUTPUT_BASE / f"pid_{pid}"

        for file_key, filename, prompt_template, marker in FILES:
            if files_filter and file_key not in files_filter:
                continue

            out_path = out_dir / filename
            if not force and out_path.exists():
                skipped += 1
                continue

            custom_id = f"pid_{pid}__{file_key}"
            prompt    = prompt_template.format(transcript=transcript)
            max_tok   = MAX_TOKENS[file_key]

            lines.append(json.dumps({
                "custom_id": custom_id,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":       MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens":  max_tok,
                },
            }))

    if missing_transcripts:
        print(f"  WARNING: no transcript found for pids: {missing_transcripts}")

    print(f"  Requests to submit : {len(lines)}")
    print(f"  Skipped (exist)    : {skipped}")

    return "\n".join(lines).encode("utf-8"), len(lines)


# ── Batch lifecycle ───────────────────────────────────────────────────────────

def submit_batch(client: OpenAI, jsonl_bytes: bytes, metadata: dict) -> str:
    print("Uploading batch JSONL...")
    batch_file = client.files.create(
        file=("batch_v4_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch",
    )
    print(f"  Uploaded file: {batch_file.id}")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata,
    )
    print(f"  Batch job  : {batch.id}")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str) -> object:
    start = time.time()
    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        c = counts.completed if counts else "?"
        t = counts.total     if counts else "?"
        f = counts.failed    if counts else "?"
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {batch.status}  "
              f"completed={c}/{t}  failed={f}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        if time.time() - start > MAX_WAIT:
            raise TimeoutError(f"Batch {batch_id} exceeded {MAX_WAIT}s timeout")
        time.sleep(POLL_INTERVAL)


def save_results(
    client: OpenAI,
    batch,
    truncation_retry: bool = True,
) -> tuple[int, int]:
    """Download batch output, parse, and save files. Returns (saved, errors)."""
    if not batch.output_file_id:
        print("  No output file — batch may have failed entirely.")
        return 0, 0

    raw_content = client.files.content(batch.output_file_id).text
    saved = errors = truncated = 0

    for line in raw_content.strip().split("\n"):
        if not line.strip():
            continue
        obj       = json.loads(line)
        custom_id = obj["custom_id"]   # e.g. "pid_1__background"

        # API-level error
        if obj.get("error"):
            print(f"  API ERROR [{custom_id}]: {obj['error']}")
            errors += 1
            continue

        # Parse custom_id → pid, file_key
        parts = custom_id.split("__")
        if len(parts) != 2:
            print(f"  BAD custom_id: {custom_id}")
            errors += 1
            continue
        pid_str, file_key = parts
        pid = pid_str.replace("pid_", "")

        # Find matching file spec
        file_spec = next((f for f in FILES if f[0] == file_key), None)
        if file_spec is None:
            print(f"  UNKNOWN file_key [{custom_id}]: {file_key}")
            errors += 1
            continue

        _, filename, _, marker = file_spec

        # Extract response content
        response_body = obj.get("response", {}).get("body", {})
        choices       = response_body.get("choices", [])
        if not choices:
            print(f"  EMPTY response [{custom_id}]")
            errors += 1
            continue

        raw     = choices[0].get("message", {}).get("content", "")
        content = parse_section(raw, marker)

        # Truncation detection: ## Summary missing
        if "## Summary" not in content:
            truncated += 1
            print(f"  TRUNCATED [{custom_id}]: ## Summary missing "
                  f"(finish_reason={choices[0].get('finish_reason', '?')})")
            # Cannot retry in batch mode — save what we have and flag
            content = content + "\n\n<!-- WARNING: response may be truncated; ## Summary section not found -->"

        # Save
        out_dir  = OUTPUT_BASE / f"pid_{pid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_text(content, encoding="utf-8")
        saved += 1

    print(f"\n  Saved     : {saved}")
    print(f"  Errors    : {errors}")
    print(f"  Truncated : {truncated}  (## Summary missing)")
    return saved, errors


# ── Error file handler ────────────────────────────────────────────────────────

def print_error_file(client: OpenAI, batch):
    if not batch.error_file_id:
        return
    errors = client.files.content(batch.error_file_id).text
    print("\nError file contents (first 2000 chars):")
    print(errors[:2000])


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_pid_arg(s: str) -> list[str]:
    """Parse '1-20' or '1,2,5' into a list of pid strings."""
    s = s.strip()
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return [str(i) for i in range(int(start), int(end) + 1)]
    return [p.strip() for p in s.split(",")]


def discover_pids() -> list[str]:
    """Find all available pids from text_personas directory."""
    persona_dir = Path("text_simulation/text_personas")
    pids = set()
    for f in persona_dir.glob("pid_*.txt"):
        m = re.search(r"pid_(\d+)", f.name)
        if m:
            pids.add(m.group(1))
    return sorted(pids, key=int)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-extract v4 skill profiles using OpenAI Batch API.")
    parser.add_argument("--pids",    default=None,
                        help="PID range '1-20' or list '1,2,5' (default: all available)")
    parser.add_argument("--file",    default=None,
                        choices=["background", "decision_profile", "evaluation_profile"],
                        help="Extract only this file type (default: all three)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-extract even if output files already exist")
    parser.add_argument("--resume",  default=None,
                        help="Batch ID to resume (skip submission, go straight to poll+save)")
    parser.add_argument("--no-poll", action="store_true",
                        help="Submit batch and exit without polling (for fire-and-forget)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be submitted without making API calls")
    args = parser.parse_args()

    # Resolve pids
    if args.pids:
        pids = parse_pid_arg(args.pids)
    else:
        pids = discover_pids()
        print(f"Discovered {len(pids)} personas in text_personas/")

    files_filter = [args.file] if args.file else None

    print(f"Personas     : {len(pids)}  ({pids[0]}..{pids[-1]})")
    print(f"Files        : {files_filter or 'all three'}")
    print(f"Output base  : {OUTPUT_BASE}")
    print(f"Model        : {MODEL}")

    if args.dry_run:
        _, n = build_jsonl(pids, files_filter, force=args.force)
        print(f"\nDry run: would submit {n} requests.")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if args.resume:
        batch_id = args.resume
        print(f"\nResuming batch: {batch_id}")
    else:
        # Build JSONL
        print("\nBuilding JSONL...")
        jsonl_bytes, n_requests = build_jsonl(pids, files_filter, force=args.force)

        if n_requests == 0:
            print("Nothing to submit — all files already exist. Use --force to re-extract.")
            return

        # Submit
        metadata = {
            "script":  "batch_extract_skills_v4",
            "pids":    args.pids or "all",
            "files":   args.file or "all",
        }
        batch_id = submit_batch(client, jsonl_bytes, metadata)

        if args.no_poll:
            print(f"\nBatch submitted: {batch_id}")
            print("Run with --resume to retrieve results when complete.")
            return

    # Poll
    print(f"\nPolling batch {batch_id} every {POLL_INTERVAL}s...")
    batch = poll_batch(client, batch_id)
    print(f"\nBatch finished with status: {batch.status}")

    if batch.status == "failed":
        print_error_file(client, batch)
        return

    if batch.status != "completed":
        print(f"Unexpected terminal status: {batch.status}")
        return

    # Save results
    print("\nSaving results...")
    saved, errors = save_results(client, batch)

    print(f"\nDone. {saved} files written, {errors} errors.")


if __name__ == "__main__":
    main()
