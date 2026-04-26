"""
batch_extract_v2.py — Extract v2 skill profiles using the OpenAI Batch API.

Builds ONE batch job containing all requests for all pids × versions × files,
submits it to the OpenAI Batch API, polls until complete, then writes output.

Why 9 API calls per persona:
  3 files × 3 versions = 9
  - background.txt         × (v1_direct, v2_inferred, v3_maximum)
  - decision_procedure.txt × (v1_direct, v2_inferred, v3_maximum)
  - evaluation_profile.txt × (v1_direct, v2_inferred, v3_maximum)
  Each combination uses a completely different prompt (different
  inference strictness), so they cannot be merged into one call.

For 20 personas: 20 × 9 = 180 batch requests total.
Batch API gives 50% cost savings and no rate-limit throttling.

Usage (from Digital-Twin-Simulation/):
    # All pids in skills_v2/ dir, all versions
    python skill_extraction/batch_extract_v2.py

    # Specific pids
    python skill_extraction/batch_extract_v2.py --pids 1,2,3

    # Only one version
    python skill_extraction/batch_extract_v2.py --version v1_direct

    # Skip already-completed pids
    python skill_extraction/batch_extract_v2.py --resume

    # Force regenerate
    python skill_extraction/batch_extract_v2.py --force
"""

import os
import io
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Import prompts and config from extract_skills_v2
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extract_skills_v2 import (
    MODEL, VERSIONS, PERSONA_DIR, OUTPUT_BASE, MAX_TOKENS,
    FILES, BACKGROUND_PROMPTS, DECISION_PROMPTS, EVALUATION_PROMPTS,
    load_transcript, parse_section,
)

load_dotenv()

POLL_INTERVAL = 30   # seconds between status polls
MAX_WAIT      = 86400  # 24h max

DEFAULT_PID_RANGE = list(range(1, 21))   # pids 1-20
V2_FILES = ["background.txt", "decision_procedure.txt", "evaluation_profile.txt"]


# ── Completion check ──────────────────────────────────────────────────────────

def is_complete(pid: str, versions: list) -> bool:
    for version in versions:
        out_dir = OUTPUT_BASE / f"pid_{pid}" / version
        if not all((out_dir / f).exists() for f in V2_FILES):
            return False
    return True


# ── Build JSONL ───────────────────────────────────────────────────────────────

def build_custom_id(pid: str, version: str, file_key: str) -> str:
    return f"pid_{pid}__{version}__{file_key}"


def parse_custom_id(custom_id: str) -> tuple[str, str, str]:
    parts = custom_id.split("__")
    # pid may be pid_1 format
    pid     = parts[0].replace("pid_", "")
    version = parts[1]
    file_key = parts[2]
    return pid, version, file_key


def build_batch_jsonl(pids: list[str], versions: list[str], force: bool) -> tuple[list[str], bytes]:
    """Build JSONL bytes for all pending (pid, version, file) combos.
    Returns (custom_ids_included, jsonl_bytes).
    """
    lines = []
    custom_ids = []

    for pid in pids:
        try:
            transcript = load_transcript(pid)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping pid_{pid}")
            continue

        for version in versions:
            output_dir = OUTPUT_BASE / f"pid_{pid}" / version
            max_tok    = MAX_TOKENS[version]

            for file_key, filename, prompt_dict, _marker in FILES:
                out_path = output_dir / filename
                if not force and out_path.exists():
                    continue   # already done, skip

                prompt = prompt_dict[version].format(transcript=transcript)
                custom_id = build_custom_id(pid, version, file_key)

                request = {
                    "custom_id": custom_id,
                    "method":    "POST",
                    "url":       "/v1/chat/completions",
                    "body": {
                        "model":       MODEL,
                        "messages":    [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens":  max_tok,
                    },
                }
                lines.append(json.dumps(request))
                custom_ids.append(custom_id)

    jsonl_bytes = "\n".join(lines).encode("utf-8")
    return custom_ids, jsonl_bytes


# ── Poll until done ───────────────────────────────────────────────────────────

def poll_batch(client: OpenAI, batch_id: str) -> object:
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        done   = counts.completed + counts.failed if counts else 0
        total  = counts.total if counts else "?"
        print(f"  [{batch_id}] status={status}  completed={done}/{total}  "
              f"elapsed={int(time.time()-start)}s")

        if status in ("completed", "failed", "cancelled", "expired"):
            return batch

        if time.time() - start > MAX_WAIT:
            raise TimeoutError(f"Batch {batch_id} did not finish within {MAX_WAIT}s")

        time.sleep(POLL_INTERVAL)


# ── Parse and write results ───────────────────────────────────────────────────

def write_results(client: OpenAI, batch, force: bool) -> tuple[int, int, list]:
    """Download output file, parse each line, write skill files.
    Returns (ok, failed, retry_needed).
    retry_needed = list of (custom_id, doubled_max_tokens) for missing ## Summary.
    """
    if not batch.output_file_id:
        print("  No output_file_id — batch may have failed entirely.")
        return 0, 0, []

    raw = client.files.content(batch.output_file_id).read()
    lines = raw.decode("utf-8").strip().split("\n")

    ok = 0
    failed = 0
    retry_needed = []

    for line in lines:
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result["custom_id"]
        pid, version, file_key = parse_custom_id(custom_id)

        # Find the matching FILES entry
        file_entry = next((f for f in FILES if f[0] == file_key), None)
        if file_entry is None:
            print(f"  Unknown file_key {file_key!r} in {custom_id}")
            failed += 1
            continue

        _fk, filename, _pd, marker = file_entry
        output_dir = OUTPUT_BASE / f"pid_{pid}" / version
        out_path   = output_dir / filename

        # Check for API-level errors
        if result.get("error"):
            print(f"  ERROR {custom_id}: {result['error']}")
            failed += 1
            continue

        body = result.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            print(f"  EMPTY response for {custom_id}")
            failed += 1
            continue

        raw_text = choices[0]["message"]["content"].strip()
        content  = parse_section(raw_text, marker)

        # Queue retry if Summary section is missing
        if "## Summary" not in content:
            doubled = min(MAX_TOKENS[version] * 2, 4000)
            retry_needed.append((custom_id, pid, version, file_key, filename, marker, doubled))
            print(f"  NEEDS RETRY (no ## Summary): {custom_id}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        ok += 1

    # Also parse error file if present
    if batch.error_file_id:
        err_raw = client.files.content(batch.error_file_id).read()
        for line in err_raw.decode("utf-8").strip().split("\n"):
            if line.strip():
                print(f"  BATCH ERROR: {line}")
                failed += 1

    return ok, failed, retry_needed


def retry_batch(client: OpenAI, retry_items: list) -> int:
    """Submit a second batch for items missing ## Summary (doubled tokens)."""
    if not retry_items:
        return 0

    print(f"\nRetrying {len(retry_items)} items with doubled max_tokens...")

    # Re-load transcripts (keyed by pid)
    transcripts: dict[str, str] = {}
    lines = []
    for custom_id, pid, version, file_key, filename, marker, doubled_max in retry_items:
        if pid not in transcripts:
            try:
                transcripts[pid] = load_transcript(pid)
            except FileNotFoundError:
                print(f"  SKIP retry for {custom_id}: transcript not found")
                continue

        file_entry = next(f for f in FILES if f[0] == file_key)
        prompt = file_entry[2][version].format(transcript=transcripts[pid])
        request = {
            "custom_id": f"retry__{custom_id}",
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":       MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens":  doubled_max,
            },
        }
        lines.append(json.dumps(request))

    jsonl_bytes = "\n".join(lines).encode("utf-8")
    file_obj = client.files.create(file=("retry_skills_v2.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"), purpose="batch")
    batch = client.batches.create(input_file_id=file_obj.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    print(f"  Retry batch submitted: {batch.id}")
    batch = poll_batch(client, batch.id)

    if not batch.output_file_id:
        print("  Retry batch produced no output.")
        return 0

    raw = client.files.content(batch.output_file_id).read()
    ok = 0
    for line in raw.decode("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        retry_custom_id = result["custom_id"]
        original_id = retry_custom_id.removeprefix("retry__")
        pid, version, file_key = parse_custom_id(original_id)

        file_entry = next((f for f in FILES if f[0] == file_key), None)
        if not file_entry:
            continue
        _fk, filename, _pd, marker = file_entry

        body    = result.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            print(f"  Retry EMPTY: {original_id}")
            continue

        raw_text = choices[0]["message"]["content"].strip()
        content  = parse_section(raw_text, marker)

        output_dir = OUTPUT_BASE / f"pid_{pid}" / version
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / filename).write_text(content, encoding="utf-8")
        ok += 1

    print(f"  Retry batch: {ok}/{len(retry_items)} saved.")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-API skill extraction for v2 profiles (pids 1-20 by default)."
    )
    parser.add_argument("--pids",    default=None,
                        help="Comma-separated pid list (default: 1-20)")
    parser.add_argument("--version", choices=VERSIONS + ["all"], default="all",
                        help="Version(s) to extract (default: all)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip pids where all output files already exist")
    parser.add_argument("--force",   action="store_true",
                        help="Regenerate even if output files already exist")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")
    client = OpenAI(api_key=api_key)

    # Resolve pids
    if args.pids:
        all_pids = [p.strip() for p in args.pids.split(",")]
    else:
        all_pids = [str(i) for i in DEFAULT_PID_RANGE]

    versions = VERSIONS if args.version == "all" else [args.version]

    # Resume: skip already-complete pids
    if args.resume:
        pending = [p for p in all_pids if not is_complete(p, versions)]
        skipped = len(all_pids) - len(pending)
        print(f"Resume: {skipped} already complete, {len(pending)} to process")
        all_pids = pending

    if not all_pids:
        print("Nothing to do.")
        return

    # ── Stage 1: build JSONL ─────────────────────────────────────────────────
    print(f"\nBuilding batch requests for {len(all_pids)} pids × {len(versions)} versions × 3 files...")
    custom_ids, jsonl_bytes = build_batch_jsonl(all_pids, versions, force=args.force)

    if not custom_ids:
        print("All requested files already exist. Nothing to submit.")
        return

    n_requests = len(custom_ids)
    print(f"Total requests in batch: {n_requests} "
          f"({len(all_pids)} pids × {len(versions)} versions × 3 files)")

    # ── Stage 2: upload + submit ─────────────────────────────────────────────
    print(f"\nUploading JSONL ({len(jsonl_bytes)/1024:.1f} KB)...")
    file_obj = client.files.create(
        file=("skills_v2_extraction.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch"
    )
    print(f"Uploaded file: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id}  (polling every {POLL_INTERVAL}s...)\n")

    # ── Stage 3: poll ────────────────────────────────────────────────────────
    batch = poll_batch(client, batch.id)
    print(f"\nBatch final status: {batch.status}")

    if batch.status != "completed":
        print(f"Batch did not complete successfully: {batch.status}")
        return

    # ── Stage 4: write results ───────────────────────────────────────────────
    print("\nParsing and writing output files...")
    ok, failed, retry_items = write_results(client, batch, force=args.force)
    print(f"Written: {ok}  Failed: {failed}  Needs retry: {len(retry_items)}")

    # ── Stage 5: retry items missing ## Summary ──────────────────────────────
    retry_ok = retry_batch(client, retry_items)

    total_ok = ok + retry_ok
    print(f"\n{'='*60}")
    print(f"Extraction complete.")
    print(f"  Files written:  {total_ok}")
    print(f"  Hard failures:  {failed}")
    print(f"  Output root:    {OUTPUT_BASE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
