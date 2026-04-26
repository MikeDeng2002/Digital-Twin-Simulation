"""
run_batch_experiment_parallel.py — Submit ALL configs to OpenAI Batch API at once.

Submits all configs in parallel (upload + create batch), then polls all jobs
simultaneously until complete, then saves all results.

Usage (from Digital-Twin-Simulation/):
    # Nano temp=0.0 (40 configs)
    python text_simulation/run_batch_experiment_parallel.py --suite nano_temp0

    # Mini temp=0.0 (40 configs)
    python text_simulation/run_batch_experiment_parallel.py --suite mini_temp0

    # Single setting filter
    python text_simulation/run_batch_experiment_parallel.py --suite nano_temp0 --setting skill_v3

    # Single config filter
    python text_simulation/run_batch_experiment_parallel.py --suite nano_temp0 --setting skill_v3 --reasoning high
"""

import os
import io
import json
import time
import re
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

POLL_INTERVAL = 20   # seconds between status polls
MAX_WAIT      = 86400  # 24h max

SUITES = {
    "nano_temp0": {
        "config_dir": "text_simulation/configs/nano_temp0",
        "log_dir":    "text_simulation/logs/nano_batch_temp0",
    },
    "mini_temp0": {
        "config_dir": "text_simulation/configs/mini_temp0",
        "log_dir":    "text_simulation/logs/mini_batch_temp0",
    },
    "nano": {
        "config_dir": "text_simulation/configs/nano",
        "log_dir":    "text_simulation/logs/nano_batch",
    },
    "mini": {
        "config_dir": "text_simulation/configs/mini",
        "log_dir":    "text_simulation/logs/mini_batch",
    },
    "nano_v2_temp0": {
        "config_dir": "text_simulation/configs/nano_v2_temp0",
        "log_dir":    "text_simulation/logs/nano_v2_batch_temp0",
    },
    "nano_v2_ablation_temp0": {
        "config_dir": "text_simulation/configs/nano_v2_ablation_temp0",
        "log_dir":    "text_simulation/logs/nano_v2_ablation_batch_temp0",
    },
    "fix_a_diag": {
        "config_dir": "text_simulation/configs/fix_a_diag",
        "log_dir":    "text_simulation/logs/fix_a_diag",
    },
    "minimal_diag": {
        "config_dir": "text_simulation/configs/minimal_diag",
        "log_dir":    "text_simulation/logs/minimal_diag",
    },
    "nano_v2_ablation_v2_temp0": {
        "config_dir": "text_simulation/configs/nano_v2_ablation_v2_temp0",
        "log_dir":    "text_simulation/logs/nano_v2_ablation_v2_batch_temp0",
    },
    "nano_v2_ablation_v3_temp0": {
        "config_dir": "text_simulation/configs/nano_v2_ablation_v3_temp0",
        "log_dir":    "text_simulation/logs/nano_v2_ablation_v3_batch_temp0",
    },
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_prompt_files(input_dir: Path, max_personas: int):
    files = sorted(
        input_dir.glob("pid_*_prompt.txt"),
        key=lambda f: int(re.search(r"pid_(\d+)", f.name).group(1))
    )
    files = files[:max_personas]
    result = []
    for f in files:
        pid = re.search(r"pid_(\d+)", f.name).group(1)
        result.append((pid, f.read_text(encoding="utf-8")))
    return result


def build_batch_jsonl(prompts, model, system_instruction, reasoning_effort, max_tokens):
    lines = []
    for pid, prompt_text in prompts:
        body = {"model": model, "input": prompt_text}
        if system_instruction:
            body["instructions"] = system_instruction
        if reasoning_effort and reasoning_effort != "none":
            body["reasoning"] = {"effort": reasoning_effort}
        if max_tokens:
            body["max_output_tokens"] = max_tokens
        lines.append(json.dumps({
            "custom_id": f"pid_{pid}",
            "method":    "POST",
            "url":       "/v1/responses",
            "body":      body,
        }))
    return "\n".join(lines).encode("utf-8")


def parse_and_save(output_content, prompts, output_dir, config_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_map = {pid: text for pid, text in prompts}
    results = {}

    for line in output_content.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        pid = obj["custom_id"].replace("pid_", "")

        if obj.get("error"):
            results[pid] = {"error": obj["error"], "response_text": "", "usage_details": {}}
            continue

        response_body = obj.get("response", {}).get("body", obj.get("response", {}))
        output_text = response_body.get("output_text", "")
        if not output_text:
            for item in response_body.get("output", []):
                if item.get("type") == "message":
                    for cb in item.get("content", []):
                        if cb.get("type") == "output_text":
                            output_text = cb.get("text", "")
                            break

        usage = response_body.get("usage", {})
        output_tokens_details = usage.get("output_tokens_details", {})
        reasoning_tokens = output_tokens_details.get("reasoning_tokens", 0)
        total_output_tokens = usage.get("output_tokens", 0)
        actual_output_tokens = total_output_tokens - reasoning_tokens
        usage_details = {
            "prompt_token_count":     usage.get("input_tokens",  0),
            "completion_token_count": total_output_tokens,
            "total_token_count":      usage.get("total_tokens",  0),
            "reasoning_tokens":       reasoning_tokens,
            "actual_output_tokens":   actual_output_tokens,
        }
        response_status = response_body.get("status", "")
        incomplete_details = response_body.get("incomplete_details")
        results[pid] = {
            "response_text":    output_text,
            "usage_details":    usage_details,
            "response_status":  response_status,
            "incomplete_reason": incomplete_details.get("reason") if incomplete_details else None,
        }

    for pid, result in results.items():
        persona_dir = output_dir / f"pid_{pid}"
        persona_dir.mkdir(exist_ok=True)
        out = {
            "persona_id":        f"pid_{pid}",
            "question_id":       f"pid_{pid}",
            "prompt_text":       prompt_map.get(pid, ""),
            "response_text":     result.get("response_text", ""),
            "usage_details":     result.get("usage_details", {}),
            "response_status":   result.get("response_status", ""),
            "incomplete_reason": result.get("incomplete_reason"),
            "llm_call_error":    result.get("error"),
            "config":            config_path,
        }
        (persona_dir / f"pid_{pid}_response.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
    return results


def bad_personas(results):
    """Return list of pids with empty or incomplete responses."""
    bad = []
    for pid, r in results.items():
        if not r.get("response_text") or r.get("response_status") == "incomplete":
            bad.append(pid)
    return bad


def retry_bad_responses(client, job, results, round_num):
    """Re-submit bad personas with doubled max_output_tokens. Returns updated results dict."""
    bad = bad_personas(results)
    if not bad:
        return results

    cfg         = job["cfg"]
    output_dir  = job["output_dir"]
    config_path = job["config_path"]
    prompt_map  = {pid: text for pid, text in job["prompts"]}

    # Double max_output_tokens, cap at 65536
    orig_max = cfg.get("max_tokens", 16384)
    new_max  = min(orig_max * (2 ** round_num), 65536)
    print(f"    ↺ retry round {round_num}: {len(bad)} personas  max_tokens {orig_max}→{new_max}  "
          f"({', '.join(f'pid_{p}' for p in sorted(bad, key=int))})")

    retry_prompts = [(pid, prompt_map[pid]) for pid in bad if pid in prompt_map]
    if not retry_prompts:
        return results

    jsonl_bytes = build_batch_jsonl(
        retry_prompts,
        cfg["model_name"],
        cfg.get("system_instruction", ""),
        cfg.get("reasoning_effort", "none"),
        new_max,
    )

    try:
        batch_file = client.files.create(
            file=("batch_retry.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"config": config_path, "name": f"{job['name']}_retry{round_num}"},
        )
    except Exception as e:
        print(f"    ✗ retry submit failed: {e}")
        return results

    # Poll until done
    start = time.time()
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            b = client.batches.retrieve(batch.id)
        except Exception:
            continue
        if b.status == "completed":
            try:
                content = client.files.content(b.output_file_id).text
                retry_results = parse_and_save(content, retry_prompts, output_dir, config_path)
                # Merge: overwrite only the retried pids
                results.update(retry_results)
                still_bad = bad_personas(retry_results)
                print(f"    ↺ retry round {round_num} done: "
                      f"{len(retry_results)-len(still_bad)}/{len(retry_results)} fixed"
                      + (f"  still bad: {still_bad}" if still_bad else ""))
            except Exception as e:
                print(f"    ✗ retry download failed: {e}")
            break
        elif b.status in ("failed", "expired", "cancelled"):
            print(f"    ✗ retry batch {b.status}")
            break
        else:
            counts = b.request_counts
            print(f"    ↺ retry polling: {b.status}  {counts.completed}/{counts.total}")
        if time.time() - start > MAX_WAIT:
            print("    ✗ retry max wait exceeded")
            break

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",     required=True, choices=list(SUITES.keys()))
    parser.add_argument("--setting",   default=None)
    parser.add_argument("--reasoning", default=None)
    args = parser.parse_args()

    suite = SUITES[args.suite]
    config_dir = Path(suite["config_dir"])
    log_dir    = Path(suite["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    # ── 1. Collect configs to run ──────────────────────────────────────────
    config_paths = sorted(config_dir.glob("*.yaml"))
    jobs = []  # list of dicts with all job metadata

    for config_path in config_paths:
        name      = config_path.stem          # e.g. skill_v3__high
        setting   = name.split("__")[0]
        reasoning = name.split("__")[-1]

        if args.setting   and setting   != args.setting:   continue
        if args.reasoning and reasoning != args.reasoning: continue

        cfg = load_config(config_path)
        input_dir  = Path("text_simulation") / cfg["input_folder_dir"]
        out_dir_str = cfg["output_folder_dir"]
        output_dir = Path("text_simulation") / out_dir_str

        jobs.append({
            "name":        name,
            "config_path": str(config_path),
            "cfg":         cfg,
            "input_dir":   input_dir,
            "output_dir":  output_dir,
            "log_path":    log_dir / f"{name}.log",
        })

    print(f"\nSuite: {args.suite}  |  {len(jobs)} configs to submit")
    print(f"{'='*60}")

    # ── 2. Submit ALL batch jobs at once ───────────────────────────────────
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Submitting {len(jobs)} batch jobs...")

    for job in jobs:
        cfg        = job["cfg"]
        input_dir  = job["input_dir"]
        output_dir = job["output_dir"]

        # Skip if already completed
        if output_dir.exists() and any(output_dir.rglob("*_response.json")):
            existing = len(list(output_dir.rglob("*_response.json")))
            max_p    = cfg.get("max_personas", 20)
            if existing >= max_p:
                print(f"  SKIP {job['name']} — {existing} results already exist")
                job["status"] = "skipped"
                continue

        prompts = get_prompt_files(input_dir, cfg.get("max_personas", 20))
        if not prompts:
            print(f"  SKIP {job['name']} — no prompt files in {input_dir}")
            job["status"] = "no_prompts"
            continue

        job["prompts"] = prompts

        jsonl_bytes = build_batch_jsonl(
            prompts,
            cfg["model_name"],
            cfg.get("system_instruction", ""),
            cfg.get("reasoning_effort", "none"),
            cfg.get("max_tokens", 16384),
        )

        try:
            batch_file = client.files.create(
                file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
                purpose="batch",
            )
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/responses",
                completion_window="24h",
                metadata={"config": job["config_path"], "name": job["name"]},
            )
            job["batch_id"]   = batch.id
            job["file_id"]    = batch_file.id
            job["status"]     = "submitted"
            job["submit_time"] = time.time()
            print(f"  ✓ {job['name']:35s} → batch {batch.id}")
        except Exception as e:
            print(f"  ✗ {job['name']} — submit error: {e}")
            job["status"] = "submit_failed"

    submitted = [j for j in jobs if j.get("status") == "submitted"]
    print(f"\nSubmitted {len(submitted)}/{len(jobs)} batch jobs")

    if not submitted:
        print("Nothing to poll. Exiting.")
        return

    # ── 3. Poll all jobs simultaneously ────────────────────────────────────
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Polling {len(submitted)} jobs every {POLL_INTERVAL}s...")

    pending  = {j["batch_id"]: j for j in submitted}
    start    = time.time()

    while pending:
        time.sleep(POLL_INTERVAL)
        completed_this_round = []

        for batch_id, job in list(pending.items()):
            try:
                batch = client.batches.retrieve(batch_id)
            except Exception as e:
                print(f"  poll error {job['name']}: {e}")
                continue

            counts = batch.request_counts
            done   = counts.completed if counts else "?"
            total  = counts.total     if counts else "?"

            if batch.status in ("completed", "failed", "expired", "cancelled"):
                elapsed = time.time() - job["submit_time"]
                if batch.status == "completed":
                    # Download and save
                    try:
                        content = client.files.content(batch.output_file_id).text
                        results = parse_and_save(
                            content, job["prompts"], job["output_dir"], job["config_path"]
                        )
                        # Retry empty / incomplete responses (up to 2 rounds)
                        for retry_round in range(1, 3):
                            if not bad_personas(results):
                                break
                            results = retry_bad_responses(client, job, results, retry_round)
                        failed_r = sum(1 for r in results.values() if r.get("error"))
                        empty_r  = len(bad_personas(results))
                        job["status"] = "done"
                        suffix = f"  ⚠ {empty_r} still empty/incomplete" if empty_r else ""
                        print(f"  ✓ {job['name']:35s}  {len(results)-failed_r}/{len(results)} ok  ({elapsed:.0f}s){suffix}")
                        with open(job["log_path"], "w") as f:
                            f.write(f"batch_id={batch_id}\nstatus=completed\n"
                                    f"results={len(results)}\nfailed={failed_r}\nempty={empty_r}\n"
                                    f"output={job['output_dir']}\n")
                    except Exception as e:
                        print(f"  ✗ {job['name']} — download error: {e}")
                        job["status"] = "download_failed"
                else:
                    print(f"  ✗ {job['name']:35s}  status={batch.status}  ({elapsed:.0f}s)")
                    job["status"] = batch.status
                completed_this_round.append(batch_id)
            else:
                print(f"  … {job['name']:35s}  {batch.status}  {done}/{total}")

        for bid in completed_this_round:
            del pending[bid]

        if pending:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {len(pending)} jobs still pending...")

        if time.time() - start > MAX_WAIT:
            print("Max wait exceeded. Exiting.")
            break

    # ── 4. Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Suite {args.suite} complete")
    for status in ["done", "skipped", "submit_failed", "failed", "download_failed"]:
        count = sum(1 for j in jobs if j.get("status") == status)
        if count:
            print(f"  {status:20s}: {count}")


if __name__ == "__main__":
    main()
