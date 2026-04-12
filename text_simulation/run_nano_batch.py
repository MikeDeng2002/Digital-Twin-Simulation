"""
run_nano_batch.py — Batch API runner for the nano experiment.

Uses OpenAI's Batch API (/v1/batches) to submit all persona prompts in one
JSONL file, poll until complete, then save results in the same format as
run_LLM_simulations.py. This reduces cost by 50% vs real-time API calls.

Usage (from Digital-Twin-Simulation/):
    # Single config, single rep
    python text_simulation/run_nano_batch.py \
        --config text_simulation/configs/nano/skill_v3__high.yaml \
        --nano_rep rep_1

    # All 120 runs via shell runner
    bash text_simulation/run_nano_experiment.sh
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

POLL_INTERVAL_SECONDS = 30   # how often to check batch status
MAX_WAIT_SECONDS      = 86400  # 24 hours max


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_prompt_files(input_dir: Path, max_personas: int) -> list[tuple[str, str]]:
    """Return [(pid, prompt_text), ...] sorted by pid, capped at max_personas."""
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


def build_batch_jsonl(
    prompts: list[tuple[str, str]],
    model: str,
    system_instruction: str,
    reasoning_effort: str,
    max_tokens: int,
) -> bytes:
    """Build JSONL content for the OpenAI Batch API."""
    lines = []
    for pid, prompt_text in prompts:
        body = {
            "model": model,
            "input": prompt_text,
        }
        if system_instruction:
            body["instructions"] = system_instruction
        if reasoning_effort and reasoning_effort != "none":
            body["reasoning"] = {"effort": reasoning_effort}
        if max_tokens:
            body["max_output_tokens"] = max_tokens

        request = {
            "custom_id": f"pid_{pid}",
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        lines.append(json.dumps(request))
    return "\n".join(lines).encode("utf-8")


def poll_batch(client: OpenAI, batch_id: str) -> object:
    """Poll batch until terminal status, return final batch object."""
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else "?"
        total    = batch.request_counts.total     if batch.request_counts else "?"
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] status={status}  "
              f"completed={completed}/{total}")

        if status in ("completed", "failed", "expired", "cancelled"):
            return batch

        elapsed = time.time() - start
        if elapsed > MAX_WAIT_SECONDS:
            raise TimeoutError(f"Batch {batch_id} did not complete within {MAX_WAIT_SECONDS}s")

        time.sleep(POLL_INTERVAL_SECONDS)


def parse_batch_output(content: str) -> dict[str, dict]:
    """Parse JSONL output file → {pid: result_dict}."""
    results = {}
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]          # "pid_1"
        pid = custom_id.replace("pid_", "")

        if obj.get("error"):
            results[pid] = {"error": obj["error"], "response_text": "", "usage_details": {}}
            continue

        response_body = obj.get("response", {})
        # responses API: output_text at top level, or inside output list
        output_text = response_body.get("output_text", "")
        if not output_text:
            output_items = response_body.get("output", [])
            for item in output_items:
                if item.get("type") == "message":
                    for content_block in item.get("content", []):
                        if content_block.get("type") == "output_text":
                            output_text = content_block.get("text", "")
                            break

        usage = response_body.get("usage", {})
        usage_details = {
            "prompt_token_count":     usage.get("input_tokens",  0),
            "completion_token_count": usage.get("output_tokens", 0),
            "total_token_count":      usage.get("total_tokens",  0),
        }
        results[pid] = {"response_text": output_text, "usage_details": usage_details}
    return results


def save_results(
    results: dict[str, dict],
    prompts: list[tuple[str, str]],
    output_dir: Path,
    config_path: str,
):
    """Save each persona's result as pid_{pid}/{pid}_response.json (same format as run_LLM_simulations.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_map = {pid: text for pid, text in prompts}

    for pid, result in results.items():
        persona_dir = output_dir / f"pid_{pid}"
        persona_dir.mkdir(exist_ok=True)
        out = {
            "persona_id":    f"pid_{pid}",
            "question_id":   f"pid_{pid}",
            "prompt_text":   prompt_map.get(pid, ""),
            "response_text": result.get("response_text", ""),
            "usage_details": result.get("usage_details", {}),
            "llm_call_error": result.get("error"),
            "config": config_path,
        }
        out_path = persona_dir / f"pid_{pid}_response.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"  Saved {len(results)} results → {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True, help="Path to YAML config file")
    parser.add_argument("--nano_rep",  required=True, help="Repetition label: rep_1, rep_2, rep_3")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model             = cfg["model_name"]
    system_instruction = cfg.get("system_instruction", "")
    reasoning_effort  = cfg.get("reasoning_effort", "none")
    max_tokens        = cfg.get("max_tokens", 16384)
    max_personas      = cfg.get("max_personas", 5)
    input_folder      = Path("text_simulation") / cfg["input_folder_dir"]
    output_folder     = Path("text_simulation") / cfg["output_folder_dir"].replace("{rep}", args.nano_rep)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    print(f"\n{'='*60}")
    print(f"Config:    {args.config}")
    print(f"Rep:       {args.nano_rep}")
    print(f"Model:     {model}  |  reasoning_effort: {reasoning_effort}")
    print(f"Input:     {input_folder}")
    print(f"Output:    {output_folder}")
    print(f"Personas:  up to {max_personas}")

    # 1. Load prompts
    prompts = get_prompt_files(input_folder, max_personas)
    if not prompts:
        print(f"No prompt files found in {input_folder}")
        return
    print(f"Loaded {len(prompts)} persona prompts")

    # 2. Build JSONL batch file
    jsonl_bytes = build_batch_jsonl(
        prompts, model, system_instruction, reasoning_effort, max_tokens
    )

    # 3. Upload JSONL file
    print("Uploading batch file...")
    batch_file = client.files.create(
        file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch",
    )
    print(f"  Uploaded file: {batch_file.id}")

    # 4. Create batch job
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "config":   args.config,
            "nano_rep": args.nano_rep,
        },
    )
    print(f"  Batch created: {batch.id}")

    # 5. Poll until complete
    print("Polling batch status...")
    batch = poll_batch(client, batch.id)

    if batch.status != "completed":
        print(f"Batch ended with status: {batch.status}")
        if batch.errors:
            print(f"Errors: {batch.errors}")
        return

    # 6. Download output
    print("Downloading results...")
    output_content = client.files.content(batch.output_file_id).text

    # 7. Parse and save
    results = parse_batch_output(output_content)
    failed = {pid: r for pid, r in results.items() if r.get("error")}
    if failed:
        print(f"  {len(failed)} failed requests: {list(failed.keys())}")

    save_results(results, prompts, output_folder, args.config)
    print(f"Done. {len(results) - len(failed)}/{len(results)} succeeded.")


if __name__ == "__main__":
    main()
