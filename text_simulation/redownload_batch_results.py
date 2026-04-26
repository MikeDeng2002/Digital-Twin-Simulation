"""
redownload_batch_results.py — Re-download and re-parse batch results using fixed parser.

Fetches the most recent completed batch for each config name from OpenAI,
re-parses the output (fixing the response.body parsing bug), and overwrites
local response JSON files.

Usage (from Digital-Twin-Simulation/):
    python text_simulation/redownload_batch_results.py --suite nano_temp0
    python text_simulation/redownload_batch_results.py --suite mini_temp0
"""

import os, io, json, re, argparse, yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUITES = {
    "nano_temp0": "text_simulation/configs/nano_temp0",
    "mini_temp0": "text_simulation/configs/mini_temp0",
}


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

        # Fix: response body is nested under response.body
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=list(SUITES.keys()))
    parser.add_argument("--setting",   default=None)
    parser.add_argument("--reasoning", default=None)
    args = parser.parse_args()

    config_dir = Path(SUITES[args.suite])
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Build map: config_name -> most recent completed batch for THIS suite
    suite_key = args.suite  # e.g. "nano_temp0" or "mini_temp0"
    print(f"Fetching batch list from OpenAI (filtering for {suite_key})...")
    all_batches = list(client.batches.list(limit=100))
    name_to_batch = {}
    for b in all_batches:
        meta = b.metadata or {}
        name = meta.get("name")
        cfg_path = meta.get("config", "")
        if not name or b.status != "completed":
            continue
        # Only match batches belonging to this suite
        if suite_key not in cfg_path:
            continue
        if name not in name_to_batch or b.created_at > name_to_batch[name].created_at:
            name_to_batch[name] = b
    print(f"Found {len(name_to_batch)} named completed batches for {suite_key}")

    config_paths = sorted(config_dir.glob("*.yaml"))
    ok = fail = skip = 0

    for config_path in config_paths:
        name = config_path.stem
        setting  = name.split("__")[0]
        reasoning = name.split("__")[-1]

        if args.setting   and setting   != args.setting:   continue
        if args.reasoning and reasoning != args.reasoning: continue

        if name not in name_to_batch:
            print(f"  SKIP {name} — no completed batch found on OpenAI")
            skip += 1
            continue

        cfg = yaml.safe_load(config_path.read_text())
        input_dir  = Path("text_simulation") / cfg["input_folder_dir"]
        output_dir = Path("text_simulation") / cfg["output_folder_dir"]
        prompts    = get_prompt_files(input_dir, cfg.get("max_personas", 20))

        batch = name_to_batch[name]
        try:
            content = client.files.content(batch.output_file_id).text
            results = parse_and_save(content, prompts, output_dir, str(config_path))
            empty   = sum(1 for r in results.values() if not r.get("response_text"))
            filled  = len(results) - empty
            print(f"  ✓ {name:40s} {filled}/{len(results)} with text  ({empty} empty)")
            ok += 1
        except Exception as e:
            print(f"  ✗ {name} — {e}")
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed, {skip} skipped")


if __name__ == "__main__":
    main()
