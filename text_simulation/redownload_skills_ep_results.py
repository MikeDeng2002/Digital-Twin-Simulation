"""
redownload_skills_ep_results.py — Re-download and re-parse skills+ep ablation batch results.

The original run_nano_batch.py used the wrong response parser (missing .get("body",...)),
producing empty response_text for all 84 configs. This script:
  1. Reads batch IDs from the log files
  2. Re-downloads the output file from OpenAI
  3. Re-parses with the correct parser (response.body structure)
  4. Overwrites the local response JSON files

Usage (from Digital-Twin-Simulation/):
    python text_simulation/redownload_skills_ep_results.py
    python text_simulation/redownload_skills_ep_results.py --version v1
    python text_simulation/redownload_skills_ep_results.py --version v2 --condition bg_ep
"""

import os, json, re, argparse, yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VERSIONS    = ["v1", "v2", "v3"]
CONDITIONS  = ["bg", "bg_dp", "bg_tools", "bg_ep", "bg_dp_tools", "bg_dp_ep", "bg_dp_tools_ep"]
REASONINGS  = ["none", "low", "medium", "high"]

LOG_BASE    = Path("text_simulation/logs/skills_ep_ablation_temp0")
CONFIG_BASE = Path("text_simulation/configs")
OUTPUT_BASE = Path("text_simulation")


def extract_batch_id(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"Batch created:\s*(batch_\S+)", line)
        if m:
            return m.group(1)
    return None


def get_prompt_files(input_dir: Path, max_personas: int):
    files = sorted(
        input_dir.glob("pid_*_prompt.txt"),
        key=lambda f: int(re.search(r"pid_(\d+)", f.name).group(1))
    )
    return [(re.search(r"pid_(\d+)", f.name).group(1), f.read_text(encoding="utf-8"))
            for f in files[:max_personas]]


def parse_and_save(output_content: str, prompts, output_dir: Path, config_path: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_map = {pid: text for pid, text in prompts}
    results = {}

    for line in output_content.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        pid = obj["custom_id"].replace("pid_", "")

        if obj.get("error"):
            results[pid] = {"response_text": "", "usage_details": {}, "error": obj["error"]}
            continue

        # Correct parser: response.body contains the actual data
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
        out_details = usage.get("output_tokens_details", {})
        reasoning_tokens = out_details.get("reasoning_tokens", 0)
        total_output = usage.get("output_tokens", 0)
        usage_details = {
            "prompt_token_count":     usage.get("input_tokens", 0),
            "completion_token_count": total_output,
            "total_token_count":      usage.get("total_tokens", 0),
            "reasoning_tokens":       reasoning_tokens,
            "actual_output_tokens":   total_output - reasoning_tokens,
        }
        response_status    = response_body.get("status", "")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",   choices=VERSIONS + ["all"], default="all")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--reasoning", default=None)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    versions = VERSIONS if args.version == "all" else [args.version]

    ok = fail = skip = 0
    total = len(versions) * len(CONDITIONS) * len(REASONINGS)

    for version in versions:
        suite = f"nano_skills_ep_ablation_{version}_temp0"
        for condition in CONDITIONS:
            if args.condition and condition != args.condition:
                continue
            for reasoning in REASONINGS:
                if args.reasoning and reasoning != args.reasoning:
                    continue

                label = f"{suite}/{condition}__{reasoning}"
                log_path = LOG_BASE / suite / f"{condition}__{reasoning}.log"
                batch_id = extract_batch_id(log_path)

                if not batch_id:
                    print(f"  SKIP {label} — no batch ID in log")
                    skip += 1
                    continue

                config_path = CONFIG_BASE / suite / f"{condition}__{reasoning}.yaml"
                cfg = yaml.safe_load(config_path.read_text())
                input_dir  = OUTPUT_BASE / cfg["input_folder_dir"]
                output_dir = OUTPUT_BASE / cfg["output_folder_dir"]
                prompts    = get_prompt_files(input_dir, cfg.get("max_personas", 20))

                try:
                    batch = client.batches.retrieve(batch_id)
                    if batch.status != "completed" or not batch.output_file_id:
                        print(f"  SKIP {label} — batch status={batch.status}")
                        skip += 1
                        continue

                    content = client.files.content(batch.output_file_id).text
                    results = parse_and_save(content, prompts, output_dir, str(config_path))
                    empty  = sum(1 for r in results.values() if not r.get("response_text"))
                    filled = len(results) - empty
                    print(f"  OK  {label:60s} {filled}/{len(results)} with text")
                    ok += 1
                except Exception as e:
                    print(f"  FAIL {label} — {e}")
                    fail += 1

    print(f"\nDone: {ok} ok, {fail} failed, {skip} skipped  (of {total} total)")


if __name__ == "__main__":
    main()
