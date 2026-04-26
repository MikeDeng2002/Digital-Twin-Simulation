"""
redownload_ablation_50p_results.py — Re-download 50p ablation batch results with correct parser.
"""

import os, json, re, yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VERSIONS   = ["v2", "v3"]
CONDITIONS = ["bg", "bg_dp", "bg_ep", "bg_dp_ep"]
LOG_BASE   = Path("text_simulation/logs/ablation_50p_temp0")
CONFIG_BASE = Path("text_simulation/configs")
OUTPUT_BASE = Path("text_simulation")


def extract_batch_id(log_path):
    if not log_path.exists(): return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"Batch created:\s*(batch_\S+)", line)
        if m: return m.group(1)
    return None


def get_prompts(input_dir, max_p=50):
    files = sorted(input_dir.glob("pid_*_prompt.txt"),
                   key=lambda f: int(re.search(r"pid_(\d+)", f.name).group(1)))
    return [(re.search(r"pid_(\d+)", f.name).group(1), f.read_text()) for f in files[:max_p]]


def parse_and_save(content, prompts, output_dir, config_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_map = {pid: text for pid, text in prompts}
    for line in content.strip().split("\n"):
        if not line.strip(): continue
        obj = json.loads(line)
        pid = obj["custom_id"].replace("pid_", "")
        if obj.get("error"):
            rt, us, rs = "", {}, ""
        else:
            rb = obj.get("response", {}).get("body", obj.get("response", {}))
            rt = rb.get("output_text", "")
            if not rt:
                for item in rb.get("output", []):
                    if item.get("type") == "message":
                        for cb in item.get("content", []):
                            if cb.get("type") == "output_text": rt = cb.get("text", ""); break
            u = rb.get("usage", {}); od = u.get("output_tokens_details", {})
            rk = od.get("reasoning_tokens", 0); to = u.get("output_tokens", 0)
            us = {"prompt_token_count": u.get("input_tokens", 0),
                  "completion_token_count": to, "total_token_count": u.get("total_tokens", 0),
                  "reasoning_tokens": rk, "actual_output_tokens": to - rk}
            rs = rb.get("status", "")
        pd = output_dir / f"pid_{pid}"; pd.mkdir(exist_ok=True)
        (pd / f"pid_{pid}_response.json").write_text(json.dumps({
            "persona_id": f"pid_{pid}", "question_id": f"pid_{pid}",
            "prompt_text": prompt_map.get(pid, ""), "response_text": rt,
            "usage_details": us, "response_status": rs,
            "llm_call_error": obj.get("error"), "config": config_path,
        }, indent=2))


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    ok = fail = skip = 0

    for version in VERSIONS:
        suite = f"nano_v2_ablation_50p_{version}_temp0"
        for condition in CONDITIONS:
            label    = f"{suite}/{condition}__high"
            log_path = LOG_BASE / suite / f"{condition}__high.log"
            batch_id = extract_batch_id(log_path)

            if not batch_id:
                print(f"  SKIP {label} — no batch ID in log")
                skip += 1; continue

            cfg    = yaml.safe_load((CONFIG_BASE / suite / f"{condition}__high.yaml").read_text())
            prompts = get_prompts(OUTPUT_BASE / cfg["input_folder_dir"], cfg.get("max_personas", 50))
            out_dir = OUTPUT_BASE / cfg["output_folder_dir"]

            try:
                batch = client.batches.retrieve(batch_id)
                if batch.status != "completed" or not batch.output_file_id:
                    print(f"  SKIP {label} — status={batch.status}"); skip += 1; continue
                content = client.files.content(batch.output_file_id).text
                parse_and_save(content, prompts, out_dir, str(cfg))
                print(f"  OK   {label}"); ok += 1
            except Exception as e:
                print(f"  FAIL {label} — {e}"); fail += 1

    print(f"\nDone: {ok} ok, {fail} failed, {skip} skipped")


if __name__ == "__main__":
    main()
