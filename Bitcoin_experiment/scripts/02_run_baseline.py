"""
02_run_baseline.py — Generate and cache baseline Bitcoin predictions for all personas.

For each persona: loads v3_maximum skill profile, calls o4-mini with zero neighbors,
stores the baseline price prediction. Run once before trials.

Usage (from Digital-Twin-Simulation/):
    poetry run python Bitcoin_experiment/scripts/02_run_baseline.py
    poetry run python Bitcoin_experiment/scripts/02_run_baseline.py --pids 1,2,3,4,5
    poetry run python Bitcoin_experiment/scripts/02_run_baseline.py --force
"""

import os
import json
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "o4-mini"
REPRESENTATION = "v3_maximum"
MAX_TOKENS = 1024

SKILLS_DIR = Path("text_simulation/skills")
RESULTS_DIR = Path("Bitcoin_experiment/results") / REPRESENTATION / "baselines"

BASELINE_PROMPT = """\
You are answering this question as the person described above.

Question: What do you think Bitcoin's price will be at the end of 2025?

Answer based solely on your own knowledge and beliefs.
You have not seen or heard anyone else's opinion on this.

Respond with a single integer. No dollar sign. No commas. No explanation.
Example: 85000"""


def build_system_prompt(pid: str) -> str | None:
    base = SKILLS_DIR / f"pid_{pid}" / REPRESENTATION
    try:
        background = (base / "background.txt").read_text(encoding="utf-8").strip()
        tools = (base / "tools.txt").read_text(encoding="utf-8").strip()
        decision = (base / "decision_procedure.txt").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return (
        "You are simulating a specific person based on their skill profile.\n\n"
        f"## Background Knowledge\n{background}\n\n"
        f"## Information Sources They Use\n{tools}\n\n"
        f"## How They Make Decisions Under Uncertainty\n{decision}\n\n"
        "Stay fully in character throughout this conversation."
    )


def parse_price(text: str) -> int | None:
    text = text.strip().replace("$", "").replace(",", "")
    # Handle shorthand: 150k, 1.5M
    text_lower = text.lower()
    if text_lower.endswith("k"):
        try:
            return int(float(text_lower[:-1]) * 1_000)
        except ValueError:
            pass
    if text_lower.endswith("m"):
        try:
            return int(float(text_lower[:-1]) * 1_000_000)
        except ValueError:
            pass
    # Extract first integer-like number
    match = re.search(r"[\d]+", text.replace(".", ""))
    if match:
        try:
            return int(text.replace(".", "").replace(",", "").replace("$", "").strip().split()[0])
        except Exception:
            pass
    match = re.search(r"([\d,]+)", text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def run_baseline(client: OpenAI, pid: str, force: bool) -> dict:
    out_path = RESULTS_DIR / f"pid_{pid}.json"
    if out_path.exists() and not force:
        data = json.loads(out_path.read_text())
        return {"pid": pid, "price": data["baseline_price"], "skipped": True}

    system_prompt = build_system_prompt(pid)
    if system_prompt is None:
        return {"pid": pid, "price": None, "error": f"Skill files not found for pid_{pid}"}

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": BASELINE_PROMPT},
            ],
            max_completion_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()
        price = parse_price(raw)
        result = {
            "pid": pid,
            "representation": REPRESENTATION,
            "model": MODEL,
            "baseline_price": price,
            "raw_response": raw,
        }
        out_path.write_text(json.dumps(result, indent=2))
        return {"pid": pid, "price": price, "skipped": False}
    except Exception as e:
        return {"pid": pid, "price": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", type=str, default=None,
                        help="Comma-separated PIDs (default: all with skills)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if cached")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",")]
    else:
        pids = sorted(
            [d.name.replace("pid_", "") for d in SKILLS_DIR.iterdir()
             if d.is_dir() and (d / REPRESENTATION / "background.txt").exists()],
            key=lambda x: int(x)
        )

    print(f"Running baselines for {len(pids)} personas (model={MODEL}, rep={REPRESENTATION})\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_baseline, client, pid, args.force): pid for pid in pids}
        done = 0
        for future in as_completed(futures):
            r = future.result()
            done += 1
            status = "SKIP" if r.get("skipped") else ("ERROR" if r.get("error") else "OK"  )
            price_str = f"${r['price']:,}" if r.get("price") else r.get("error", "?")
            print(f"  [{done}/{len(pids)}] pid_{r['pid']} {status} → {price_str}")
            results.append(r)

    successful = [r for r in results if r.get("price")]
    prices = [r["price"] for r in successful]

    print(f"\n{'='*50}")
    print(f"BASELINE SUMMARY  (n={len(successful)} successful)")
    print(f"{'='*50}")
    if prices:
        import statistics
        print(f"  Mean:   ${statistics.mean(prices):,.0f}")
        print(f"  Median: ${statistics.median(prices):,.0f}")
        print(f"  Std:    ${statistics.stdev(prices):,.0f}")
        print(f"  Min:    ${min(prices):,}")
        print(f"  Max:    ${max(prices):,}")
    print(f"\nBaselines saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
