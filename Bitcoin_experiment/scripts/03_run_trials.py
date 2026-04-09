"""
03_run_trials.py — Run 50 trials per persona for the Bitcoin celebrity influence experiment.

Per trial:
  Call 2 (recognition): persona sees 4 celebrities + prices → "do I know them?"
  Call 3 (prediction):  chained from Call 2 → Type A (number) or Type B (number+reasoning)

Celebrities: 2 bulls + 2 bears drawn randomly per trial (rotating).
All API calls use o4-mini with v3_maximum persona representation.

Usage (from Digital-Twin-Simulation/):
    poetry run python Bitcoin_experiment/scripts/03_run_trials.py --pids 1,2,3,4,5
    poetry run python Bitcoin_experiment/scripts/03_run_trials.py --pids 1,2,3,4,5 --trials 5
    poetry run python Bitcoin_experiment/scripts/03_run_trials.py --pids 1,2,3,4,5 --type both
"""

import os
import json
import re
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "o4-mini"
REPRESENTATION = "v3_maximum"
MAX_TOKENS = 2048
N_TRIALS = 50

CELEBRITIES_DIR = Path("Bitcoin_experiment/data/celebrities")
SKILLS_DIR = Path("text_simulation/skills")
RESULTS_DIR = Path("Bitcoin_experiment/results") / REPRESENTATION

BULLS = ["michael_saylor", "cathie_wood", "elon_musk", "jack_dorsey", "larry_fink", "donald_trump"]
BEARS = ["warren_buffett", "peter_schiff", "jamie_dimon", "nouriel_roubini"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RECOGNITION_PROMPT = """\
You are about to hear Bitcoin price predictions from four people in your network.
Before forming your own view, reflect on each one:

{neighbor_block}

For each person, answer three things:
1. Do you know who this person is? (yes / no / vaguely)
2. If yes — what do you know about them and how much do you trust their judgment on Bitcoin or financial markets?
3. If no or vaguely — would you look them up before forming an opinion? If yes, what would you likely find and how would that change your view of their prediction?

Answer as yourself, based on your background, information sources, and what kind of person you are. Be honest if you would not recognize someone."""

PREDICTION_TYPE_A = """\
Now, based on your assessment of these four people above and your own beliefs, what is your Bitcoin price prediction for end of 2025?

Respond with a single integer. No dollar sign. No commas. No explanation.
Example: 85000"""

PREDICTION_TYPE_B = """\
Now, based on your assessment of these four people above and your own beliefs, provide:

1. Your prediction: a single dollar number
2. Your reasoning: 3-5 sentences explaining why you predict this price. Draw on your background, the credibility you assigned to each person above, and how you typically reason under uncertainty.

Format your response EXACTLY as:
PREDICTION: [integer only, no $ or commas]
REASONING: [3-5 sentences]"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_celebrity_data() -> dict:
    """Load all celebrity profiles and reasoning from cache."""
    celebrities = {}
    for slug in BULLS + BEARS:
        base = CELEBRITIES_DIR / slug
        reasoning_path = base / "reasoning.txt"
        profile_path = base / "profile.txt"
        if not reasoning_path.exists():
            raise FileNotFoundError(
                f"Missing reasoning.txt for {slug}. Run 01_generate_celebrity_data.py first."
            )
        profile = profile_path.read_text(encoding="utf-8")
        name = profile.split("\n")[0].replace("Name:", "").strip()
        title_lines = [l for l in profile.split("\n") if l.startswith("Title:")]
        title = title_lines[0].replace("Title:", "").strip() if title_lines else slug

        raw = reasoning_path.read_text(encoding="utf-8").strip()
        price = parse_price_from_reasoning(raw)
        reasoning = parse_reasoning_text(raw)
        celebrities[slug] = {
            "slug": slug, "name": name, "title": title,
            "price": price, "reasoning": reasoning,
        }
    return celebrities


def parse_price_from_reasoning(text: str) -> int | None:
    for line in text.split("\n"):
        if line.startswith("PREDICTION:"):
            raw = line.replace("PREDICTION:", "").strip().replace("$", "").replace(",", "")
            try:
                return int(float(raw))
            except ValueError:
                pass
    return None


def parse_reasoning_text(text: str) -> str | None:
    parts = text.split("REASONING:")
    if len(parts) > 1:
        return parts[1].strip()
    return text.strip()


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


def load_baseline(pid: str) -> int | None:
    path = RESULTS_DIR / "baselines" / f"pid_{pid}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("baseline_price")


# ---------------------------------------------------------------------------
# Trial logic
# ---------------------------------------------------------------------------

def draw_celebrities(trial: int, pid: int) -> list[str]:
    """Draw 2 bulls + 2 bears using fixed seed = pid * 1000 + trial."""
    rng = random.Random(int(pid) * 1000 + trial)
    bulls = rng.sample(BULLS, 2)
    bears = rng.sample(BEARS, 2)
    neighbors = bulls + bears
    rng.shuffle(neighbors)
    return neighbors


def build_neighbor_block_type_a(celebrities: dict, slugs: list[str]) -> str:
    lines = []
    for slug in slugs:
        c = celebrities[slug]
        lines.append(f"- {c['name']} ({c['title']}): ${c['price']:,}")
    return "\n".join(lines)


def build_neighbor_block_type_b(celebrities: dict, slugs: list[str]) -> str:
    lines = []
    for slug in slugs:
        c = celebrities[slug]
        reasoning_short = c["reasoning"]
        if reasoning_short and len(reasoning_short) > 200:
            reasoning_short = reasoning_short[:200].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {c['name']} ({c['title']}): ${c['price']:,}")
        if reasoning_short:
            lines.append(f'  "{reasoning_short}"')
        lines.append("")
    return "\n".join(lines).strip()


def parse_price(text: str) -> int | None:
    text = text.strip()
    # Try PREDICTION: format first
    for line in text.split("\n"):
        if line.startswith("PREDICTION:"):
            raw = line.replace("PREDICTION:", "").strip().replace("$", "").replace(",", "")
            try:
                return int(float(raw))
            except ValueError:
                pass
    # Try bare number
    clean = text.replace("$", "").replace(",", "").strip()
    match = re.search(r"\b(\d{4,7})\b", clean)
    if match:
        return int(match.group(1))
    return None


def parse_reasoning(text: str) -> str | None:
    parts = text.split("REASONING:")
    if len(parts) > 1:
        return parts[1].strip()
    return None


def call_api(client: OpenAI, messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def run_trial(
    client: OpenAI,
    pid: str,
    trial: int,
    system_prompt: str,
    celebrities: dict,
    baseline_price: int | None,
    experiment_type: str,  # "A", "B", or "both"
    force: bool,
) -> dict:
    slugs = draw_celebrities(trial, int(pid))
    neighbor_celeb_data = [celebrities[s] for s in slugs]
    celeb_mean = int(sum(c["price"] for c in neighbor_celeb_data) / len(neighbor_celeb_data))

    result = {
        "pid": pid,
        "trial": trial,
        "representation": REPRESENTATION,
        "model": MODEL,
        "neighbors": [
            {"slug": c["slug"], "name": c["name"], "price": c["price"]}
            for c in neighbor_celeb_data
        ],
        "celeb_mean": celeb_mean,
        "baseline_price": baseline_price,
    }

    # -- Call 2: Recognition (shared for both types) --
    neighbor_block_a = build_neighbor_block_type_a(celebrities, slugs)
    recognition_prompt = RECOGNITION_PROMPT.format(neighbor_block=neighbor_block_a)
    messages_base = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": recognition_prompt},
    ]
    recognition_response = call_api(client, messages_base)
    result["recognition_response"] = recognition_response

    # -- Call 3A: Type A prediction --
    if experiment_type in ("A", "both"):
        messages_a = messages_base + [
            {"role": "assistant", "content": recognition_response},
            {"role": "user",      "content": PREDICTION_TYPE_A},
        ]
        response_a = call_api(client, messages_a)
        result["type_a_raw"] = response_a
        result["type_a_prediction"] = parse_price(response_a)

    # -- Call 3B: Type B prediction (use Type B neighbor block with reasoning) --
    if experiment_type in ("B", "both"):
        neighbor_block_b = build_neighbor_block_type_b(celebrities, slugs)
        recognition_prompt_b = RECOGNITION_PROMPT.format(neighbor_block=neighbor_block_b)
        messages_b_recognition = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": recognition_prompt_b},
        ]
        recognition_response_b = call_api(client, messages_b_recognition)
        result["recognition_response_b"] = recognition_response_b

        messages_b = messages_b_recognition + [
            {"role": "assistant", "content": recognition_response_b},
            {"role": "user",      "content": PREDICTION_TYPE_B},
        ]
        response_b = call_api(client, messages_b)
        result["type_b_raw"] = response_b
        result["type_b_prediction"] = parse_price(response_b)
        result["type_b_reasoning"] = parse_reasoning(response_b)

    return result


def run_persona(
    client: OpenAI,
    pid: str,
    n_trials: int,
    celebrities: dict,
    experiment_type: str,
    force: bool,
) -> dict:
    system_prompt = build_system_prompt(pid)
    if system_prompt is None:
        return {"pid": pid, "error": "Skill files not found"}

    baseline_price = load_baseline(pid)
    if baseline_price is None:
        print(f"  WARNING: No baseline for pid_{pid} — run 02_run_baseline.py first")

    out_dir = RESULTS_DIR / f"type_{'ab' if experiment_type == 'both' else experiment_type.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pid_{pid}.json"

    # Load existing trials if resuming
    existing = {}
    if out_path.exists() and not force:
        try:
            existing_list = json.loads(out_path.read_text())
            existing = {t["trial"]: t for t in existing_list}
        except Exception:
            existing = {}

    trials_skipped = sum(1 for t in range(1, n_trials + 1) if t in existing and not force)
    pending_trials = [t for t in range(1, n_trials + 1) if t not in existing or force]

    completed_trials = list(existing.values())
    save_lock = threading.Lock()

    def run_and_save(trial: int) -> dict:
        result = run_trial(
            client, pid, trial, system_prompt, celebrities,
            baseline_price, experiment_type, force
        )
        with save_lock:
            completed_trials.append(result)
            out_path.write_text(json.dumps(completed_trials, indent=2))
        return result

    trials_done = 0
    # Run trials for this persona in parallel (up to 5 at a time)
    with ThreadPoolExecutor(max_workers=5) as trial_executor:
        futures = {trial_executor.submit(run_and_save, t): t for t in pending_trials}
        for future in as_completed(futures):
            try:
                future.result()
                trials_done += 1
            except Exception as e:
                print(f"  Trial error for pid_{pid}: {e}")

    return {
        "pid": pid,
        "done": trials_done,
        "skipped": trials_skipped,
        "total": len(completed_trials),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", type=str, required=True,
                        help="Comma-separated PIDs to run")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"Number of trials per persona (default: {N_TRIALS})")
    parser.add_argument("--type", choices=["A", "B", "both"], default="both",
                        help="Experiment type (default: both)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all trials even if cached")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel personas (default: 10)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    celebrities = load_celebrity_data()
    print(f"Loaded {len(celebrities)} celebrities:")
    for slug, c in celebrities.items():
        cat = "BULL" if slug in BULLS else "BEAR"
        print(f"  {cat}  {c['name']:<25} ${c['price']:>10,}")

    pids = [p.strip() for p in args.pids.split(",")]
    print(f"\nRunning {args.trials} trials × {len(pids)} personas "
          f"(type={args.type}, model={MODEL}, rep={REPRESENTATION}, "
          f"workers={args.workers})\n")

    completed = 0
    errors = 0

    def run_one_persona(pid: str) -> dict:
        return run_persona(client, pid, args.trials, celebrities, args.type, args.force)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one_persona, pid): pid for pid in pids}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                result = future.result()
                completed += 1
                if "error" in result:
                    errors += 1
                    print(f"  [{completed}/{len(pids)}] pid_{pid} ERROR: {result['error']}")
                else:
                    print(f"  [{completed}/{len(pids)}] pid_{pid} done — "
                          f"{result['done']} new, {result['skipped']} skipped")
            except Exception as e:
                completed += 1
                errors += 1
                print(f"  [{completed}/{len(pids)}] pid_{pid} EXCEPTION: {e}")

    print(f"\nFinished. {completed - errors}/{len(pids)} personas succeeded.")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
