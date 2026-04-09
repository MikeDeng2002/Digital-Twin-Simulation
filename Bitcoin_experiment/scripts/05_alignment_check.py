"""
05_alignment_check.py — Check alignment between stated trust (Call 2 recognition)
and behavioral influence (regression weights from 04_analyze.py).

For each persona × celebrity pair:
  1. Score Call 2 recognition responses → stated trust (-2 to +2)
  2. Estimate per-celebrity behavioral contribution from trial data
  3. Correlate stated trust vs actual influence
  4. Flag outliers (high trust / low influence, low trust / high influence)

Usage (from Digital-Twin-Simulation/):
    poetry run python Bitcoin_experiment/scripts/05_alignment_check.py
    poetry run python Bitcoin_experiment/scripts/05_alignment_check.py --pids 1,2,3,4,5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SCORING_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_TOKENS = 512

REPRESENTATION = "v3_maximum"
RESULTS_DIR = Path("Bitcoin_experiment/results") / REPRESENTATION
ANALYSIS_DIR = Path("Bitcoin_experiment/analysis")

BULLS = ["michael_saylor", "cathie_wood", "elon_musk", "jack_dorsey", "larry_fink", "donald_trump"]
BEARS = ["warren_buffett", "peter_schiff", "jamie_dimon", "nouriel_roubini"]
ALL_SLUGS = BULLS + BEARS

CELEBRITIES_DIR = Path("Bitcoin_experiment/data/celebrities")

SCORING_PROMPT = """\
Read this person's assessment of four celebrities and their Bitcoin price predictions.
For each celebrity mentioned, extract a trust score and recognition level.

Trust score scale:
  -2 = actively distrusts / would dismiss their view entirely
  -1 = skeptical or doubtful of their judgment
   0 = neutral, doesn't know them, or would not weight their view
  +1 = somewhat credible, would consider their view
  +2 = highly credible, would weight their view significantly

Also extract:
  known: "yes" / "vaguely" / "no"
  would_search: true / false (would they look up the person if unfamiliar?)

The assessment text:
{recognition_text}

Output a JSON object mapping celebrity name to their scores. Use the exact celebrity names from the text.
Example format:
{{
  "Michael Saylor": {{"known": "no", "trust": 0, "would_search": true}},
  "Warren Buffett": {{"known": "yes", "trust": 2, "would_search": false}}
}}

Output ONLY the JSON. No explanation."""


def load_celebrity_names() -> dict:
    """Map slug → display name."""
    names = {}
    for slug in ALL_SLUGS:
        profile = (CELEBRITIES_DIR / slug / "profile.txt").read_text(encoding="utf-8")
        name = profile.split("\n")[0].replace("Name:", "").strip()
        names[slug] = name
    return names


def score_recognition(client: OpenAI, recognition_text: str, cache_path: Path) -> dict | None:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    prompt = SCORING_PROMPT.format(recognition_text=recognition_text)
    try:
        response = client.chat.completions.create(
            model=SCORING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
        cache_path.write_text(json.dumps(scores, indent=2))
        return scores
    except Exception as e:
        print(f"    Scoring error: {e}")
        return None


def compute_per_celebrity_influence(pid: str, trials: list[dict], celeb_names: dict) -> dict:
    """
    Estimate per-celebrity influence by comparing prediction to baseline
    when that celebrity was in the neighbor group.
    Returns dict: {celebrity_name: mean_shift_per_1k_price}
    """
    name_to_slug = {v: k for k, v in celeb_names.items()}
    influence = {name: [] for name in celeb_names.values()}

    for t in trials:
        pred = t.get("type_a_prediction")
        baseline = t.get("baseline_price")
        if pred is None or baseline is None:
            continue
        shift = pred - baseline
        for neighbor in t.get("neighbors", []):
            name = neighbor.get("name")
            price = neighbor.get("price", 1)
            if name in influence and price:
                # Normalize: shift per $1000 of celebrity price (measures pull per unit)
                influence[name].append(shift / (price / 1000))

    return {name: float(np.mean(vals)) if vals else None
            for name, vals in influence.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", type=str, default=None,
                        help="Comma-separated PIDs (default: all with results)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    celeb_names = load_celebrity_names()
    scores_cache_dir = ANALYSIS_DIR / "recognition_scores"
    scores_cache_dir.mkdir(exist_ok=True)

    # Find personas with trial data
    trial_dir = RESULTS_DIR / "type_ab"
    if not trial_dir.exists():
        trial_dir = RESULTS_DIR / "type_a"

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",")]
    else:
        pids = sorted(
            [f.stem.replace("pid_", "") for f in trial_dir.glob("pid_*.json")],
            key=lambda x: int(x)
        )

    print(f"Running alignment check for {len(pids)} personas...\n")

    all_rows = []

    for pid in pids:
        path = trial_dir / f"pid_{pid}.json"
        if not path.exists():
            continue
        trials = json.loads(path.read_text())

        # Compute behavioral influence per celebrity
        celeb_influence = compute_per_celebrity_influence(pid, trials, celeb_names)

        # Score recognition responses (sample up to 10 trials to save API calls)
        sampled_trials = trials[:10]
        stated_trust_by_celeb: dict[str, list[float]] = {n: [] for n in celeb_names.values()}

        for t in sampled_trials:
            recognition_text = t.get("recognition_response", "")
            if not recognition_text:
                continue
            cache_path = scores_cache_dir / f"pid_{pid}_trial_{t['trial']}.json"
            scores = score_recognition(client, recognition_text, cache_path)
            if scores is None:
                continue
            for celeb_name, score_data in scores.items():
                if celeb_name in stated_trust_by_celeb:
                    trust = score_data.get("trust", 0)
                    stated_trust_by_celeb[celeb_name].append(float(trust))

        # Average stated trust per celebrity for this persona
        for celeb_name in celeb_names.values():
            stated_trust_vals = stated_trust_by_celeb[celeb_name]
            behavioral_influence = celeb_influence.get(celeb_name)
            if not stated_trust_vals or behavioral_influence is None:
                continue
            mean_stated_trust = float(np.mean(stated_trust_vals))
            all_rows.append({
                "pid": pid,
                "celebrity": celeb_name,
                "stated_trust": round(mean_stated_trust, 3),
                "behavioral_influence": round(behavioral_influence, 3),
            })

    if not all_rows:
        print("No alignment data computed. Check that trials have run and recognition responses exist.")
        return

    df = pd.DataFrame(all_rows)

    # --- Overall correlation ---
    corr = df["stated_trust"].corr(df["behavioral_influence"])
    print(f"Overall alignment correlation (stated trust vs behavioral influence): r = {corr:.3f}\n")

    # --- Per-persona alignment ---
    persona_corrs = df.groupby("pid").apply(
        lambda g: g["stated_trust"].corr(g["behavioral_influence"])
    ).reset_index(name="alignment_r")
    print("Per-persona alignment (Pearson r between stated trust and behavioral influence):")
    print(persona_corrs.sort_values("alignment_r", ascending=False).to_string(index=False))

    # --- Per-celebrity alignment ---
    celeb_corrs = df.groupby("celebrity").apply(
        lambda g: g["stated_trust"].corr(g["behavioral_influence"])
    ).reset_index(name="alignment_r")
    print("\nPer-celebrity alignment:")
    print(celeb_corrs.sort_values("alignment_r", ascending=False).to_string(index=False))

    # --- Outliers ---
    df["trust_z"] = (df["stated_trust"] - df["stated_trust"].mean()) / df["stated_trust"].std()
    df["influence_z"] = (df["behavioral_influence"] - df["behavioral_influence"].mean()) / df["behavioral_influence"].std()
    df["misalignment"] = (df["trust_z"] - df["influence_z"]).abs()

    print("\nTop 10 misaligned cases (high trust but low influence, or vice versa):")
    outliers = df.nlargest(10, "misalignment")[
        ["pid", "celebrity", "stated_trust", "behavioral_influence", "misalignment"]
    ]
    print(outliers.to_string(index=False))

    # --- Save ---
    out_path = ANALYSIS_DIR / "alignment_results.csv"
    df.to_csv(out_path, index=False)
    persona_corrs.to_csv(ANALYSIS_DIR / "alignment_per_persona.csv", index=False)
    celeb_corrs.to_csv(ANALYSIS_DIR / "alignment_per_celebrity.csv", index=False)
    print(f"\nSaved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
