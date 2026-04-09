"""
01_generate_celebrity_data.py — Generate and cache celebrity Bitcoin predictions.

For each celebrity: reads profile.txt + quotes.txt, calls gpt-4.1-mini to
generate both a price prediction and reasoning, stores combined in reasoning.txt.

Usage (from Digital-Twin-Simulation/):
    poetry run python Bitcoin_experiment/scripts/01_generate_celebrity_data.py
    poetry run python Bitcoin_experiment/scripts/01_generate_celebrity_data.py --force
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4.1-mini-2025-04-14"
MAX_TOKENS = 1024

CELEBRITIES_DIR = Path("Bitcoin_experiment/data/celebrities")

BULLS = ["michael_saylor", "cathie_wood", "elon_musk", "jack_dorsey", "larry_fink", "donald_trump"]
BEARS = ["warren_buffett", "peter_schiff", "jamie_dimon", "nouriel_roubini"]

GENERATION_PROMPT = """\
You are simulating {name}'s public prediction for Bitcoin's price at the end of 2025.

Here is who {name} is:
{profile}

Here is what {name} has actually said publicly about Bitcoin:
{quotes}

Based on their documented positions and public statements above, generate:
1. Their specific predicted price for Bitcoin AT THE END OF 2025 (a single realistic dollar number).
   IMPORTANT: Bitcoin currently trades around $85,000. Even the most bearish critics predict a specific
   near-term price (e.g., $5,000-$30,000), not $0. Predict where they think the price will be by
   December 31, 2025 — not where they think it will ultimately end up.
2. Their reasoning in 3-5 sentences, written in first person as {name}, using their actual documented
   arguments and typical information sources.

Format your response EXACTLY as:
PREDICTION: [integer only, no $ or commas, must be greater than 1000]
REASONING: [3-5 sentences in first person as {name}]
"""


def load_celebrity(slug: str) -> dict:
    base = CELEBRITIES_DIR / slug
    profile = (base / "profile.txt").read_text(encoding="utf-8").strip()
    quotes = (base / "quotes.txt").read_text(encoding="utf-8").strip()
    # Extract display name from profile first line
    name_line = profile.split("\n")[0]
    name = name_line.replace("Name:", "").strip()
    title_line = [l for l in profile.split("\n") if l.startswith("Title:")]
    title = title_line[0].replace("Title:", "").strip() if title_line else slug
    return {"slug": slug, "name": name, "title": title, "profile": profile, "quotes": quotes}


def parse_response(text: str) -> tuple[int | None, str | None]:
    """Extract (price, reasoning) from PREDICTION:/REASONING: format."""
    price = None
    reasoning = None
    for line in text.split("\n"):
        if line.startswith("PREDICTION:"):
            raw = line.replace("PREDICTION:", "").strip()
            raw = raw.replace("$", "").replace(",", "").strip()
            try:
                price = int(float(raw))
            except ValueError:
                pass
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    # Reasoning may span multiple lines
    if reasoning is None:
        parts = text.split("REASONING:")
        if len(parts) > 1:
            reasoning = parts[1].strip()
    return price, reasoning


def generate_celebrity(client: OpenAI, celeb: dict, force: bool = False) -> dict:
    out_path = CELEBRITIES_DIR / celeb["slug"] / "reasoning.txt"

    if out_path.exists() and not force:
        print(f"  [{celeb['slug']}] already exists — skipping (use --force to regenerate)")
        content = out_path.read_text(encoding="utf-8")
        price, reasoning = parse_response(content)
        return {"slug": celeb["slug"], "name": celeb["name"], "price": price, "reasoning": reasoning}

    prompt = GENERATION_PROMPT.format(
        name=celeb["name"],
        profile=celeb["profile"],
        quotes=celeb["quotes"],
    )

    print(f"  [{celeb['slug']}] calling API...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    text = response.choices[0].message.content.strip()
    price, reasoning = parse_response(text)

    # Save raw response (contains both PREDICTION and REASONING)
    out_path.write_text(text, encoding="utf-8")
    print(f"  [{celeb['slug']}] saved → {out_path}")

    return {"slug": celeb["slug"], "name": celeb["name"], "price": price, "reasoning": reasoning}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached")
    parser.add_argument("--slug", type=str, default=None, help="Only generate for this celebrity slug")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")
    client = OpenAI(api_key=api_key)

    all_slugs = BULLS + BEARS
    if args.slug:
        all_slugs = [args.slug]

    print(f"Generating celebrity data for {len(all_slugs)} celebrities...\n")
    results = []
    for slug in all_slugs:
        celeb = load_celebrity(slug)
        result = generate_celebrity(client, celeb, force=args.force)
        results.append(result)

    print("\n" + "=" * 60)
    print("CELEBRITY PREDICTIONS SUMMARY")
    print("=" * 60)
    print(f"{'Celebrity':<25} {'Category':<8} {'Prediction':>15}")
    print("-" * 60)
    for r in results:
        category = "BULL" if r["slug"] in BULLS else "BEAR"
        price_str = f"${r['price']:,}" if r["price"] else "PARSE ERROR"
        print(f"{r['name']:<25} {category:<8} {price_str:>15}")
    print("-" * 60)

    print("\nREASONINGS:")
    for r in results:
        print(f"\n[{r['name']}]")
        print(f"  {r['reasoning']}")


if __name__ == "__main__":
    main()
