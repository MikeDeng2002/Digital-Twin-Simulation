"""
reasoning_token_summary.py — Summarize reasoning token usage across all configs.

Usage (from Digital-Twin-Simulation/):
    python evaluation/reasoning_token_summary.py --suite nano_temp0
    python evaluation/reasoning_token_summary.py --suite mini_temp0
    python evaluation/reasoning_token_summary.py --suite nano_temp0 --reasoning high
"""

import json, re, argparse
import pandas as pd
from pathlib import Path

SUITES = {
    "nano_temp0": Path("text_simulation/text_simulation_output_nano_temp0"),
    "mini_temp0": Path("text_simulation/text_simulation_output_mini_temp0"),
}

SETTINGS = [
    "skill_v1", "skill_v2", "skill_v3", "raw",
    "raw_start_v1", "raw_start_v2", "raw_start_v3",
    "skill_v1_raw_end", "skill_v2_raw_end", "skill_v3_raw_end",
]
REASONING_LEVELS = ["none", "low", "medium", "high"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",     required=True, choices=list(SUITES.keys()))
    parser.add_argument("--setting",   default=None)
    parser.add_argument("--reasoning", default=None)
    args = parser.parse_args()

    base_dir = SUITES[args.suite]
    settings   = [args.setting]   if args.setting   else SETTINGS
    reasonings = [args.reasoning] if args.reasoning else REASONING_LEVELS

    rows = []
    for setting in settings:
        for reasoning in reasonings:
            trial_dir = base_dir / setting / reasoning
            if not trial_dir.exists():
                continue

            files = sorted(trial_dir.rglob("pid_*_response.json"),
                           key=lambda f: int(re.search(r"pid_(\d+)", f.name).group(1)))
            for f in files:
                try:
                    d = json.loads(f.read_text())
                    ud = d.get("usage_details", {})
                    rows.append({
                        "setting":              setting,
                        "reasoning":            reasoning,
                        "persona":              d.get("persona_id", ""),
                        "response_status":      d.get("response_status", "unknown"),
                        "incomplete_reason":    d.get("incomplete_reason"),
                        "prompt_tokens":        ud.get("prompt_token_count", 0),
                        "reasoning_tokens":     ud.get("reasoning_tokens", 0),
                        "actual_output_tokens": ud.get("actual_output_tokens", 0),
                        "total_tokens":         ud.get("total_token_count", 0),
                        "response_text_len":    len(d.get("response_text", "")),
                    })
                except Exception:
                    continue

    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows)

    # Per-config summary
    print(f"\n{'='*80}")
    print(f"REASONING TOKEN SUMMARY — {args.suite.upper()}")
    print(f"{'='*80}")

    summary = df.groupby(["setting", "reasoning"]).agg(
        n_personas        = ("persona", "count"),
        n_complete        = ("response_status", lambda x: (x == "completed").sum()),
        n_incomplete      = ("response_status", lambda x: (x == "incomplete").sum()),
        avg_reasoning_tok = ("reasoning_tokens", "mean"),
        max_reasoning_tok = ("reasoning_tokens", "max"),
        avg_output_tok    = ("actual_output_tokens", "mean"),
    ).round(0)

    for (setting, reasoning), row in summary.iterrows():
        flag = " ⚠" if row["n_incomplete"] > 0 else ""
        print(f"\n{setting} / {reasoning}:{flag}")
        print(f"  personas: {int(row['n_personas'])}  complete: {int(row['n_complete'])}  incomplete: {int(row['n_incomplete'])}")
        print(f"  reasoning tokens — avg: {int(row['avg_reasoning_tok']):,}  max: {int(row['max_reasoning_tok']):,}")
        print(f"  actual output tokens — avg: {int(row['avg_output_tok']):,}")

    # Pivot table: avg reasoning tokens by setting × reasoning
    print(f"\n{'='*80}")
    print("AVG REASONING TOKENS (setting × reasoning level)")
    print(f"{'='*80}")
    pivot = df.groupby(["setting", "reasoning"])["reasoning_tokens"].mean().unstack("reasoning")
    col_order = [c for c in REASONING_LEVELS if c in pivot.columns]
    pivot = pivot[col_order].reindex([s for s in SETTINGS if s in pivot.index])
    print(pivot.round(0).astype("Int64").to_string())

    # Incomplete summary
    incomplete = df[df["response_status"] == "incomplete"]
    if not incomplete.empty:
        print(f"\n{'='*80}")
        print(f"INCOMPLETE RESPONSES ({len(incomplete)} total)")
        print(f"{'='*80}")
        for _, row in incomplete.iterrows():
            print(f"  {row['setting']:25s} {row['reasoning']:8s} {row['persona']}  "
                  f"reasoning_tok={int(row['reasoning_tokens']):,}  "
                  f"output_tok={int(row['actual_output_tokens'])}  "
                  f"reason={row['incomplete_reason']}")

    # Save CSV
    out_csv = Path(f"evaluation/{args.suite}_reasoning_tokens.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
