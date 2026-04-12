"""
extract_skills.py — Generate skill profiles for a single persona.

Usage (run from Digital-Twin-Simulation/):
    poetry run python skill_extraction/extract_skills.py --pid 1
    poetry run python skill_extraction/extract_skills.py --pid 1 --version v1_direct
    poetry run python skill_extraction/extract_skills.py --pid 1 --force
"""

import os
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

MODEL = "gpt-4.1-mini-2025-04-14"
MAX_TOKENS = 4000

VERSIONS = ["v1_direct", "v2_inferred", "v3_maximum", "v4_chained"]

PERSONA_DIR = Path("text_simulation/text_personas")
OUTPUT_BASE = Path("text_simulation/skills")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = {
    "v1_direct": """\
You are extracting a digital twin skill profile from a survey transcript.

STRICT RULE: Use ONLY questions that directly and explicitly ask about the
topic. Do NOT infer from demographics. Do NOT guess from related answers.
If no direct evidence exists for a component, say so explicitly rather than
fabricating an answer.

The survey covers personality scales (Big Five, Need for Closure, Need for
Cognition, Maximizer, Self-monitoring, Self-concept clarity), values, economic
games, cognitive tests, and demographics.

Output EXACTLY this format with no other text:

---BACKGROUND---
What this person explicitly knows and believes, from direct self-report items
only. Include demographic facts as stated (age, income, education, region) but
do NOT infer knowledge domains from them. Reference the specific scale or
question for each claim.

---TOOLS---
ONLY information sources this person explicitly confirmed they use. If the
transcript has no direct questions about media habits or information sources,
write exactly:
"No direct evidence available. The dataset does not contain direct questions
about media use or information-seeking behavior."
Do not guess or infer.

---DECISION_PROCEDURE---
How this person reasons under uncertainty. Use ONLY direct psychometric scales:
Need for Closure, Need for Cognition, Maximizer, Self-monitoring, Self-concept
clarity. For every claim, cite the specific item number and answer. End with:
w_social estimate: [0.0-1.0] -- [one sentence justification citing the specific
items that support this estimate]

TRANSCRIPT:
{transcript}""",

    "v2_inferred": """\
You are extracting a digital twin skill profile from a survey transcript.

RULES:
- Direct evidence: cite the scale name and item number
- Demographic inference: allowed but label it explicitly as:
  "Inferred from [demographic fact]: [conclusion] because [reasoning]"
- Cross-scale inference: allowed when two or more scales point in the same
  direction -- note which scales agree
- No speculation beyond what the data can reasonably support

The survey covers personality scales (Big Five, Need for Closure, Need for
Cognition, Maximizer, Self-monitoring, Self-concept clarity), values, economic
games, cognitive tests, and demographics.

Output EXACTLY this format with no other text:

---BACKGROUND---
Synthesize direct self-report items on knowledge, values, beliefs, and
experiences. Use demographics to contextualize domain knowledge -- but label
every inference. Cover: stated values and their rankings, Big Five profile
summary, life situation and what it implies about lived knowledge.

---TOOLS---
Infer realistic information sources from education, income, political identity,
need for cognition score, and lifestyle signals. For each source: state the
evidence and reasoning. Also state 2-3 sources this person would NOT use and why.
Acknowledge this section is inferred since the dataset contains no direct
media-use questions.

---DECISION_PROCEDURE---
Use all relevant psychometric scales: Need for Closure, Need for Cognition,
Maximizer, Self-monitoring, Self-concept clarity, and economic game behavior.
Cross-reference scales where they agree. Note any contradictions between scales.
End with:
w_social estimate: [0.0-1.0] -- [2-3 sentence justification citing which scales
and their agreement/disagreement drive this estimate]

TRANSCRIPT:
{transcript}""",

    "v3_maximum": """\
You are extracting the richest possible digital twin skill profile from a survey
transcript. Use every available signal -- direct scales, cognitive test performance
and error patterns, behavioral tasks, word associations, economic game strategy,
loss framing responses, and full demographic context. Infer aggressively but
always state your reasoning.

Goal: produce the most complete, behaviorally predictive skill profile for
simulating this person's numeric predictions on open questions like "What will
Bitcoin's price be next year?" or "What year will AGI arrive?"

The survey covers: Big Five personality (44 items), Need for Cognition (18 items),
Need for Closure (15 items), Maximizer scale (6 items), Self-monitoring (13 items),
Self-concept clarity (12 items), Values (24 items), Minimalism (12 items), Empathy
(20 items), consumerism/uniqueness scales, economic games (dictator + ultimatum),
intertemporal choice tasks, cognitive tests (syllogisms, Wason, analogies), word
association chain, social desirability scale, and demographics.

Output EXACTLY this format with no other text:

---BACKGROUND---
Full synthesis covering: demographics and what they imply about lived experience,
Big Five profile and what it means for how this person engages with information,
values hierarchy (which values scored highest and what this reveals about their
worldview), minimalism and consumerism attitudes, cognitive test performance
including specific error types and what they reveal about reasoning style, word
association chain if it reveals cognitive patterns, social desirability score and
what it implies about self-report reliability. Be specific -- name actual scores
and items.

---TOOLS---
Infer every plausible information source using ALL available signals:
- Education + income: access and familiarity with different sources
- Need for Cognition score: depth of research this person engages in
- Political identity: partisan vs neutral source preference
- Empathy scale: how much they rely on social networks for information
- Maximizer scale: do they comparison-shop information sources?
- Consumerism/uniqueness scale: do they follow trends or ignore them?
- Minimalism scale: do they curate information carefully?
Name actual platforms and source types. Rate likely research depth
(shallow / moderate / deep) with reasoning. List sources they would NOT use.
State confidence level for each inference.

---DECISION_PROCEDURE---
Synthesize ALL behavioral signals into a coherent reasoning profile:
- Need for Closure: how fast do they want resolution?
- Need for Cognition: how much analytical effort do they invest?
- Maximizer: do they seek the objectively best answer or satisfice?
- Self-monitoring: do they notice what others think even if they don't defer?
- Self-concept clarity: is their self-view stable enough to resist social pressure?
- Economic game gap: compare dictator vs ultimatum -- does stated behavior match
  strategic behavior?
- Intertemporal patience: do they think long-term when forming views?
- Cognitive errors: what reasoning failures appeared? What does this imply for
  numeric prediction tasks?
- Social desirability: how much should we trust their self-reports?
End with:
w_social estimate: [0.0-1.0]
Reasoning: [full paragraph explaining which signals drive this estimate, which
signals conflict, and how you resolved the conflicts]

TRANSCRIPT:
{transcript}""",
}

# ---------------------------------------------------------------------------
# v4_chained — three sequential API calls, each building on the previous
# ---------------------------------------------------------------------------

V4_BACKGROUND_PROMPT = """\
You are building a digital twin skill profile from a survey transcript.
This is Step 1 of 3: write the BACKGROUND section only.

Use every available signal: Big Five (44 items), Need for Cognition (18),
Need for Closure (15), Maximizer (6), Self-monitoring (13), Self-concept
clarity (12), Values (24), Minimalism (12), Empathy (20), consumerism/
uniqueness scales, economic games (dictator + ultimatum), intertemporal
choice tasks, cognitive tests (syllogisms, Wason, analogies), word
association chain, social desirability scale, and demographics.

Write ONLY the BACKGROUND section — a dense synthesis of:
- Demographics and what they imply about lived experience
- Big Five profile and how it shapes information engagement
- Values hierarchy and what it reveals about worldview
- Cognitive test performance including specific error types
- Economic game behavior and what it reveals about self-interest vs fairness
- Social desirability score and self-report reliability

Be specific: name actual scores, scale items, and error patterns.
Do not write section headers. Just write the content.

TRANSCRIPT:
{transcript}"""

V4_TOOLS_PROMPT = """\
This is Step 2 of 3: write the TOOLS section only.

You have already written the BACKGROUND for this person. Now infer their
realistic information sources using ALL available signals from the transcript
AND the background you just wrote.

- Education + income: access and familiarity with different sources
- Need for Cognition score: depth of research they engage in
- Political identity: partisan vs neutral source preference
- Empathy scale: reliance on social networks for information
- Maximizer scale: do they comparison-shop information sources?
- Minimalism scale: do they curate information carefully?
- Cognitive errors identified in background: what does poor reasoning imply
  about source quality they can critically evaluate?

Name actual platforms and source types. Rate research depth (shallow /
moderate / deep). List 2-3 sources they would NOT use and why.
State confidence level for each inference.
Acknowledge that no direct media-use questions exist in the dataset.

Do not write section headers. Just write the content."""

V4_DECISION_PROMPT = """\
This is Step 3 of 3: write the DECISION_PROCEDURE section only.

You have already written the BACKGROUND and TOOLS for this person. Now
synthesize ALL behavioral signals into a coherent decision procedure,
drawing explicitly on what you established in those sections.

Cover:
- Need for Closure: how fast do they want resolution?
- Need for Cognition: how much analytical effort do they invest?
- Maximizer: do they seek the objectively best answer or satisfice?
- Self-monitoring: do they notice what others think even if they don't defer?
- Self-concept clarity: is their self-view stable enough to resist social pressure?
- Economic game gap: compare dictator vs ultimatum — does stated behavior
  match strategic behavior? Reconcile with the character in BACKGROUND.
- Intertemporal patience: do they think long-term when forming views?
- Cognitive errors from BACKGROUND: how do these constrain numeric prediction?
- Cross-reference with TOOLS: does their source behavior match their
  stated reasoning style, or is there a contradiction?

End with EXACTLY this format on the last two lines:
w_social estimate: [0.0-1.0]
Reasoning: [full paragraph — which signals drive this estimate, which
conflict, and how you resolved them, referencing specifics from BACKGROUND
and TOOLS]

Do not write section headers. Just write the content."""


def call_api_chained(client: OpenAI, transcript: str) -> dict:
    """Three sequential calls: BACKGROUND → TOOLS → DECISION_PROCEDURE.
    Each call passes the full conversation history forward.
    Returns dict with keys BACKGROUND, TOOLS, DECISION_PROCEDURE.
    """
    # Step 1: BACKGROUND
    messages = [{"role": "user", "content": V4_BACKGROUND_PROMPT.format(transcript=transcript)}]
    resp1 = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0, max_tokens=MAX_TOKENS
    )
    background = resp1.choices[0].message.content.strip()

    # Step 2: TOOLS (append prior turn)
    messages += [
        {"role": "assistant", "content": background},
        {"role": "user", "content": V4_TOOLS_PROMPT},
    ]
    resp2 = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0, max_tokens=MAX_TOKENS
    )
    tools = resp2.choices[0].message.content.strip()

    # Step 3: DECISION_PROCEDURE (append full history)
    messages += [
        {"role": "assistant", "content": tools},
        {"role": "user", "content": V4_DECISION_PROMPT},
    ]
    resp3 = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0, max_tokens=MAX_TOKENS
    )
    decision = resp3.choices[0].message.content.strip()

    return {"BACKGROUND": background, "TOOLS": tools, "DECISION_PROCEDURE": decision}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_transcript(pid: str) -> str:
    # Try both naming conventions
    for name in [f"pid_{pid}.txt", f"pid_{pid}_mega_persona.txt"]:
        path = PERSONA_DIR / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Transcript not found for pid_{pid} in {PERSONA_DIR}")


def parse_sections(response_text: str) -> dict:
    """Split response on ---BACKGROUND---, ---TOOLS---, ---DECISION_PROCEDURE---."""
    sections = {}
    markers = ["BACKGROUND", "TOOLS", "DECISION_PROCEDURE"]
    pattern = r"---(" + "|".join(markers) + r")---"
    parts = re.split(pattern, response_text)

    # parts alternates: [pre, key, content, key, content, ...]
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip()
        content = parts[i + 1].strip()
        sections[key] = content

    return sections


def call_api(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def save_sections(sections: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    name_map = {
        "BACKGROUND": "background.txt",
        "TOOLS": "tools.txt",
        "DECISION_PROCEDURE": "decision_procedure.txt",
    }
    for key, filename in name_map.items():
        content = sections.get(key, "")
        if content:
            (output_dir / filename).write_text(content, encoding="utf-8")
        else:
            print(f"  Warning: section {key} missing from response")


def extract_skills_for_persona(pid: str, versions: list, force: bool = False):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env or export it.")

    client = OpenAI(api_key=api_key)
    transcript = load_transcript(pid)
    print(f"Loaded transcript for pid_{pid} ({len(transcript)} chars)")

    for version in versions:
        output_dir = OUTPUT_BASE / f"pid_{pid}" / version

        # Skip if already done and not forcing
        if not force and all(
            (output_dir / f).exists()
            for f in ["background.txt", "tools.txt", "decision_procedure.txt"]
        ):
            print(f"  [{version}] already exists — skipping (use --force to regenerate)")
            continue

        if version == "v4_chained":
            print(f"  [{version}] calling API (3 chained calls)...")
            sections = call_api_chained(client, transcript)
        else:
            print(f"  [{version}] calling API...")
            prompt = PROMPTS[version].format(transcript=transcript)
            raw = call_api(client, prompt)
            sections = parse_sections(raw)
            if len(sections) != 3:
                print(f"  [{version}] Warning: expected 3 sections, got {len(sections)}: {list(sections.keys())}")
                print(f"  Raw response preview: {raw[:300]}")

        save_sections(sections, output_dir)
        print(f"  [{version}] saved to {output_dir}")

    print(f"Done: pid_{pid}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract skill profiles for one persona.")
    parser.add_argument("--pid", required=True, help="Persona ID (e.g. 1, 42, 1000)")
    parser.add_argument(
        "--version",
        choices=VERSIONS + ["all"],
        default="all",
        help="Which version(s) to generate (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if output files already exist",
    )
    args = parser.parse_args()

    versions_to_run = VERSIONS if args.version == "all" else [args.version]
    extract_skills_for_persona(args.pid, versions_to_run, force=args.force)
