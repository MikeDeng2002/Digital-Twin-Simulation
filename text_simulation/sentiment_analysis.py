"""Sentiment scoring for custom questions using OpenAI API.

After each simulation round, scores each persona's answers on a -3 to +3 scale
based on user-defined sentiment labels per question.
"""

import os
import re
import csv
import json
import asyncio
from typing import Dict, List, Tuple, Optional

import openai
import httpx


def parse_sentiment_scales(text: str) -> Dict[str, dict]:
    """Parse Sentiment: lines from custom questions text.

    Returns {q_key: {neg_label, pos_label, question_text}} for each question
    that has a Sentiment: annotation.
    """
    scales = {}
    current_q_key = None
    current_q_text = None

    for line in text.split("\n"):
        # Detect question key like "Q1:" or "Q2:"
        q_match = re.match(r'^(Q\d+):\s*$', line.strip())
        if q_match:
            current_q_key = q_match.group(1)
            current_q_text = None
            continue

        # The line right after "Q<n>:" is the question text
        if current_q_key and current_q_text is None and line.strip() and not line.strip().startswith("Question Type:"):
            current_q_text = line.strip()

        # Parse Sentiment: line
        sent_match = re.match(
            r'^Sentiment:\s*-3\s*=\s*(.+?),\s*\+3\s*=\s*(.+?)\s*$',
            line.strip()
        )
        if sent_match and current_q_key:
            scales[current_q_key] = {
                "neg_label": sent_match.group(1).strip(),
                "pos_label": sent_match.group(2).strip(),
                "question_text": current_q_text or "",
            }

    return scales


def strip_sentiment_lines(text: str) -> str:
    """Remove Sentiment: lines from custom questions text before sending to the main LLM."""
    lines = text.split("\n")
    filtered = [l for l in lines if not re.match(r'^Sentiment:', l.strip())]
    return "\n".join(filtered)


def extract_answer_texts(q_key: str, answer_data: dict, q_body_text: str) -> List[Tuple[str, str]]:
    """Convert a parsed answer into [(sub_key, answer_text)] pairs.

    Handles:
    - Single Choice: ("Q1", "Yes")
    - Matrix: ("Q2_1", "Agree"), ("Q2_2", "Disagree")
    - Slider: ("Q3", "75")
    - Text Entry: ("Q1", "full text answer...")
    """
    if not isinstance(answer_data, dict):
        return [(q_key, str(answer_data))]

    q_type = answer_data.get("Question Type", "")
    answers = answer_data.get("Answers", {})

    if q_type == "Single Choice":
        text = answers.get("SelectedText", "")
        if not text:
            pos = answers.get("SelectedByPosition", "")
            text = str(pos)
        return [(q_key, text)]

    elif q_type == "Matrix":
        selected = answers.get("SelectedText", [])
        if isinstance(selected, list):
            return [(f"{q_key}_{i+1}", str(v)) for i, v in enumerate(selected)]
        return [(q_key, str(selected))]

    elif q_type == "Slider":
        values = answers.get("Values", [])
        if isinstance(values, list):
            return [(q_key, str(values[0])) if values else (q_key, "")]
        return [(q_key, str(values))]

    elif q_type == "Text Entry":
        text = answers.get("Text", "")
        return [(q_key, str(text))]

    else:
        # Fallback: stringify the answers
        return [(q_key, json.dumps(answers))]


def build_sentiment_prompt(question_text: str, answer_text: str,
                           neg_label: str, pos_label: str) -> str:
    """Build the scoring prompt for the sentiment model."""
    return (
        "You are a sentiment scorer. Rate the following answer on a scale from -3 to +3.\n"
        "\n"
        "Scale:\n"
        f"  -3 = {neg_label}\n"
        "  -2 = moderately against\n"
        "  -1 = slightly against\n"
        "   0 = neutral / unclear\n"
        "  +1 = slightly for\n"
        "  +2 = moderately for\n"
        f"  +3 = {pos_label}\n"
        "\n"
        f"Question: {question_text}\n"
        f"Answer: {answer_text}\n"
        "\n"
        "Respond with ONLY a single integer from -3 to 3."
    )


async def score_single(prompt: str, api_key: str, semaphore: asyncio.Semaphore,
                        model: str = "gpt-4.1-nano") -> Optional[int]:
    """Call the sentiment model and parse the integer score. Retries up to 3 times."""
    async with semaphore:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    aclient = openai.AsyncOpenAI(api_key=api_key, http_client=client)
                    response = await aclient.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=5,
                    )
                    text = response.choices[0].message.content.strip()
                    # Parse integer from response
                    match = re.search(r'-?[0-3]', text)
                    if match:
                        score = int(match.group())
                        if -3 <= score <= 3:
                            return score
                    # If parsing failed, retry
                    if attempt < 2:
                        await asyncio.sleep(1)
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"Sentiment scoring failed after 3 attempts: {e}")
    return None


def _load_response_text(round_dir: str, pid: str) -> Optional[dict]:
    """Load and parse a persona's response JSON from a round directory."""
    response_path = os.path.join(round_dir, pid, f"{pid}_response.json")
    if not os.path.exists(response_path):
        return None
    try:
        with open(response_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = data.get("response_text", "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
    except (json.JSONDecodeError, IOError):
        return None


async def score_round_sentiment(
    round_dir: str,
    persona_ids: List[str],
    sentiment_scales: Dict[str, dict],
    model: str = "gpt-4.1-nano",
    max_concurrent: int = 50,
) -> List[dict]:
    """Score all personas' answers for one round.

    Returns list of {persona_id, question, score} dicts.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set, skipping sentiment scoring.")
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []  # (persona_id, sub_key, coroutine)

    for pid in persona_ids:
        answers = _load_response_text(round_dir, pid)
        if answers is None:
            continue

        for q_key, scale_info in sentiment_scales.items():
            answer_data = answers.get(q_key)
            if answer_data is None:
                continue

            pairs = extract_answer_texts(q_key, answer_data, scale_info["question_text"])
            for sub_key, answer_text in pairs:
                if not answer_text:
                    continue
                prompt = build_sentiment_prompt(
                    scale_info["question_text"],
                    answer_text,
                    scale_info["neg_label"],
                    scale_info["pos_label"],
                )
                tasks.append((pid, sub_key, score_single(prompt, api_key, semaphore, model)))

    if not tasks:
        return []

    # Run all scoring calls concurrently
    coroutines = [t[2] for t in tasks]
    results = await asyncio.gather(*coroutines)

    scores = []
    for (pid, sub_key, _), score in zip(tasks, results):
        if score is not None:
            scores.append({"persona_id": pid, "question": sub_key, "score": score})

    return scores


def append_round_to_csv(csv_path: str, round_num: int, scores: List[dict]):
    """Append rows to sentiment_scores.csv.

    CSV columns: round, persona_id, question, score
    Creates the file with header if it doesn't exist.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["round", "persona_id", "question", "score"])
        if not file_exists:
            writer.writeheader()
        for entry in scores:
            writer.writerow({
                "round": round_num,
                "persona_id": entry["persona_id"],
                "question": entry["question"],
                "score": entry["score"],
            })
