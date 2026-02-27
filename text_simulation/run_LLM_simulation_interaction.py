import os
import json
import argparse
import re
import yaml
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from postprocess_responses import postprocess_simulation_outputs_with_pid
from llm_helper import LLMConfig, process_prompts_batch
from sentiment_analysis import (
    parse_sentiment_scales,
    strip_sentiment_lines,
    score_round_sentiment,
    append_round_to_csv,
)

load_dotenv()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data


def load_interaction_graph(path):
    """Load and validate a directed interaction graph from JSON.

    Expected format: {"pid_1000": ["pid_1001", "pid_1002"], ...}
    Each key is a persona ID, values are lists of neighbor persona IDs
    whose answers this persona can see.
    """
    with open(path, 'r') as f:
        graph = json.load(f)

    if not isinstance(graph, dict):
        raise ValueError("Interaction graph must be a JSON object (dict).")

    for pid, neighbors in graph.items():
        if not isinstance(neighbors, list):
            raise ValueError(f"Neighbors for {pid} must be a list, got {type(neighbors).__name__}.")
        for n in neighbors:
            if not isinstance(n, str):
                raise ValueError(f"Neighbor IDs must be strings, got {type(n).__name__} in {pid}'s list.")

    return graph


def load_persona_answers(pid, output_dir):
    """Load the response_text from a persona's response JSON file.

    Returns the parsed JSON answers dict (Q1, Q2, ...) or None if not found/invalid.
    """
    response_path = os.path.join(output_dir, pid, f"{pid}_response.json")
    if not os.path.exists(response_path):
        return None

    try:
        with open(response_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    response_text = data.get("response_text", "")
    if not response_text:
        return None

    # Parse the response_text as JSON
    try:
        # Handle potential markdown code block wrapping
        text = response_text.strip()
        if text.startswith("```"):
            # Remove ```json ... ``` wrapping
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_demographic_summary(base_prompt):
    """Extract a short demographic summary from a persona's prompt text.

    Parses the Q&A pairs at the top of the persona profile and returns a
    one-line string like: "White male, 30-49, West, high school, self-employed,
    moderate, atheist, <$30k"
    """
    # Map question keywords to the field we want to extract
    fields_to_extract = [
        ("What is your race or origin?", "race"),
        ("What is the sex that you were assigned at birth?", "sex"),
        ("How old are you?", "age"),
        ("Which part of the United States do you currently live in?", "region"),
        ("What is the highest level of schooling", "education"),
        ("What is your current employment status?", "employment"),
        ("would you describe your political views", "politics"),
        ("In politics today, do you consider yourself", "party"),
        ("What is your present religion", "religion"),
        ("what was your total family income", "income"),
    ]

    extracted = {}
    for field_question, field_name in fields_to_extract:
        # Find the question, then find the Answer: line after it
        idx = base_prompt.find(field_question)
        if idx == -1:
            continue
        # Search for "Answer:" after this question
        answer_idx = base_prompt.find("Answer:", idx)
        if answer_idx == -1:
            continue
        # Get the rest of the line
        line_end = base_prompt.find("\n", answer_idx)
        if line_end == -1:
            line_end = len(base_prompt)
        answer_line = base_prompt[answer_idx:line_end].strip()
        # Extract just the text after "Answer: N - "
        match = re.match(r'Answer:\s*\d+\s*-\s*(.*)', answer_line)
        if match:
            extracted[field_name] = match.group(1).strip()

    if not extracted:
        return ""

    # Build compact summary
    parts = []
    if "race" in extracted:
        parts.append(extracted["race"])
    if "sex" in extracted:
        parts.append(extracted["sex"].lower())
    if "age" in extracted:
        parts.append(f"age {extracted['age']}")
    if "region" in extracted:
        # Shorten region: "West (WA, OR, ...)" -> "West"
        region = extracted["region"].split("(")[0].strip()
        parts.append(region)
    if "education" in extracted:
        parts.append(extracted["education"])
    if "employment" in extracted:
        parts.append(extracted["employment"].lower())
    if "politics" in extracted:
        parts.append(extracted["politics"].lower())
    if "party" in extracted:
        parts.append(extracted["party"])
    if "religion" in extracted:
        parts.append(extracted["religion"])
    if "income" in extracted:
        parts.append(f"income {extracted['income']}")

    return ", ".join(parts)


def split_questions_section(questions_section):
    """Split the questions section into a header and individual Q blocks.

    Returns (header, q_blocks) where:
        header: the intro text before Q1 (e.g. "---\\n## New Survey Question...\\n---\\n")
        q_blocks: OrderedDict mapping q_key (e.g. "Q1") to the full text block for that question
    """
    from collections import OrderedDict

    # Find where Q1 starts
    q1_match = re.search(r'^Q1:', questions_section, re.MULTILINE)
    if not q1_match:
        return questions_section, OrderedDict()

    header = questions_section[:q1_match.start()]
    body = questions_section[q1_match.start():]

    # Use finditer to locate each Q block boundary
    q_blocks = OrderedDict()
    q_starts = list(re.finditer(r'^Q(\d+):', body, re.MULTILINE))
    for idx, match in enumerate(q_starts):
        q_num = int(match.group(1))
        q_key = f"Q{q_num}"
        start = match.start()
        end = q_starts[idx + 1].start() if idx + 1 < len(q_starts) else len(body)
        q_blocks[q_key] = body[start:end]

    return header, q_blocks


def load_all_neighbor_answers(neighbors, answers_dir):
    """Load answers for all neighbors. Returns dict: neighbor_pid -> answers_dict."""
    result = {}
    for neighbor_pid in neighbors:
        answers = load_persona_answers(neighbor_pid, answers_dir)
        if answers is not None:
            result[neighbor_pid] = answers
    return result


def compact_answer(answer_data):
    """Extract a compact representation of an answer, dropping redundant metadata.

    Instead of: {"Question Type": "Matrix", "Answers": {"SelectedByPosition": [2,3], "SelectedText": ["Somewhat oppose", "Neither"]}}
    Returns:    ["Somewhat oppose", "Neither"]

    Instead of: {"Question Type": "Multiple Choice", "Answers": {"SelectedByPosition": 3, "SelectedText": "Somewhat agree"}}
    Returns:    "Somewhat agree"
    """
    if not isinstance(answer_data, dict):
        return answer_data

    answers = answer_data.get("Answers", answer_data)
    if isinstance(answers, dict):
        # Prefer SelectedText (human-readable) over SelectedByPosition
        if "SelectedText" in answers:
            return answers["SelectedText"]
        if "SelectedByPosition" in answers:
            return answers["SelectedByPosition"]
        # For text entry or other formats, return the answers dict
        return answers
    return answer_data


def build_inline_question_context(q_key, own_prev_answers, neighbor_answers,
                                  persona_summaries):
    """Build inline context for a single question.

    Returns a small text block to prepend before the question, showing only
    that question's compact answer from self (previous round) and neighbors.
    """
    lines = []

    # Own previous answer for this question
    if own_prev_answers and q_key in own_prev_answers:
        compact = compact_answer(own_prev_answers[q_key])
        lines.append(f"[Your previous answer] {json.dumps(compact)}")

    # Neighbor answers for this question
    for neighbor_pid, answers in neighbor_answers.items():
        if q_key in answers:
            summary = ""
            if persona_summaries and neighbor_pid in persona_summaries:
                summary = f" ({persona_summaries[neighbor_pid]})"
            compact = compact_answer(answers[q_key])
            lines.append(f"[{neighbor_pid}{summary}] {json.dumps(compact)}")

    if not lines:
        return ""

    return "[Context from previous round]\n" + "\n".join(lines) + "\n\n"


def format_neighbor_answers_section(pid, neighbors, answers_dir,
                                    persona_summaries=None, own_prev_answers=None):
    """Build the text block showing neighbors' answers to inject into the prompt.

    This is the BLOCK-STYLE version, used only for custom questions (where we can't
    parse individual Q blocks). For standard 63-question prompts, the inline
    per-question injection in build_interaction_prompt() is used instead.
    """
    sections = []

    if own_prev_answers is not None:
        lines = ["## Your own answer from the previous round:"]
        for q_key in sorted(own_prev_answers.keys(), key=lambda k: int(re.sub(r'\D', '', k) or '0')):
            q_data = own_prev_answers[q_key]
            lines.append(f"{q_key}: {json.dumps(q_data)}")
        sections.append("\n".join(lines))

    neighbor_blocks = []
    for neighbor_pid in neighbors:
        answers = load_persona_answers(neighbor_pid, answers_dir)
        if answers is None:
            continue

        summary = ""
        if persona_summaries and neighbor_pid in persona_summaries:
            summary = f" — {persona_summaries[neighbor_pid]}"

        lines = [f"### {neighbor_pid}{summary}:"]
        for q_key in sorted(answers.keys(), key=lambda k: int(re.sub(r'\D', '', k) or '0')):
            q_data = answers[q_key]
            lines.append(f"{q_key}: {json.dumps(q_data)}")
        neighbor_blocks.append("\n".join(lines))

    if not neighbor_blocks and not sections:
        return ""

    header = (
        "---\n"
        "## Opinions from people in your social network\n"
        "You have heard opinions from other people in your social network. "
        "Each person comes from a different background — their demographic "
        "information (race, gender, age, region, education, employment, "
        "political views, party, religion, income) is shown next to their answer.\n"
        "Consider their perspectives, but answer based on who you are "
        "as described in your persona profile above.\n\n"
    )

    if neighbor_blocks:
        sections.append("\n\n".join(neighbor_blocks))

    return header + "\n\n".join(sections) + "\n\n"


def extract_persona_section(base_prompt):
    """Extract the persona profile section from a base prompt.

    Returns (persona_section, questions_section). The persona_section ends just
    before the '---\\n## New Survey Question' marker.
    """
    split_marker = "---\n## New Survey Question"
    split_idx = base_prompt.find(split_marker)

    if split_idx == -1:
        for variant in ["---\r\n## New Survey Question", "---\n\n## New Survey Question"]:
            split_idx = base_prompt.find(variant)
            if split_idx != -1:
                split_marker = variant
                break

    if split_idx == -1:
        return base_prompt, ""

    return base_prompt[:split_idx], base_prompt[split_idx:]


def build_interaction_prompt(base_prompt, pid, neighbors, answers_dir,
                             custom_questions_text=None,
                             persona_summaries=None, own_prev_answers=None):
    """Split the base prompt and inject context per-question inline.

    For standard prompts (63 questions): injects each question's specific
    neighbor/self answer right before that question block, saving tokens.

    For custom questions: falls back to block-style injection since we can't
    parse custom question format.
    """
    persona_section, questions_section = extract_persona_section(base_prompt)

    if not questions_section and not custom_questions_text:
        print(f"Warning: Could not find split marker in prompt for {pid}. "
              "Appending neighbor answers before the entire prompt.")
        neighbor_section = format_neighbor_answers_section(
            pid, neighbors, answers_dir,
            persona_summaries=persona_summaries,
            own_prev_answers=own_prev_answers,
        )
        if neighbor_section:
            return neighbor_section + base_prompt
        return base_prompt

    # Custom questions: use block-style injection (can't parse Q blocks)
    if custom_questions_text is not None:
        neighbor_section = format_neighbor_answers_section(
            pid, neighbors, answers_dir,
            persona_summaries=persona_summaries,
            own_prev_answers=own_prev_answers,
        )
        return persona_section + neighbor_section + custom_questions_text

    # Standard questions: per-question inline injection
    has_context = bool(neighbors) or (own_prev_answers is not None)
    if not has_context:
        return persona_section + questions_section

    # Load all neighbor answers once
    neighbor_answers = load_all_neighbor_answers(neighbors, answers_dir)

    # Parse questions into individual blocks
    header, q_blocks = split_questions_section(questions_section)

    if not q_blocks:
        # Couldn't parse Q blocks, fall back to block-style
        neighbor_section = format_neighbor_answers_section(
            pid, neighbors, answers_dir,
            persona_summaries=persona_summaries,
            own_prev_answers=own_prev_answers,
        )
        return persona_section + neighbor_section + questions_section

    # Add a brief instruction note in the header
    context_note = (
        "\nNote: For each question below, you may see context from your previous answer "
        "and/or answers from individuals in your social network. Consider their perspectives "
        "but remain true to your persona profile.\n\n"
    )

    # Rebuild questions section with inline context per question
    rebuilt_parts = [header + context_note]
    for q_key, q_text in q_blocks.items():
        inline_ctx = build_inline_question_context(
            q_key, own_prev_answers, neighbor_answers, persona_summaries
        )
        rebuilt_parts.append(inline_ctx + q_text)

    return persona_section + "".join(rebuilt_parts)


def get_output_path(base_output_dir, persona_id):
    """Get the output path for a persona's response JSON."""
    persona_output_folder = os.path.join(base_output_dir, persona_id)
    os.makedirs(persona_output_folder, exist_ok=True)
    return os.path.join(persona_output_folder, f"{persona_id}_response.json")


def save_and_verify_callback(prompt_id: str, llm_response_data: dict, original_prompt_text: str, **kwargs) -> bool:
    """Save LLM response and verify via postprocessing.

    Same pattern as run_LLM_simulations.py but with round-specific output dir.
    When skip_verification=True (custom questions mode), saves the response
    and checks that response_text is parseable JSON, but skips postprocess verification.
    """
    base_output_dir = kwargs.get("base_output_dir")
    question_json_base_dir = kwargs.get("question_json_base_dir")
    output_updated_questions_dir = kwargs.get("output_updated_questions_dir_for_verify")
    skip_verification = kwargs.get("skip_verification", False)

    if not base_output_dir:
        print(f"Error for {prompt_id}: Missing base_output_dir in verification_callback_args.")
        return False

    persona_id = prompt_id
    output_path = get_output_path(base_output_dir, persona_id)

    output_json_data = {
        "persona_id": persona_id,
        "question_id": persona_id,
        "prompt_text": original_prompt_text,
        "response_text": llm_response_data.get("response_text", ""),
        "usage_details": llm_response_data.get("usage_details", {}),
        "llm_call_error": llm_response_data.get("error")
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=2)
    except Exception as e:
        print(f"Error writing output file {output_path} for {prompt_id}: {e}")
        return False

    if "error" in llm_response_data and llm_response_data["error"]:
        return False

    if skip_verification:
        # For custom questions: just check response_text is valid JSON
        response_text = llm_response_data.get("response_text", "")
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            print(f"Custom questions: response for {prompt_id} is not valid JSON, will retry.")
            return False

    try:
        is_verified = postprocess_simulation_outputs_with_pid(
            persona_id,
            base_output_dir,
            question_json_base_dir,
            output_updated_questions_dir
        )
        return is_verified
    except Exception as e:
        print(f"Error during verification for persona {prompt_id}: {e}")
        return False


async def run_interaction_simulation(
    config,
    interaction_graph,
    num_rounds,
    run_output_dir,
    input_folder_dir,
    baseline_output_dir,
    custom_questions_text=None,
    include_prev_answers=True,
    sentiment_scales=None,
    start_round=1,
):
    """Main async loop: run multi-round interaction simulation.

    If custom_questions_text is provided, it replaces the original 63-question
    section for all personas, and postprocess verification is skipped (since the
    answer blocks schema won't match custom questions).
    """
    provider = config.get('provider', 'openai')
    num_workers = config.get('num_workers', 5)
    max_retries = config.get('max_retries', 10)
    question_json_base_dir = "./data/mega_persona_json/answer_blocks"
    use_custom_questions = custom_questions_text is not None

    # Determine which persona IDs to process (intersection of graph keys and available prompts)
    graph_pids = set(interaction_graph.keys())
    # Also include personas that appear only as neighbors
    all_pids_in_graph = set(graph_pids)
    for neighbors in interaction_graph.values():
        all_pids_in_graph.update(neighbors)

    # Find available prompt files
    available_prompts = {}
    for fname in sorted(os.listdir(input_folder_dir)):
        if fname.endswith('_prompt.txt'):
            match = re.search(r'(pid_\d+)', fname)
            if match:
                pid = match.group(1)
                if pid in all_pids_in_graph:
                    available_prompts[pid] = os.path.join(input_folder_dir, fname)

    if not available_prompts:
        print("No prompt files found for personas in the interaction graph.")
        return

    print(f"Found {len(available_prompts)} personas with prompts in the interaction graph.")
    if use_custom_questions:
        print("Using custom questions (postprocess verification disabled).")

    # Load base prompts once and extract demographic summaries
    base_prompts = {}
    persona_summaries = {}
    for pid, path in available_prompts.items():
        with open(path, 'r', encoding='utf-8') as f:
            base_prompts[pid] = f.read()
        persona_summaries[pid] = extract_demographic_summary(base_prompts[pid])

    print("Persona summaries:")
    for pid in sorted(persona_summaries):
        print(f"  {pid}: {persona_summaries[pid]}")

    # Previous round's output directory
    if start_round > 1:
        prev_round_dir = os.path.join(run_output_dir, f"round_{start_round - 1}")
    else:
        prev_round_dir = baseline_output_dir

    metadata_path = os.path.join(run_output_dir, "metadata.json")

    if start_round == 1:
        # Save metadata for a fresh run
        metadata = {
            "interaction_graph": interaction_graph,
            "config": {k: v for k, v in config.items() if k != 'system_instruction'},
            "system_instruction": config.get('system_instruction', ''),
            "num_rounds": num_rounds,
            "personas": sorted(available_prompts.keys()),
            "baseline_output_dir": baseline_output_dir,
            "custom_questions": use_custom_questions,
            "custom_questions_path": config.get('custom_questions_path'),
            "start_time": datetime.now().isoformat(),
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        print(f"Resuming from round {start_round} (previous round dir: {prev_round_dir})")

    # Sentiment scoring setup
    sentiment_csv_path = os.path.join(run_output_dir, "sentiment_scores.csv")
    sentiment_model = config.get("sentiment_model", "gpt-4.1-nano")
    sentiment_max_concurrent = config.get("sentiment_max_concurrent", 50)
    persona_id_list = sorted(available_prompts.keys())

    # Score baseline (round 0) if sentiment is enabled — only for fresh runs
    if sentiment_scales and start_round == 1:
        print("\nScoring baseline (round 0) sentiment...")
        baseline_scores = await score_round_sentiment(
            baseline_output_dir, persona_id_list, sentiment_scales,
            model=sentiment_model, max_concurrent=sentiment_max_concurrent,
        )
        append_round_to_csv(sentiment_csv_path, 0, baseline_scores)
        print(f"Baseline sentiment: {len(baseline_scores)} scores recorded.")

    for round_num in range(start_round, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*60}")

        round_dir = os.path.join(run_output_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        answer_blocks_dir = os.path.join(round_dir, "answer_blocks_llm_imputed")
        os.makedirs(answer_blocks_dir, exist_ok=True)

        verification_args = {
            "base_output_dir": round_dir,
            "question_json_base_dir": question_json_base_dir,
            "output_updated_questions_dir_for_verify": answer_blocks_dir,
            "skip_verification": use_custom_questions,
        }

        llm_config = LLMConfig(
            model_name=config.get('model_name'),
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens'),
            system_instruction=config.get('system_instruction'),
            max_retries=max_retries,
            max_concurrent_requests=num_workers,
            verification_callback=save_and_verify_callback,
            verification_callback_args=verification_args,
        )

        # Build prompts for this round
        prompts_to_process = []
        for pid in sorted(available_prompts.keys()):
            neighbors = interaction_graph.get(pid, [])

            # Load this persona's own previous-round answers (unless disabled)
            own_prev_answers = None
            if include_prev_answers:
                own_prev_answers = load_persona_answers(pid, prev_round_dir)

            if neighbors or own_prev_answers:
                prompt = build_interaction_prompt(
                    base_prompts[pid], pid, neighbors, prev_round_dir,
                    custom_questions_text=custom_questions_text,
                    persona_summaries=persona_summaries,
                    own_prev_answers=own_prev_answers,
                )
            else:
                # No neighbors and no previous answers — still need to swap questions if custom
                if use_custom_questions:
                    persona_section, _ = extract_persona_section(base_prompts[pid])
                    prompt = persona_section + custom_questions_text
                else:
                    prompt = base_prompts[pid]

            prompts_to_process.append((pid, prompt))

        print(f"Processing {len(prompts_to_process)} personas for round {round_num}...")

        final_results = await process_prompts_batch(
            prompts_to_process,
            llm_config,
            provider,
            desc=f"Round {round_num} - {provider.capitalize()} LLM calls & Verification"
        )

        # Report results
        success_count = 0
        fail_count = 0
        for prompt_id, result_data in final_results.items():
            if "error" in result_data and result_data["error"]:
                fail_count += 1
                print(f"  FAILED: {prompt_id}: {result_data['error']}")
            else:
                success_count += 1

        print(f"Round {round_num} complete: {success_count} succeeded, {fail_count} failed.")

        # Score sentiment for this round
        if sentiment_scales:
            print(f"Scoring round {round_num} sentiment...")
            round_scores = await score_round_sentiment(
                round_dir, persona_id_list, sentiment_scales,
                model=sentiment_model, max_concurrent=sentiment_max_concurrent,
            )
            append_round_to_csv(sentiment_csv_path, round_num, round_scores)
            print(f"Round {round_num} sentiment: {len(round_scores)} scores recorded.")

        # This round's output becomes next round's input
        prev_round_dir = round_dir

    # Update metadata with end time and final num_rounds
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
    metadata["num_rounds"] = num_rounds
    metadata["end_time"] = datetime.now().isoformat()
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll {num_rounds} rounds complete. Output saved to: {run_output_dir}")


def find_last_completed_round(run_dir):
    """Find the highest round N where round_N directory exists and has response JSONs.

    Returns 0 if no valid round directories found.
    """
    max_round = 0
    if not os.path.isdir(run_dir):
        return 0
    for entry in os.listdir(run_dir):
        match = re.match(r'^round_(\d+)$', entry)
        if not match:
            continue
        round_num = int(match.group(1))
        round_path = os.path.join(run_dir, entry)
        # Check that it contains at least one persona response JSON
        has_response = False
        for sub in os.listdir(round_path):
            sub_path = os.path.join(round_path, sub)
            if os.path.isdir(sub_path):
                response_file = os.path.join(sub_path, f"{sub}_response.json")
                if os.path.exists(response_file):
                    has_response = True
                    break
        if has_response and round_num > max_round:
            max_round = round_num
    return max_round


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM interaction simulations with social influence via directed graph."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--num_rounds", type=int, help="Override number of rounds from config.")
    parser.add_argument("--run_name", type=str, help="Custom run name (default: run_<timestamp>).")
    parser.add_argument("--custom_questions", type=str,
                        help="Path to a custom questions text file. Overrides custom_questions_path in config.")
    parser.add_argument("--no_prev_answers", action="store_true",
                        help="Do not include own previous-round answers in the prompt.")
    parser.add_argument("--no_sentiment", action="store_true",
                        help="Disable per-question sentiment scoring even if Sentiment: lines are present.")
    parser.add_argument("--continue_from", type=str,
                        help="Path to existing run directory to continue from. Resumes after last completed round.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve paths (run from project root: Digital-Twin-Simulation/)
    input_folder_dir = os.path.join("./text_simulation", config.get('input_folder_dir', 'text_simulation_input'))
    baseline_output_dir = os.path.join("./text_simulation", config.get('baseline_output_dir', 'text_simulation_output'))
    output_base_dir = os.path.join("./text_simulation", config.get('output_folder_dir', 'text_simulation_output_interaction'))
    graph_path = config.get('interaction_graph_path', 'text_simulation/interaction_graph.json')

    num_rounds = args.num_rounds or config.get('num_rounds', 3)

    # Load custom questions if specified (CLI arg takes precedence over config)
    custom_questions_path = args.custom_questions or config.get('custom_questions_path')
    custom_questions_text = None
    sentiment_scales = None
    if custom_questions_path:
        print(f"Loading custom questions from: {custom_questions_path}")
        with open(custom_questions_path, 'r', encoding='utf-8') as f:
            raw_custom_text = f.read()

        # Parse sentiment scales before stripping
        if not args.no_sentiment:
            sentiment_scales = parse_sentiment_scales(raw_custom_text)
            if sentiment_scales:
                print(f"Found sentiment scales for {len(sentiment_scales)} question(s): {list(sentiment_scales.keys())}")
            else:
                print("No Sentiment: lines found in custom questions.")

        # Strip Sentiment: lines from the text sent to the main LLM
        custom_questions_text = strip_sentiment_lines(raw_custom_text)
        print(f"Custom questions loaded ({len(custom_questions_text)} chars).")

    # Load and validate interaction graph
    print(f"Loading interaction graph from: {graph_path}")
    interaction_graph = load_interaction_graph(graph_path)
    print(f"Graph has {len(interaction_graph)} personas with defined neighbors.")

    # Handle --continue_from or create a new run directory
    start_round = 1
    if args.continue_from:
        # Resolve path: bare name -> prepend output_base_dir; absolute -> use as-is
        continue_path = args.continue_from
        if not os.path.isabs(continue_path):
            candidate = os.path.join(output_base_dir, continue_path)
            if os.path.isdir(candidate):
                continue_path = candidate
            elif not os.path.isdir(continue_path):
                continue_path = candidate  # fall through, will error below

        if not os.path.isdir(continue_path):
            print(f"Error: --continue_from directory does not exist: {continue_path}")
            exit(1)

        run_output_dir = continue_path

        # Load metadata to recover baseline_output_dir
        meta_path = os.path.join(run_output_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                prev_metadata = json.load(f)
            baseline_output_dir = prev_metadata.get("baseline_output_dir", baseline_output_dir)
            print(f"Loaded metadata from previous run. Baseline: {baseline_output_dir}")
        else:
            print(f"Warning: No metadata.json found in {run_output_dir}, using baseline from config.")

        last_completed = find_last_completed_round(run_output_dir)
        start_round = last_completed + 1
        print(f"Last completed round: {last_completed}. Will resume from round {start_round}.")

        if start_round > num_rounds:
            print(f"Already completed {last_completed} rounds, target is {num_rounds}. Nothing to do.")
            exit(0)
    else:
        run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_output_dir = os.path.join(output_base_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)

    print(f"Starting interaction simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Provider: {config.get('provider')}, Model: {config.get('model_name')}")
    print(f"Input prompts: {input_folder_dir}")
    print(f"Baseline outputs: {baseline_output_dir}")
    print(f"Interaction output: {run_output_dir}")
    include_prev_answers = not args.no_prev_answers
    print(f"Number of rounds: {num_rounds}")
    if start_round > 1:
        print(f"Resuming from round: {start_round}")
    print(f"Include own previous answers: {include_prev_answers}")
    if custom_questions_text:
        print(f"Custom questions: {custom_questions_path}")
    if sentiment_scales:
        print(f"Sentiment scoring: enabled ({len(sentiment_scales)} question(s), model={config.get('sentiment_model', 'gpt-4.1-nano')})")
    else:
        print("Sentiment scoring: disabled")

    asyncio.run(run_interaction_simulation(
        config=config,
        interaction_graph=interaction_graph,
        num_rounds=num_rounds,
        run_output_dir=run_output_dir,
        input_folder_dir=input_folder_dir,
        baseline_output_dir=baseline_output_dir,
        custom_questions_text=custom_questions_text,
        include_prev_answers=include_prev_answers,
        sentiment_scales=sentiment_scales,
        start_round=start_round,
    ))
