"""Single Digital Twin + Random Agents Experiment.

Only ONE persona is a real digital twin (answered by LLM). All other n-1 agents
are fake — assigned random sentiment scores from -3 to +3 each round. This tests
how a single digital twin's opinion evolves when exposed to random social signals.
"""

import os
import sys
import json
import argparse
import random
import asyncio
import csv
from datetime import datetime

import yaml
from dotenv import load_dotenv

# Add parent dir so imports work when run from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_helper import LLMConfig, process_prompts_batch
from sentiment_analysis import (
    parse_sentiment_scales,
    strip_sentiment_lines,
    score_round_sentiment,
    append_round_to_csv,
)
from run_LLM_simulation_interaction import (
    extract_demographic_summary,
    extract_persona_section,
)

load_dotenv()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_custom_questions(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def get_ring_pids(graph_path="text_simulation/interaction_graph_sparse_ring.json"):
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    pids = set(graph.keys())
    for neighbors in graph.values():
        pids.update(neighbors)
    return sorted(pids)


def build_context_block(target_prev_response, target_prev_score,
                        agent_summaries, agent_random_scores):
    """Build the context block injected into the prompt for rounds >= 2.

    Shows the sentiment scale, the target's own previous answer + score,
    and each fake agent's demographic summary + random score.
    """
    lines = [
        "---",
        "## Opinions from people in your social network",
        "",
        "Sentiment scale: -3 = strongly against bitcoin, +3 = strongly in favor of bitcoin",
        "",
    ]

    # Own previous answer
    if target_prev_response is not None:
        lines.append(f"[Your previous answer (sentiment score: {target_prev_score})]")
        lines.append(target_prev_response)
        lines.append("")

    # Fake agents
    for pid in sorted(agent_random_scores.keys()):
        summary = agent_summaries.get(pid, pid)
        score = agent_random_scores[pid]
        lines.append(f"[{pid} ({summary}) — sentiment score: {score}]")

    lines.append("")
    lines.append("Consider their perspectives, but answer based on who you are "
                 "as described in your persona profile above.")
    lines.append("---\n")
    return "\n".join(lines)


def save_and_verify_callback(prompt_id, llm_response_data, original_prompt_text, **kwargs):
    """Save LLM response JSON. For custom questions, check response is valid JSON."""
    base_output_dir = kwargs.get("base_output_dir")
    persona_output_folder = os.path.join(base_output_dir, prompt_id)
    os.makedirs(persona_output_folder, exist_ok=True)
    output_path = os.path.join(persona_output_folder, f"{prompt_id}_response.json")

    output_data = {
        "persona_id": prompt_id,
        "prompt_text": original_prompt_text,
        "response_text": llm_response_data.get("response_text", ""),
        "usage_details": llm_response_data.get("usage_details", {}),
        "llm_call_error": llm_response_data.get("error"),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    if llm_response_data.get("error"):
        return False

    # Verify response is valid JSON
    text = llm_response_data.get("response_text", "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        print(f"Response for {prompt_id} is not valid JSON, will retry.")
        return False


def extract_response_text(round_dir, pid):
    """Load the response text from a persona's response JSON."""
    path = os.path.join(round_dir, pid, f"{pid}_response.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data.get("response_text", "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # Try to extract the actual text answer from JSON
    try:
        parsed = json.loads(text)
        if "Q1" in parsed:
            q1 = parsed["Q1"]
            if isinstance(q1, dict):
                answers = q1.get("Answers", {})
                if isinstance(answers, dict) and "Text" in answers:
                    return answers["Text"]
        return text
    except (json.JSONDecodeError, ValueError):
        return text


def get_score_for_round(csv_path, round_num, pid):
    """Read the sentiment score for a specific round and persona from CSV."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["round"]) == round_num and row["persona_id"] == pid:
                return int(row["score"])
    return 0


def plot_sentiment_trajectory(csv_path, output_path, target_pid):
    """Plot the target persona's sentiment score over rounds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    rounds = []
    scores = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["persona_id"] == target_pid:
                rounds.append(int(row["round"]))
                scores.append(int(row["score"]))

    if not rounds:
        print("No sentiment data to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, scores, 'o-', linewidth=2, markersize=6)
    plt.xlabel("Round")
    plt.ylabel("Sentiment Score (-3 to +3)")
    plt.title(f"Sentiment Trajectory for {target_pid}")
    plt.ylim(-3.5, 3.5)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved sentiment trajectory plot: {output_path}")


async def run_experiment(args):
    config = load_config(args.config)
    rng = random.Random(args.seed)

    # Load custom questions
    cq_path = args.custom_questions
    raw_cq = load_custom_questions(cq_path)
    sentiment_scales = parse_sentiment_scales(raw_cq)
    custom_questions_text = strip_sentiment_lines(raw_cq)

    if not sentiment_scales:
        print("Warning: No Sentiment: lines found in custom questions.")

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, config.get("input_folder_dir", "text_simulation_input_simple"))

    # Determine agent PIDs
    if args.agent_pids == "all_ring":
        all_ring = get_ring_pids(
            os.path.join(os.path.dirname(script_dir), "text_simulation", "interaction_graph_sparse_ring.json")
            if not os.path.exists(os.path.join(script_dir, "interaction_graph_sparse_ring.json"))
            else os.path.join(script_dir, "interaction_graph_sparse_ring.json")
        )
        agent_pids = [p for p in all_ring if p != args.target_pid]
    else:
        agent_pids = [p.strip() for p in args.agent_pids.split(",")]

    target_pid = args.target_pid
    all_pids = [target_pid] + agent_pids
    print(f"Target persona: {target_pid}")
    print(f"Fake agents: {len(agent_pids)} personas")

    # Load persona prompts and extract summaries
    base_prompts = {}
    persona_summaries = {}
    for pid in all_pids:
        prompt_path = os.path.join(input_dir, f"{pid}_prompt.txt")
        if not os.path.exists(prompt_path):
            print(f"Warning: prompt file not found for {pid}, skipping.")
            continue
        with open(prompt_path, 'r', encoding='utf-8') as f:
            base_prompts[pid] = f.read()
        persona_summaries[pid] = extract_demographic_summary(base_prompts[pid])

    if target_pid not in base_prompts:
        print(f"Error: target persona {target_pid} prompt file not found.")
        return

    # Remove agents without prompt files
    agent_pids = [p for p in agent_pids if p in base_prompts]
    print(f"Loaded {len(agent_pids)} agent personas + 1 target.")

    print("\nPersona summaries:")
    print(f"  TARGET {target_pid}: {persona_summaries[target_pid]}")
    for pid in sorted(agent_pids):
        print(f"  {pid}: {persona_summaries[pid]}")

    # Output directory
    output_base = os.path.join(script_dir, config.get("output_folder_dir", "text_simulation_output_interaction"))
    run_name = args.run_name or f"single_twin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_base, f"single_twin_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    sentiment_csv_path = os.path.join(run_dir, "sentiment_scores.csv")
    random_scores_path = os.path.join(run_dir, "random_scores.json")
    all_random_scores = {}

    # Provider / LLM config
    provider = config.get("provider", "openai")
    sentiment_model = config.get("sentiment_model", "gpt-4.1-nano")
    sentiment_max_concurrent = config.get("sentiment_max_concurrent", 50)
    num_rounds = args.num_rounds

    # Save metadata
    metadata = {
        "target_pid": target_pid,
        "agent_pids": agent_pids,
        "num_rounds": num_rounds,
        "seed": args.seed,
        "config": {k: v for k, v in config.items() if k != "system_instruction"},
        "custom_questions_path": cq_path,
        "start_time": datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Extract persona section for target
    persona_section, _ = extract_persona_section(base_prompts[target_pid])

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*60}")

        round_dir = os.path.join(run_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Build prompt
        if round_num == 1:
            # Round 1: persona + question only, no context
            prompt = persona_section + custom_questions_text
        else:
            # Rounds 2+: include context block
            prev_round_dir = os.path.join(run_dir, f"round_{round_num - 1}")
            target_prev_response = extract_response_text(prev_round_dir, target_pid)
            target_prev_score = get_score_for_round(sentiment_csv_path, round_num - 1, target_pid)

            # Generate fresh random scores for fake agents
            agent_random = {pid: rng.randint(-3, 3) for pid in agent_pids}
            all_random_scores[str(round_num)] = agent_random

            context = build_context_block(
                target_prev_response, target_prev_score,
                persona_summaries, agent_random,
            )
            prompt = persona_section + context + custom_questions_text

        # LLM config for this round
        verification_args = {
            "base_output_dir": round_dir,
        }
        llm_config = LLMConfig(
            model_name=config.get("model_name"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens"),
            system_instruction=config.get("system_instruction"),
            max_retries=config.get("max_retries", 10),
            max_concurrent_requests=config.get("num_workers", 5),
            verification_callback=save_and_verify_callback,
            verification_callback_args=verification_args,
        )

        # Call LLM for target only
        prompts_to_process = [(target_pid, prompt)]
        results = await process_prompts_batch(
            prompts_to_process, llm_config, provider,
            desc=f"Round {round_num} — LLM call for {target_pid}",
        )

        # Report
        for pid, result in results.items():
            if result.get("error"):
                print(f"  FAILED: {pid}: {result['error']}")
            else:
                print(f"  OK: {pid}")

        # Score sentiment
        if sentiment_scales:
            print(f"Scoring round {round_num} sentiment...")
            round_scores = await score_round_sentiment(
                round_dir, [target_pid], sentiment_scales,
                model=sentiment_model, max_concurrent=sentiment_max_concurrent,
            )
            append_round_to_csv(sentiment_csv_path, round_num, round_scores)
            if round_scores:
                print(f"  Sentiment score: {round_scores[0]['score']}")
            else:
                print("  No sentiment score returned.")

    # Save random scores
    with open(random_scores_path, 'w') as f:
        json.dump(all_random_scores, f, indent=2)

    # Update metadata
    metadata["end_time"] = datetime.now().isoformat()
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Plot sentiment trajectory
    plot_sentiment_trajectory(
        sentiment_csv_path,
        os.path.join(run_dir, "sentiment_trajectory.png"),
        target_pid,
    )

    print(f"\nExperiment complete. Output: {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Single Digital Twin + Random Agents Experiment"
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    parser.add_argument("--target_pid", default="pid_1000",
                        help="The single persona to use as real digital twin (default: pid_1000)")
    parser.add_argument("--agent_pids", default="all_ring",
                        help='Comma-separated fake agent PIDs, or "all_ring" for 30 ring personas minus target')
    parser.add_argument("--num_rounds", type=int, default=30,
                        help="Number of rounds (default: 30)")
    parser.add_argument("--custom_questions", default=None,
                        help="Path to custom questions file (default: custom_questions_bitcoin.txt)")
    parser.add_argument("--run_name", default=None,
                        help="Output directory name suffix")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Default custom questions path
    if args.custom_questions is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.custom_questions = os.path.join(script_dir, "custom_questions_bitcoin.txt")

    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
