# Interaction Simulation

This script (`run_LLM_simulation_interaction.py`) extends the baseline digital twin simulation by adding **social influence via a directed graph**. In each round, a persona can see its neighbors' answers from the previous round before answering, enabling multi-round opinion dynamics.

## Prerequisites

1. **Baseline simulation must be completed first.** The interaction simulation uses the baseline outputs (from `run_LLM_simulations.py`) as Round 0 — the initial answers that neighbors see.

2. **Environment variables.** Make sure your `.env` file contains the API key for your chosen provider (e.g. `OPENAI_API_KEY`).

## Files

| File | Purpose |
|------|---------|
| `run_LLM_simulation_interaction.py` | Main script |
| `configs/interaction_config.yaml` | Configuration (model, paths, rounds) |
| `interaction_graph.json` | Directed graph defining who sees whose answers |
| `custom_questions_example.txt` | Example custom questions file (use as template) |

## Quick Start

From the project root (`Digital-Twin-Simulation/`):

```bash
poetry run python text_simulation/run_LLM_simulation_interaction.py \
  --config text_simulation/configs/interaction_config.yaml \
  --num_rounds 2
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to the YAML config file |
| `--num_rounds` | No | Override the number of rounds set in config |
| `--run_name` | No | Custom name for the output folder (default: `run_<timestamp>`) |
| `--custom_questions` | No | Path to a custom questions text file (overrides config) |

## Configuration (`interaction_config.yaml`)

```yaml
# LLM settings
provider: "openai"
model_name: "gpt-4.1-mini-2025-04-14"
temperature: 0.0
max_tokens: 16384
max_retries: 10        # retries per persona if verification fails
num_workers: 300       # concurrent API requests

# Interaction-specific settings
interaction_graph_path: "text_simulation/interaction_graph.json"
num_rounds: 3
baseline_output_dir: "text_simulation_output"     # where Round 0 answers live
input_folder_dir: "text_simulation_input"          # base prompt files (pid_X_prompt.txt)
output_folder_dir: "text_simulation_output_interaction"  # output root

# System instruction
system_instruction: |
  You are an AI assistant. Your task is to answer the 'New Survey Question'...
  You may also be shown answers from other individuals in your social network.
  Consider their perspectives but remain true to your persona profile.
```

All directory paths are relative to `Digital-Twin-Simulation/text_simulation/`.

To use custom questions, add the path to your config:
```yaml
custom_questions_path: "text_simulation/my_questions.txt"
```
Or pass it via CLI (takes precedence over config):
```bash
--custom_questions text_simulation/my_questions.txt
```

## Interaction Graph (`interaction_graph.json`)

A JSON object where each key is a persona ID and the value is a list of neighbor persona IDs whose answers this persona will see. The graph is **directed** — if `pid_1000` lists `pid_1001` as a neighbor, that does not mean `pid_1001` sees `pid_1000`'s answers (unless explicitly listed).

```json
{
  "pid_1000": ["pid_1001", "pid_1002"],
  "pid_1001": ["pid_1000"],
  "pid_1002": ["pid_1000", "pid_1001"],
  "pid_1003": ["pid_1004"],
  "pid_1004": ["pid_1003"]
}
```

Only personas that appear in the graph (as keys or as neighbors) will be processed. If you want to include a persona that has no neighbors but should still be re-simulated, add it as a key with an empty list: `"pid_999": []`.

## How It Works

```
Round 0 (baseline):  Each persona answers independently (already completed)
                     Outputs live in text_simulation_output/

Round 1:             For each persona P:
                       1. Load P's base prompt (persona profile + 63 questions)
                       2. Look up P's neighbors in the directed graph
                       3. Load neighbors' answers from Round 0
                       4. Inject neighbors' answers into the prompt
                       5. Send to LLM, save & verify response

Round 2:             Same as above, but neighbors' answers come from Round 1

...and so on for R rounds.
```

The neighbor answers are injected between the persona profile section and the survey questions:

```
## Persona Profile (past survey responses):
[existing persona content]

---
## Context: Answers from other individuals in your social network
Below are answers from other people you interact with...

### Answers from Individual (pid_1001):
Q1: {"Question Type": "Matrix", "Answers": {...}}
Q2: ...

---
## New Survey Question & Instructions...
[63 survey questions]
```

## Output Structure

```
text_simulation_output_interaction/
  run_20260222_220246/          # or your custom --run_name
    metadata.json               # graph, config, timestamps, persona list
    round_1/
      pid_1000/
        pid_1000_response.json  # raw LLM response
      pid_1001/
        pid_1001_response.json
      ...
      answer_blocks_llm_imputed/  # verified/postprocessed answer blocks
        pid_1000_wave4_Q_wave4_A.json
        ...
    round_2/
      ...                        # same structure
```

Each `pid_XXXX_response.json` has the same format as the baseline:

```json
{
  "persona_id": "pid_1000",
  "question_id": "pid_1000",
  "prompt_text": "...",
  "response_text": "{\"Q1\": {...}, \"Q2\": {...}, ...}",
  "usage_details": {"prompt_token_count": ..., "completion_token_count": ..., "total_token_count": ...},
  "llm_call_error": null
}
```

## Custom Questions

By default, the simulation uses the 63 survey questions baked into the prompt files. You can replace them with your own questions using a plain text file.

### How to use

1. Create a text file with your questions. Use `custom_questions_example.txt` as a template.
2. Run with `--custom_questions`:

```bash
poetry run python text_simulation/run_LLM_simulation_interaction.py \
  --config text_simulation/configs/interaction_config.yaml \
  --num_rounds 2 \
  --custom_questions text_simulation/my_questions.txt
```

Or set `custom_questions_path` in your config YAML.

### Question file format

Your text file replaces everything after the persona profile. It should include:
- A header line (e.g. `---\n## New Survey Question & Instructions...`)
- Your questions, each labeled Q1, Q2, Q3, etc.
- Each question needs: question text, `Question Type` (Matrix / Single Choice / Slider / Text Entry), options, and `Answer: [Masked]`
- Output format instructions at the end telling the LLM how to structure its JSON response

See `custom_questions_example.txt` for a complete working example with 3 questions.

### What changes with custom questions

- The persona profile is still loaded from the base prompt files (persona background stays the same)
- Your questions replace the original 63-question section
- Postprocess verification is **skipped** (since the original answer block schema won't match your questions). The script only checks that the LLM response is valid JSON.
- Neighbor answer injection still works the same way — neighbors' answers from the previous round are shown between the persona profile and your questions

## Example: Checking for Social Influence

After a run, you can compare how answers shifted across rounds:

```python
import json, os

def load_answers(pid, directory):
    with open(os.path.join(directory, pid, f"{pid}_response.json")) as f:
        return json.loads(json.load(f)["response_text"])

baseline = load_answers("pid_1000", "text_simulation/text_simulation_output")
round1   = load_answers("pid_1000", "text_simulation/text_simulation_output_interaction/run_XXXXX/round_1")

changed = sum(1 for q in baseline if json.dumps(baseline[q]) != json.dumps(round1.get(q, {})))
print(f"{changed}/{len(baseline)} answers changed after seeing neighbors' responses")
```
