# LLM Interaction Simulation — Full Guide

This document explains how to run the multi-round interaction simulation with custom questions and per-question sentiment scoring.

## Overview

The interaction simulation (`run_LLM_simulation_interaction.py`) lets you:

1. Define **custom survey questions** (e.g. "What do you think about bitcoin?")
2. Simulate **multi-round social influence** — each persona sees other personas' answers from the previous round before answering again
3. Automatically **score each answer's sentiment** on a -3 to +3 scale using a lightweight LLM

## Prerequisites

1. **Python environment**: `poetry install` from the project root (`Digital-Twin-Simulation/`)
2. **API key**: Set `OPENAI_API_KEY=sk-...` in a `.env` file at the project root
3. **Baseline outputs**: New personas need a baseline (Round 0) run first — see [Generating Baselines](#generating-baselines)

All commands below should be run from the project root: `Digital-Twin-Simulation/`

---

## Quick Start

```bash
poetry run python text_simulation/run_LLM_simulation_interaction.py \
  --config text_simulation/configs/interaction_config.yaml \
  --num_rounds 10 \
  --custom_questions text_simulation/custom_questions_bitcoin.txt \
  --run_name my_experiment
```

This runs 10 rounds of interaction with the bitcoin opinion question across all personas defined in the interaction graph.

---

## Files

| File | Purpose |
|------|---------|
| `run_LLM_simulation_interaction.py` | Main simulation script |
| `sentiment_analysis.py` | Sentiment scoring module |
| `configs/interaction_config.yaml` | Configuration (model, paths, rounds, sentiment) |
| `interaction_graph.json` | Directed graph — who sees whose answers |
| `custom_questions_bitcoin.txt` | Example: bitcoin opinion question with sentiment |
| `custom_questions_with_sentiment.txt` | Example: 3 questions (Single Choice, Matrix, Slider) with sentiment |
| `custom_questions_example.txt` | Example: 3 questions without sentiment |
| `custom_questions_simple.txt` | Example: simple healthcare yes/no question |
| `text_simulation_input/` | Full persona profiles (~2,951 lines each, 63 survey questions + answers) |
| `text_simulation_input_simple/` | Simplified persona profiles (~144 lines each, demographics only) |

---

## Persona Input Folders

Two versions of persona profiles are available:

### `text_simulation_input/` — Full profiles (default)

Each file contains ~2,951 lines with all 63 survey questions and answers, including demographics, personality traits, political opinions, and more. This gives the LLM rich context to role-play the persona but results in **large prompts** (~27k tokens per persona in a 15-persona fully connected graph).

### `text_simulation_input_simple/` — Demographics only

Each file contains ~144 lines with only the **14 demographic questions**:

| Question | Example Answer |
|----------|---------------|
| Region | West |
| Sex | Male |
| Age | 30-49 |
| Education | High school graduate |
| Race | White |
| Citizenship | Yes |
| Marital status | Never been married |
| Religion | Atheist |
| Religious attendance | Never |
| Political party | Something else |
| Family income | Less than $30,000 |
| Political views | Moderate |
| Household size | 3 |
| Employment status | Self-employed |

**Benefits of using simplified profiles:**
- **Much smaller prompts** — faster and cheaper API calls
- **Higher `num_workers`** — less likely to hit rate limits
- **Better for large-scale experiments** — can run more personas concurrently

**Trade-off:** The LLM has less context to role-play the persona. Answers will be based only on demographic identity, not personality traits or detailed political opinions.

### Switching between input folders

Set `input_folder_dir` in your config YAML:

```yaml
# Use full profiles (default)
input_folder_dir: "text_simulation_input"

# Use simplified demographics-only profiles
input_folder_dir: "text_simulation_input_simple"
```

### Example: Running with simplified profiles

```bash
poetry run python text_simulation/run_LLM_simulation_interaction.py \
  --config text_simulation/configs/interaction_config_simple.yaml \
  --num_rounds 10 \
  --custom_questions text_simulation/custom_questions_bitcoin.txt \
  --run_name bitcoin_simple_profiles
```

Where `interaction_config_simple.yaml` has `input_folder_dir: "text_simulation_input_simple"`. You can also use higher `num_workers` (e.g., 50-300) since prompts are much smaller.

---

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to the YAML config file |
| `--num_rounds` | No | From config (3) | Number of interaction rounds |
| `--run_name` | No | `run_<timestamp>` | Name for the output directory |
| `--custom_questions` | No | From config | Path to a custom questions text file |
| `--no_prev_answers` | No | False | Don't show personas their own previous-round answers |
| `--no_sentiment` | No | False | Disable sentiment scoring even if `Sentiment:` lines are present |

---

## Configuration (`interaction_config.yaml`)

```yaml
# LLM settings
provider: "openai"                    # LLM provider ("openai" or "gemini")
model_name: "gpt-4.1-mini-2025-04-14" # Model for generating persona answers
temperature: 0.0                       # 0.0 = deterministic answers
max_tokens: 16384                      # Max output tokens per LLM call
max_retries: 10                        # Retries per persona if LLM call or verification fails
num_workers: 300                       # Max concurrent API requests (see Rate Limits below)

# Interaction-specific settings
interaction_graph_path: "text_simulation/interaction_graph.json"
num_rounds: 3                          # Default rounds (overridden by --num_rounds)
baseline_output_dir: "text_simulation_output"         # Where Round 0 answers live
input_folder_dir: "text_simulation_input"              # Base prompt files — use "text_simulation_input_simple" for demographics-only
output_folder_dir: "text_simulation_output_interaction" # Output root directory

# Processing options
force_regenerate: false                # Re-generate even if output already exists

# Sentiment scoring
sentiment_model: "gpt-4.1-nano"       # Cheap/fast model for scoring sentiment
sentiment_max_concurrent: 50           # Max concurrent sentiment API calls

# System instruction (sent as system message to the LLM)
system_instruction: |
  You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described in the 'Persona Profile'...
```

### Parameter Details

| Parameter | What It Does | Recommended Value |
|-----------|-------------|-------------------|
| `num_workers` | Controls how many personas are queried concurrently. Higher = faster but may hit rate limits. | **3-5** for 15+ personas, **50-300** for 5 personas |
| `max_retries` | Total attempts per persona per round (includes LLM call + response verification). | 10 |
| `temperature` | LLM randomness. 0.0 = deterministic, same persona gives same answer every time. | 0.0 for reproducibility |
| `sentiment_model` | Model used for scoring sentiment after each round. Should be cheap/fast. | `gpt-4.1-nano` |
| `sentiment_max_concurrent` | Concurrent sentiment scoring calls. Sentiment calls are tiny (~100 tokens each). | 50 |

### Rate Limits

With **full profiles** (`text_simulation_input`), each prompt for a fully-connected 15-persona graph is ~27k tokens. With `num_workers: 5`, that's up to 5 x 27k = 135k tokens per minute, which is under the 200k TPM limit for most OpenAI tiers.

With **simplified profiles** (`text_simulation_input_simple`), prompts are much smaller (~2-5k tokens), so you can safely use higher `num_workers` values.

**If you see 429 errors**, reduce `num_workers`:

| Personas in Graph | Full Profiles `num_workers` | Simple Profiles `num_workers` |
|---|---|---|
| 5 or fewer | 300 | 300 |
| 6-15 | 3-5 | 50-100 |
| 16-50 | 2-3 | 10-30 |
| 50+ | 1-2 | 5-10 |

---

## Interaction Graph (`interaction_graph.json`)

A JSON object defining who sees whose answers. **Directed** — if pid_1000 lists pid_1001, that means pid_1000 sees pid_1001's answers, not the other way around.

### Fully connected (everyone sees everyone)

```json
{
  "pid_1000": ["pid_1001", "pid_1002"],
  "pid_1001": ["pid_1000", "pid_1002"],
  "pid_1002": ["pid_1000", "pid_1001"]
}
```

### One-directional influence (A influences B, but B doesn't influence A)

```json
{
  "pid_1000": [],
  "pid_1001": ["pid_1000"],
  "pid_1002": ["pid_1000"]
}
```

Here pid_1000 is an "influencer" — pid_1001 and pid_1002 see pid_1000's answers, but pid_1000 doesn't see anyone else's.

### Isolated persona (re-simulated each round but sees no one)

```json
{
  "pid_1000": ["pid_1001"],
  "pid_1001": [],
  "pid_999": []
}
```

pid_999 and pid_1001 will be re-asked each round but see no social context. Useful as a control.

### Adding new personas

1. Find available persona IDs by listing `text_simulation/text_simulation_input/` (files are named `pid_XXXX_prompt.txt`)
2. Add the persona ID to the graph JSON
3. Generate their baseline if they don't have one yet (see [Generating Baselines](#generating-baselines))

---

## Custom Questions

### Creating a custom questions file

Use this template:

```
---
## New Survey Question & Instructions (Please respond as the persona described above):
Please answer the following question as if you were taking this survey. The expected output is a JSON object in the format shown below.
---

Q1:
What do you think about bitcoin? State your opinion. Keep your answer under 150 words.
Question Type: Text Entry
Answer: [Masked]
Sentiment: -3 = strongly against bitcoin, +3 = strongly in favor of bitcoin

---
Expected output format:

The output should be a JSON object with key "Q1".

Example:
{
    "Q1": {
        "Question Type": "Text Entry",
        "Answers": {
            "Text": "Your opinion here..."
        }
    }
}
```

### Supported Question Types

**Text Entry** — open-ended text response:
```
Q1:
What do you think about bitcoin? Keep your answer under 150 words.
Question Type: Text Entry
Answer: [Masked]
Sentiment: -3 = strongly against bitcoin, +3 = strongly for bitcoin
```

**Single Choice** — pick one option:
```
Q1:
Do you think the government should provide free healthcare to all citizens?
Question Type: Single Choice
Options:
  1 - Yes
  2 - No
Answer: [Masked]
Sentiment: -3 = strongly against free healthcare, +3 = strongly for free healthcare
```

**Matrix** — multiple sub-items rated on the same scale:
```
Q2:
To what extent do you agree or disagree with the following statements?
Question Type: Matrix
Options:
  1 = Strongly disagree
  2 = Disagree
  3 = Neither agree nor disagree
  4 = Agree
  5 = Strongly agree
1. The government should increase spending on social welfare.
Answer: [Masked]
2. Tax dollars are being spent efficiently.
Answer: [Masked]
Sentiment: -3 = strongly against government spending, +3 = strongly for government spending
```

Matrix questions produce sub-keys in sentiment output: `Q2_1`, `Q2_2`, etc.

**Slider** — numeric value on a range:
```
Q3:
On a scale of 0 to 100, how much do you trust the federal government?
Question Type: Slider
1. [No Statement Needed]
Answer: [Masked]
Sentiment: -3 = complete distrust, +3 = complete trust
```

### The `Sentiment:` line

Adding a `Sentiment:` line to any question enables automatic sentiment scoring for that question. The format is:

```
Sentiment: -3 = <what -3 means>, +3 = <what +3 means>
```

The sentiment line is **stripped before sending to the main LLM** — the persona never sees it. After each round completes, a separate cheap model (`gpt-4.1-nano` by default) reads each persona's answer and scores it on the -3 to +3 scale.

You can mix questions with and without `Sentiment:` lines. Only questions with the line get scored.

### Expected output format section

Always end your file with an "Expected output format" section showing the LLM exactly what JSON structure to produce. Match the question types you used. See `custom_questions_with_sentiment.txt` for a complete multi-question example.

---

## Sentiment Scoring

### How it works

1. After each round (including baseline Round 0), the script reads every persona's answer
2. For each question that has a `Sentiment:` line, it builds a prompt asking `gpt-4.1-nano` to rate the answer from -3 to +3
3. Scores are appended to `sentiment_scores.csv` in the run output directory

### The scoring prompt

The sentiment model sees:
```
You are a sentiment scorer. Rate the following answer on a scale from -3 to +3.

Scale:
  -3 = strongly against bitcoin
  -2 = moderately against
  -1 = slightly against
   0 = neutral / unclear
  +1 = slightly for
  +2 = moderately for
  +3 = strongly in favor of bitcoin

Question: What do you think about bitcoin?
Answer: I think bitcoin is risky and volatile...

Respond with ONLY a single integer from -3 to 3.
```

### Output format (`sentiment_scores.csv`)

```csv
round,persona_id,question,score
0,pid_1000,Q1,1
0,pid_1001,Q1,-2
1,pid_1000,Q1,2
1,pid_1001,Q1,-2
...
```

- **round 0** = baseline (scores from existing Round 0 answers)
- **round 1+** = interaction rounds
- **score** = integer from -3 to +3
- **question** = `Q1`, `Q2_1`, `Q2_2`, `Q3`, etc. (matrix sub-items get `_N` suffix)

### Disabling sentiment

Use `--no_sentiment` to skip scoring even if your questions file has `Sentiment:` lines.

### Note on Round 0 baseline scoring

Round 0 scores the existing baseline outputs against your sentiment scales. If the baseline was run with the original 63 survey questions (not your custom questions), the Round 0 scores will be meaningless — the scorer will try to match Q1 from the old 63-question survey against your custom Q1 sentiment scale. This is expected. The meaningful sentiment data starts at Round 1.

---

## Generating Baselines

New personas added to the graph need baseline (Round 0) outputs. If a persona has no baseline, it still works — they just won't have "previous answers" shown to their neighbors in Round 1.

To generate baselines for specific personas:

1. Create a temp directory with just the prompt files you need:
```bash
mkdir -p text_simulation/text_simulation_input_temp
cp text_simulation/text_simulation_input/pid_1064_prompt.txt text_simulation/text_simulation_input_temp/
cp text_simulation/text_simulation_input/pid_1365_prompt.txt text_simulation/text_simulation_input_temp/
# ... add more as needed
```

2. Create a temporary config pointing to the temp directory:
```yaml
# /tmp/baseline_new.yaml
provider: "openai"
model_name: "gpt-4.1-mini-2025-04-14"
temperature: 0.0
max_tokens: 16384
max_retries: 10
num_workers: 300
force_regenerate: false
input_folder_dir: "text_simulation_input_temp"
output_folder_dir: "text_simulation_output"
system_instruction: |
  You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described in the 'Persona Profile' (which consists of their past survey responses).
  Adhere to the persona by being consistent with their previous answers and stated characteristics.
  Follow all instructions provided for the new question carefully regarding the format of your answer.
```

3. Run the baseline:
```bash
poetry run python text_simulation/run_LLM_simulations.py --config /tmp/baseline_new.yaml
```

4. Clean up:
```bash
rm -rf text_simulation/text_simulation_input_temp
```

The baseline outputs are saved to `text_simulation/text_simulation_output/pid_XXXX/`.

---

## Output Structure

```
text_simulation_output_interaction/
  my_experiment/                    # --run_name
    metadata.json                   # Full run config, graph, timestamps, persona list
    sentiment_scores.csv            # All sentiment scores across all rounds
    round_1/
      pid_1000/
        pid_1000_response.json      # Raw LLM response
      pid_1001/
        pid_1001_response.json
      ...
      answer_blocks_llm_imputed/    # Verified answer blocks (standard questions only)
    round_2/
      ...
    round_20/
      ...
```

### Response JSON format

Each `pid_XXXX_response.json`:
```json
{
  "persona_id": "pid_1000",
  "question_id": "pid_1000",
  "prompt_text": "...(full prompt sent to LLM)...",
  "response_text": "{\"Q1\": {\"Question Type\": \"Text Entry\", \"Answers\": {\"Text\": \"I think bitcoin...\"}}}",
  "usage_details": {
    "prompt_token_count": 25000,
    "completion_token_count": 200,
    "total_token_count": 25200
  },
  "llm_call_error": null
}
```

### Reading answers programmatically

```python
import json, os

def load_answer(pid, round_dir):
    path = os.path.join(round_dir, pid, f"{pid}_response.json")
    with open(path) as f:
        data = json.load(f)
    text = data["response_text"].strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)

# Example: read pid_1000's answer from round 5
answers = load_answer("pid_1000", "text_simulation/text_simulation_output_interaction/my_experiment/round_5")
print(answers["Q1"]["Answers"]["Text"])
```

---

## How It Works

```
Round 0 (baseline):   Each persona answers independently (pre-existing outputs)

Round 1:              For each persona P:
                        1. Load P's base prompt (persona profile)
                        2. Replace questions with custom questions (if provided)
                        3. Look up P's neighbors in the directed graph
                        4. Load neighbors' Round 0 answers, inject into prompt
                        5. Send to LLM, save response
                        6. Score sentiment (if Sentiment: lines exist)

Round 2:              Same, but neighbors' answers come from Round 1

...

Round N:              Same, but neighbors' answers come from Round N-1
```

Each persona sees:
- Their own previous-round answer (unless `--no_prev_answers`)
- All neighbors' previous-round answers (with demographic summaries)

---

## Troubleshooting

### 429 Rate Limit Errors

```
FAILED: pid_1128: Error code: 429 - Rate limit reached for gpt-4.1-mini ...
```

**Cause**: Too many concurrent requests exceed your OpenAI tokens-per-minute (TPM) limit.

**Fix**: Reduce `num_workers` in your config. For 15 personas in a fully connected graph (~27k tokens per prompt), use `num_workers: 3`.

### Missing data in specific rounds

If a persona fails in a round (rate limit, API error), its answer for that round is missing. The next round will use the **most recent available answer** from that persona. The sentiment CSV will also skip that persona for that round. To avoid gaps, reduce `num_workers`.

### Custom questions: "response is not valid JSON, will retry"

The LLM occasionally returns malformed JSON. The script retries up to `max_retries` times. If it keeps failing:
- Check your "Expected output format" section — make sure the example JSON matches your question types exactly
- Try increasing `max_tokens` if the response might be getting truncated

### Sentiment scores all 0

Make sure your `Sentiment:` line follows the exact format:
```
Sentiment: -3 = <negative description>, +3 = <positive description>
```

The parser expects `-3 = ...` and `+3 = ...` separated by a comma.
