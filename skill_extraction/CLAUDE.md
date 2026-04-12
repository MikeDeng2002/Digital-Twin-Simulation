# Skill Extraction — CLAUDE.md

## What this folder does

This module extracts structured **skill profiles** from persona transcripts in the
Twin-2K-500 dataset. Each persona's ~450 survey responses (waves 1–3) are distilled
into three text files that represent what they know, what they look up, and how they
reason. These skill files replace the raw transcript as the persona input for numeric
prediction experiments (e.g. Bitcoin price, AGI year).

---

## Folder structure

```
skill_extraction/
├── CLAUDE.md                  # This file
├── config.py                  # API key and model settings
├── extract_skills.py          # Main extraction script (one persona)
└── batch_extract.py           # Batch runner (all or N personas)
```

Output files are written to (relative to `Digital-Twin-Simulation/`):
```
text_simulation/skills/pid_{pid}/
├── v1_direct/
│   ├── background.txt
│   ├── tools.txt
│   └── decision_procedure.txt
├── v2_inferred/
│   ├── background.txt
│   ├── tools.txt
│   └── decision_procedure.txt
└── v3_maximum/
    ├── background.txt
    ├── tools.txt
    └── decision_procedure.txt
```

---

## Data inputs

| What | Path |
|------|------|
| Persona transcript (text) | `text_simulation/text_personas/pid_{pid}.txt` |
| Persona transcript (raw JSON) | `data/mega_persona_json/answer_blocks/pid_{pid}_wave4_Q_wave1_3_A.json` |

The text version (already converted) is what the prompts use.

---

## API configuration

- **Provider:** OpenAI
- **Model:** `gpt-4.1-mini-2025-04-14`
- **Max tokens:** 1500 per call
- **API key:** set as environment variable `OPENAI_API_KEY`
  - Key is stored in `.env` at the project root (`Digital-Twin-Simulation/.env`)
- **Calls per persona:** 3 (one per version: v1, v2, v3)

**.env file format:**
```
OPENAI_API_KEY=your_api_key_here
```

---

## Three skill versions

### v1_direct — direct evidence only
- Uses only ~120 of ~450 questions that explicitly ask about the relevant topic
- No demographic inference, no cross-referencing scales
- If no direct evidence exists for a component, states this explicitly

### v2_inferred — direct + labeled demographic inference
- Uses ~200 of ~450 questions
- Demographic inference allowed but every inference must be labeled:
  `"Inferred from [source]: [conclusion] because [reasoning]"`
- Cross-scale inference allowed when 2+ scales agree

### v3_maximum — all signals, maximum inference
- Uses all ~450 questions
- Uses every signal: Big Five, Need for Cognition/Closure, Maximizer,
  Self-monitoring, Self-concept clarity, cognitive test errors, economic game
  behavior, word associations, intertemporal patience, social desirability
- Aggressive inference, always with stated reasoning chain

---

## Output components

### background.txt
What this person already knows without looking anything up.
Sources: education, domain expertise, values, life experiences, stated beliefs,
personality traits.

### tools.txt
What information sources this person would realistically reach for before forming
an opinion. Sources: research habits, education level, income, political identity,
Need for Cognition score, lifestyle signals.

### decision_procedure.txt
How this person reasons under uncertainty before committing to a number.
Sources: Need for Closure, Need for Cognition, Maximizer, Self-monitoring,
Self-concept clarity, economic game behavior, intertemporal patience.

**Always ends with:**
```
w_social estimate: [0.0–1.0] — [justification]
```
where `0.0` = fully self-reliant, `1.0` = fully follows others.

---

## Extraction prompts

Each version uses a distinct prompt. Insert the full persona transcript where
`{transcript}` appears.

---

### Prompt for v1_direct

```
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
w_social estimate: [0.0–1.0] — [one sentence justification citing the specific
items that support this estimate]

TRANSCRIPT:
{transcript}
```

---

### Prompt for v2_inferred

```
You are extracting a digital twin skill profile from a survey transcript.

RULES:
- Direct evidence: cite the scale name and item number
- Demographic inference: allowed but label it explicitly as:
  "Inferred from [demographic fact]: [conclusion] because [reasoning]"
- Cross-scale inference: allowed when two or more scales point in the same
  direction — note which scales agree
- No speculation beyond what the data can reasonably support

The survey covers personality scales (Big Five, Need for Closure, Need for
Cognition, Maximizer, Self-monitoring, Self-concept clarity), values, economic
games, cognitive tests, and demographics.

Output EXACTLY this format with no other text:

---BACKGROUND---
Synthesize direct self-report items on knowledge, values, beliefs, and
experiences. Use demographics to contextualize domain knowledge — but label
every inference. Cover: stated values and their rankings, Big Five profile
summary, life situation and what it implies about lived knowledge.

---TOOLS---
Infer realistic information sources from education, income, political identity,
need for cognition score, and lifestyle signals. For each source: state the
evidence and reasoning. Also state 2-3 sources she would NOT use and why.
Acknowledge this section is inferred since the dataset contains no direct
media-use questions.

---DECISION_PROCEDURE---
Use all relevant psychometric scales: Need for Closure, Need for Cognition,
Maximizer, Self-monitoring, Self-concept clarity, and economic game behavior.
Cross-reference scales where they agree. Note any contradictions between scales.
End with:
w_social estimate: [0.0–1.0] — [2-3 sentence justification citing which scales
and their agreement/disagreement drive this estimate]

TRANSCRIPT:
{transcript}
```

---

### Prompt for v3_maximum

```
You are extracting the richest possible digital twin skill profile from a survey
transcript. Use every available signal — direct scales, cognitive test performance
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
Big Five profile and what it means for how she engages with information, values
hierarchy (which values scored highest and what this reveals about her worldview),
minimalism and consumerism attitudes, cognitive test performance including specific
error types and what they reveal about her reasoning style, word association chain
if it reveals cognitive patterns, social desirability score and what it implies
about self-report reliability. Be specific — name actual scores and items.

---TOOLS---
Infer every plausible information source using ALL available signals:
- Education + income → access and familiarity with different sources
- Need for Cognition score → depth of research she engages in
- Political identity → partisan vs neutral source preference
- Empathy scale → how much she relies on social networks for information
- Maximizer scale → does she comparison-shop information sources?
- Consumerism/uniqueness scale → does she follow trends or ignore them?
- Minimalism scale → does she curate information carefully?
Name actual platforms and source types. Rate her likely research depth
(shallow / moderate / deep) with reasoning. List sources she would NOT use.
State confidence level for each inference.

---DECISION_PROCEDURE---
Synthesize ALL behavioral signals into a coherent reasoning profile:
- Need for Closure: how fast does she want resolution?
- Need for Cognition: how much analytical effort does she invest?
- Maximizer: does she seek the objectively best answer or satisfice?
- Self-monitoring: does she notice what others think even if she doesn't defer?
- Self-concept clarity: is her self-view stable enough to resist social pressure?
- Economic game gap: compare dictator vs ultimatum — does stated behavior match
  strategic behavior?
- Intertemporal patience: does she think long-term when forming views?
- Cognitive errors: what reasoning failures appeared? What does this imply for
  numeric prediction tasks?
- Social desirability: how much should we trust her self-reports?
End with:
w_social estimate: [0.0–1.0]
Reasoning: [full paragraph explaining which signals drive this estimate, which
signals conflict, and how you resolved the conflicts]

TRANSCRIPT:
{transcript}
```

---

## Output parsing

Each API response contains three sections separated by:
- `---BACKGROUND---`
- `---TOOLS---`
- `---DECISION_PROCEDURE---`

Parse by splitting on these markers. Save each section as its own `.txt` file.

---

## Research context

### Why three versions?
Tests whether extraction depth changes prediction quality (Experiment 1) and social
influenceability (Experiments 2 and 3). Creates a spectrum from minimal to maximal
information use.

### What is w_social?
The social influence weight — how much a twin's numeric prediction shifts when shown
randomly injected neighbor numbers. Measured by regressing center predictions on
neighbor mean across 50 trials. The `decision_procedure.txt` is the theoretical
predictor of w_social.

### Experiment grid

|                  | No interaction | With interaction |
|------------------|----------------|------------------|
| **Raw text**     | Experiment 1a  | Experiment 2     |
| **Skill method** | Experiment 1b  | Experiment 3     |

---

## Full evaluation pipeline

The findings in `findings.md` are produced by running three stages in sequence.

### Stage 1 — Extract skill profiles
```bash
# From Digital-Twin-Simulation/
poetry run python skill_extraction/batch_extract.py
```
Output: `text_simulation/skills/pid_*/v{1,2,3,4}_*/background.txt` etc.

### Stage 2 — Build prompt inputs
Each profile version needs its own input folder of combined prompts (skill profile +
survey question). Run `create_text_simulation_input.py` pointing to the correct skill
version directory:
```bash
# Example for v3_maximum
python text_simulation/create_text_simulation_input.py \
  --persona_text_dir text_simulation/skills_as_text/v3_maximum \
  --question_prompts_dir text_simulation/question_prompts \
  --output_combined_prompts_dir text_simulation/text_simulation_input_skill_v3_maximum
```
Repeat for each version (v1_direct, v2_inferred, v3_maximum, v4_chained) and for
the demographic-only baseline (uses `text_simulation/text_personas/` directly).

### Stage 3 — Run LLM simulation
Profile choice is controlled entirely by the **YAML config file** passed to `--config`.
The script reads whichever `input_folder_dir` the config specifies — there is no
profile flag in the script itself.

```bash
# Demographic only (baseline)
python text_simulation/run_LLM_simulations.py \
  --config text_simulation/configs/demographic_4mini_config.yaml

# Skill v1
python text_simulation/run_LLM_simulations.py \
  --config text_simulation/configs/skill_v1_config.yaml

# Skill v2
python text_simulation/run_LLM_simulations.py \
  --config text_simulation/configs/skill_v2_config.yaml

# Skill v3 (maximum) — used for Bitcoin experiment
python text_simulation/run_LLM_simulations.py \
  --config text_simulation/configs/skill_v3_config.yaml

# Skill v4
python text_simulation/run_LLM_simulations.py \
  --config text_simulation/configs/skill_v4_config.yaml
```

Each config specifies model, temperature, input/output folders, and worker count.
Configs for different models (gpt-4.1-mini, o4-mini, gpt-4o) exist for each version:

| Profile | gpt-4.1-mini | o4-mini | gpt-4o |
|---------|-------------|---------|--------|
| Demographic | `demographic_4mini_config.yaml` | `demographic_o4mini_config.yaml` | `demographic_gpt4o_config.yaml` |
| Skill v1 | `skill_v1_config.yaml` | `skill_v1_o4mini_config.yaml` | `skill_v1_gpt4o_config.yaml` |
| Skill v2 | `skill_v2_config.yaml` | `skill_v2_o4mini_config.yaml` | `skill_v2_gpt4o_config.yaml` |
| Skill v3 | `skill_v3_config.yaml` | `skill_v3_o4mini_config.yaml` | `skill_v3_gpt4o_config.yaml` |
| Skill v4 | `skill_v4_config.yaml` | `skill_v4_o4mini_config.yaml` | `skill_v4_gpt4o_config.yaml` |

All configs live in `text_simulation/configs/`.

### Stage 4 — Evaluate accuracy
```bash
# From Digital-Twin-Simulation/
python evaluation/mad_accuracy_evaluation.py \
  --predictions text_simulation/text_simulation_output_skill_v3/answer_blocks_llm_imputed \
  --ground_truth data/mega_persona_json/answer_blocks \
  --output evaluation/results/skill_v3_4mini_accuracy.xlsx
```
This computes **Accuracy = 1 − normalized MAD** vs wave-4 ground truth, with 95% CI
using a t-distribution. The numbers in `findings.md` come from running this for every
profile × model combination.

---

## Config file structure

Example (`skill_v3_config.yaml`):
```yaml
provider: "openai"
model_name: "gpt-4.1-mini-2025-04-14"
temperature: 0.0
max_tokens: 16384
max_retries: 10
num_workers: 50
force_regenerate: false
max_personas: 100

input_folder_dir: "text_simulation_input_skill_v3_maximum"
output_folder_dir: "text_simulation_output_skill_v3"

system_instruction: |
  You are an AI assistant. Your task is to answer the 'New Survey Question' as if
  you are the person described in the 'Persona Skill Profile' above (which summarizes
  their background, information sources, and decision procedure).
  Adhere to the persona by being consistent with their skill profile characteristics.
  Follow all instructions provided for the new question carefully regarding the format
  of your answer.
```

Key fields:
- `input_folder_dir` — subfolder inside `text_simulation/` containing `pid_*_prompt.txt` files
- `output_folder_dir` — where LLM responses are saved
- `provider` — `"openai"` or `"gemini"`
- `num_workers` — parallel API requests
- `force_regenerate` — set `true` to re-run already-completed personas
