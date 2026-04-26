# Minimal Diagnostic Experiment — Execution Plan

## Goal

Determine whether two specific sections in `evaluation_profile.txt` are the
primary cause of v2's underperformance on the pricing study (40-item product
preference task). If yes, this becomes a cheap fix before considering a full
v3 redesign.

## Hypothesis

Two sections in v2's `evaluation_profile.txt` introduce biased priors that
cause the downstream simulation LLM to systematically reject low-price
purchases it would otherwise accept:

1. `## Product-category predictions for the pricing study`
2. `## Willingness-to-pay calibration`

These sections pre-commit the simulator to category-level predictions (e.g.,
"indulgence items: low likelihood") that override the question-level price
information.

## Experimental conditions

Four conditions, run on **pid_1** only for the first round:

| Condition | Description | File used |
|---|---|---|
| **Control (v2)** | Current v2 design, unchanged | Existing v2 `evaluation_profile.txt` |
| **Fix A** | v2 with the 2 target sections removed; Summary unchanged | New file, see Task 1 below |
| **Fix B** | v2 with the 2 sections removed AND Summary rewritten | New file, see Task 1 below |
| **Raw** | No skill profile; LLM reads raw transcript directly | Existing raw-question config |

The `background.txt` and `decision_procedure.txt` files stay identical across
Control, Fix A, and Fix B. Only `evaluation_profile.txt` changes.

---

## Task 1: Create the two fix-version files

Start from the existing file:
`text_simulation/skills/pid_1/v2_inferred/evaluation_profile.txt`

### Fix A — minimal removal

Copy the existing file, then DELETE these two sections completely:
- The entire `## Product-category predictions for the pricing study`
  section (from its header until the next `## ` header)
- The entire `## Willingness-to-pay calibration` section (from its header
  until the next `## ` header)

Keep everything else unchanged, including the existing Summary section.

Save as:
`text_simulation/skills/pid_1/v2_inferred_fix_A/evaluation_profile.txt`

Also copy the unchanged `background.txt` and `decision_procedure.txt` into
the `v2_inferred_fix_A/` directory so it's a complete profile set.

### Fix B — removal plus Summary rewrite

Start from Fix A. Replace the existing `## Summary` section with this text:

```
## Summary
This is a moderately low-income, part-time employed Black female in her
30s-40s living in the Midwest, with a strong minimalist attitude toward
possessions and a self-identified spendthrift spending style. She is
highly conscientious, organized, and reliable but shows behavioral
impatience and moderate risk aversion. Socially, she is an in-group
cooperator: generous and loyal to family but selfish toward strangers.
Her economic game behavior and intertemporal choices indicate pragmatic,
self-interested decision-making despite stated altruism. Key tension to
watch: minimalism (attitude toward accumulating possessions) and
spendthrift self-ID (attitude toward spending money) coexist — they are
not contradictory and should not be collapsed into a single "frugal" or
"free-spending" prediction.
```

Save as:
`text_simulation/skills/pid_1/v2_inferred_fix_B/evaluation_profile.txt`

Also copy the unchanged `background.txt` and `decision_procedure.txt` into
the `v2_inferred_fix_B/` directory.

### Verification

After creating both files, confirm:
- Fix A's file is missing the two target sections but contains every
  other section that was in the original
- Fix B's file is identical to Fix A except for the rewritten Summary
- Both files still start with `## Budget constraint` and end with a
  `## Summary` section

---

## Task 2: Create prompt input files for each condition

Use the existing script `text_simulation/create_text_simulation_input.py`
(or whatever script builds the combined prompts in this repo) to generate
the per-persona prompt files for each condition.

Run once per condition, pointing at the corresponding skill directory:

```bash
# Control (existing v2)
python text_simulation/create_text_simulation_input.py \
  --persona_text_dir text_simulation/skills_as_text/v2_inferred \
  --question_prompts_dir text_simulation/question_prompts \
  --output_combined_prompts_dir text_simulation/text_simulation_input_control \
  --pids 1

# Fix A
python text_simulation/create_text_simulation_input.py \
  --persona_text_dir text_simulation/skills_as_text/v2_inferred_fix_A \
  --question_prompts_dir text_simulation/question_prompts \
  --output_combined_prompts_dir text_simulation/text_simulation_input_fix_A \
  --pids 1

# Fix B
python text_simulation/create_text_simulation_input.py \
  --persona_text_dir text_simulation/skills_as_text/v2_inferred_fix_B \
  --question_prompts_dir text_simulation/question_prompts \
  --output_combined_prompts_dir text_simulation/text_simulation_input_fix_B \
  --pids 1
```

If `create_text_simulation_input.py` does not accept a `--pids` flag, filter
for pid_1 however the script supports (e.g., only place `pid_1.txt` in the
input directory before running, or modify to filter).

The Raw condition uses the existing raw-transcript prompt pipeline, no new
input generation needed.

---

## Task 3: Run the simulation for each condition

Run the standard LLM simulation script once per condition. Use whichever
script is normally used for single-config runs (probably
`text_simulation/run_LLM_simulations.py` or the equivalent).

Use the **same model, temperature, and reasoning level** for all four
conditions — the only thing that should vary is the input prompt. Any
model/temperature difference would confound the experiment.

Suggested configuration (match whatever the existing pipeline uses):
- Model: same as current v2 pipeline (e.g., `gpt-4.1-mini` or `gpt-5.4-mini`)
- Temperature: 0.0 (deterministic)
- Max tokens: same as existing config

Output directories (suggested):
- Control → `text_simulation/text_simulation_output_control/`
- Fix A → `text_simulation/text_simulation_output_fix_A/`
- Fix B → `text_simulation/text_simulation_output_fix_B/`
- Raw → `text_simulation/text_simulation_output_raw/` (may already exist)

Only the 40 pricing-study questions (product preference items) need to be
run — filter to those if the pipeline allows. If not, run all questions but
only evaluate the 40 pricing items in the next task.

---

## Task 4: Extract pricing-study answers and compute accuracy

For each condition, parse the LLM's answers on the 40 product preference
questions and compare against pid_1's wave 4 ground truth.

### 4.1 — Per-condition accuracy

For each condition compute:
- Number correct out of 40
- Overall accuracy percentage

Present a single table:

| Condition | Correct / 40 | Accuracy |
|---|---|---|
| Control (v2) | X / 40 | XX% |
| Fix A | X / 40 | XX% |
| Fix B | X / 40 | XX% |
| Raw | X / 40 | XX% |

### 4.2 — Price-tier accuracy (the key diagnostic)

This is the most diagnostic analysis. Split the 40 questions into three
groups by how the asking price compares to a reasonable market price:

- **Low-price-ratio group**: asking price is clearly below typical market
  price (e.g., $0.95 for raspberries, $0.00 for toilet paper, $3.34 for
  Häagen-Dazs, $0.87 for lunch meat, $2.19 for Tylenol). Expected behavior:
  most people say Yes.
- **Normal-price-ratio group**: asking price is approximately market price.
- **High-price-ratio group**: asking price is clearly above typical market
  price (e.g., $53.98 for ham, $17.95 for ramen, $25.55 for batteries,
  $11.66 for Hershey bars). Expected behavior: most people say No.

For classifying each of the 40 items, use best judgment about typical US
grocery prices for the specific product and size as of 2024–2025. Produce
the classification as a table and include it in the output for
reproducibility.

Then compute accuracy per condition per price tier:

| Condition | Low-ratio | Normal-ratio | High-ratio |
|---|---|---|---|
| Control (v2) | X/N | X/N | X/N |
| Fix A | X/N | X/N | X/N |
| Fix B | X/N | X/N | X/N |
| Raw | X/N | X/N | X/N |

**Interpretation guide:**
- If Fix A/B improves mainly on the **low-ratio group**, hypothesis
  confirmed: the profile was making the LLM incorrectly reject good-price
  purchases.
- If Fix A/B improves uniformly across all three tiers, the mechanism is
  different — the two sections were affecting accuracy globally, not just
  for low-price items.
- If Fix A/B does NOT improve at all, the two sections are not the cause;
  the problem lies elsewhere in the profile.

### 4.3 — Qualitative check (optional but valuable)

If the pipeline supports chain-of-thought or reasoning traces in the LLM
output, inspect the reasoning on 5–10 low-price-ratio items for each
condition. Specifically check whether the LLM's reasoning references any
of these profile phrases:

- "minimalism" / "minimalist"
- "indulgence items" / "novelty items"
- "low likelihood"
- "price sensitive"
- "willingness to pay"
- "bulk items" / "party-size items"

Report approximately how often such language appears in each condition.
Expectation: Control will reference these phrases often, Fix A less so,
Fix B least often, Raw never.

If reasoning traces are not currently in the pipeline, skip this task.

---

## Task 5: Report

Produce a single markdown file summarizing all results. Include:

1. The four-condition overall accuracy table (from 4.1)
2. The price-tier accuracy table (from 4.2)
3. The price-tier classification of the 40 items (for reproducibility)
4. Qualitative check results if available (from 4.3)
5. A one-paragraph interpretation: does Fix A or Fix B close the gap to
   Raw? Is the improvement concentrated in the predicted price tier? What
   does this suggest for the next step?

Save as:
`experiments/minimal_diagnostic_pid1/results.md`

Also save all raw outputs in that directory for audit:
- Per-condition raw LLM responses (JSON files per condition)
- The generated Fix A and Fix B profile files for reference

---

## Important constraints

**Do NOT**:
- Modify `background.txt` or `decision_procedure.txt` in any condition
- Change the model, temperature, or max_tokens across conditions
- Run the experiment on multiple personas yet — this is a single-persona
  diagnostic, and multi-persona validation is a follow-up step after
  results are interpreted
- Implement a v3 redesign based on partial results — wait for the full
  four-condition comparison

**Do**:
- Use deterministic decoding (temperature = 0.0)
- Log the exact model version and config used
- Save intermediate files so the experiment is reproducible
- Flag any issues (truncated responses, malformed outputs, missing ground
  truth) before computing final accuracy

---

## Expected cost

Per persona:
- 40 pricing questions × 4 conditions = 160 LLM calls
- At a reasonable input size (~2000-3000 tokens per prompt), total cost
  should be under $2 per run using gpt-4.1-mini-tier models.

Budget ceiling for this diagnostic: $5. If costs exceed this, stop and
flag before continuing.

---

## After this experiment

Do not take further action on profile redesign until the results of this
diagnostic are reviewed. Specifically:

- If results confirm the hypothesis → next step is to repeat on 5-10 more
  personas to confirm the fix generalizes
- If results reject the hypothesis → next step is to reconsider what in
  the v2 design is causing the pricing-study regression
- If results are ambiguous → propose a refined diagnostic before scaling
  up

The goal of this experiment is to LEARN WHAT'S WRONG, not to ship a fix.
The fix ships only after the mechanism is confirmed.
