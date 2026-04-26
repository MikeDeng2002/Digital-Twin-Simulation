# How to Evaluate Nano and Mini Experiments

This document describes the evaluation pipeline for the **nano_temp0** and **mini_temp0** 20-persona experiments.

Each experiment covers 10 prompt settings × 4 reasoning levels = **40 configs**, using:
- **nano_temp0**: model `gpt-5.4-nano`, temperature 0.0
- **mini_temp0**: model `gpt-5.4-mini`, temperature 0.0

---

## Prerequisites

Before evaluating, all batch jobs must be completed and response files downloaded.
Each config directory should contain **20 `pid_N_response.json` files** (one per persona).

Check file counts:
```bash
for setting in skill_v1 skill_v2 skill_v3 raw raw_start_v1 raw_start_v2 raw_start_v3 skill_v1_raw_end skill_v2_raw_end skill_v3_raw_end; do
  for reasoning in none low medium high; do
    dir="text_simulation/text_simulation_output_nano_temp0/$setting/$reasoning"
    count=$(find "$dir" -name "pid_*_response.json" 2>/dev/null | wc -l)
    echo "$setting/$reasoning: $count"
  done
done
```
Replace `nano_temp0` with `mini_temp0` to check the mini experiment.

---

## Step 1: Postprocessing

The OpenAI Batch API downloads raw LLM response JSON files but does **not** run postprocessing. This step parses each model response and writes structured answer files needed by the evaluator.

Run from the project root (`Digital-Twin-Simulation/`):

```bash
# For nano
python text_simulation/run_postprocess.py --suite nano_temp0

# For mini
python text_simulation/run_postprocess.py --suite mini_temp0
```

**What it produces:** For each config, creates `answer_blocks_llm_imputed/` containing
`pid_N_wave4_Q_wave4_A.json` files — the structured LLM answers used for accuracy scoring.

**Optional filters** (re-run a single config):
```bash
python text_simulation/run_postprocess.py --suite nano_temp0 --setting skill_v3 --reasoning high
```

---

## Step 2: Accuracy Evaluation

Runs the full evaluation pipeline for all 40 configs and prints a summary table.

```bash
# For nano
python evaluation/eval_temp0_suite.py --suite nano_temp0

# For mini
python evaluation/eval_temp0_suite.py --suite mini_temp0
```

**What it does for each config:**
1. Runs `evaluation/json2csv.py` — converts imputed answer JSON files to CSV format
2. Runs `evaluation/mad_accuracy_evaluation.py` — computes MAD (Mean Absolute Deviation) accuracy against wave-4 ground truth; writes `accuracy_evaluation/mad_accuracy_summary.xlsx`
3. Extracts metrics: LLM Accuracy, 95% CI, Random Baseline, Human Ceiling

**Output:**
- Prints a pivot table: rows = settings, columns = reasoning levels (`none / low / medium / high`), plus an `avg` column
- Saves results to:
  - `evaluation/nano_temp0_20p_results.csv`
  - `evaluation/mini_temp0_20p_results.csv`

**Optional filters** (re-run a single config):
```bash
python evaluation/eval_temp0_suite.py --suite nano_temp0 --setting raw --reasoning high
```

---

## Full Sequence (both experiments)

```bash
# 1. Postprocess both suites
python text_simulation/run_postprocess.py --suite nano_temp0
python text_simulation/run_postprocess.py --suite mini_temp0

# 2. Evaluate both suites
python evaluation/eval_temp0_suite.py --suite nano_temp0
python evaluation/eval_temp0_suite.py --suite mini_temp0
```

All commands should be run from the project root: `Digital-Twin-Simulation/`

---

## Output Directory Structure (per config)

```
text_simulation_output_nano_temp0/
└── {setting}/
    └── {reasoning}/
        ├── pid_1/pid_1_response.json          ← raw LLM output (from batch API)
        ├── ...
        ├── pid_20/pid_20_response.json
        ├── answer_blocks_llm_imputed/          ← created by Step 1 (postprocess)
        │   ├── pid_1_wave4_Q_wave4_A.json
        │   └── ...
        ├── csv_comparison/                     ← created by Step 2 (json2csv)
        └── accuracy_evaluation/               ← created by Step 2 (mad evaluation)
            └── mad_accuracy_summary.xlsx
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `SKIP: no imputed answers` | Step 1 not run yet | Run `run_postprocess.py` first |
| Negative accuracy (e.g. -32) | Model output dollar amounts instead of ordinal index | Set outlier values to `null` in the affected `pid_N_wave4_Q_wave4_A.json` and re-run Step 2 |
| Only 5 response files in a config | Batch job failed or was not submitted | Re-run `run_batch_experiment_parallel.py` — it will skip completed configs and resubmit missing ones |
| `openpyxl` import error | Missing dependency | `pip install openpyxl` |
