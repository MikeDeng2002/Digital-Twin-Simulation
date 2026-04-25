# Twin-2K-500: Digital Twin Simulation with LLMs

[![HuggingFace](https://img.shields.io/badge/🤗%20Datasets-Twin--2K--500-blue)](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
![Downloads](https://img.shields.io/badge/downloads-1552-brightgreen)
[![arXiv](https://img.shields.io/badge/arXiv-2505.17479-b31b1b.svg)](https://arxiv.org/abs/2505.17479)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://digital-twin-simulation-version2.readthedocs.io/en/latest/index.html)


This repository contains the code and tutorial for ["Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions"](https://arxiv.org/abs/2505.17479). The Twin-2K-500 dataset is designed to support the benchmarking and advancement of LLM-based persona simulation methods.

- **Dataset:** [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)

## Citation

```
@article{toubia2025twin2k500,
  title     = {Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions},
  author    = {Toubia, Olivier and Gui, George Z. and Peng, Tianyi and Merlau, Daniel J. and Li, Ang and Chen, Haozhe},
  journal   = {arXiv preprint arXiv:2505.17479},
  year      = {2025}
}
```

## How to Use Our Dataset
Before getting started, we highly recommend reviewing our documentation for detailed information about the dataset and tutorials for various use cases: **[Documentation](https://digital-twin-simulation-version2.readthedocs.io/en/latest/index.html)**

## Overview

The digital twin simulation system creates virtual representations of individuals based on their survey responses and simulates their behavior in response to new survey questions. The system uses LLMs to generate realistic responses that maintain consistency with the original persona profiles.

## Project Structure

```
.
├── skill_extraction/          # Skill persona extraction (NEW)
│   ├── extract_skills.py             # v1 / v2 / v3 single-persona extraction
│   ├── batch_extract_skills_v2.py    # Batch driver for the 3-version skill build
│   ├── EXTRACTION_PROMPTS.md         # Prompt specs for v1 / v2 / v3
│   └── create_skill_inputs.py        # Merge skills with questions → simulation_input
├── text_simulation/           # Main simulation code
│   ├── configs/                      # Configuration files
│   ├── text_personas/                # Persona profile data (raw transcript form)
│   ├── text_questions/               # Survey questions (one txt per pid)
│   ├── skills/                       # Extracted skills, per persona
│   │   └── pid_{N}/{v1_direct, v2_inferred, v3_maximum}/
│   │       ├── background.txt
│   │       ├── tools.txt
│   │       └── decision_procedure.txt
│   ├── text_simulation_input*/       # Persona/skill + questions merged inputs
│   └── text_simulation_output*/      # Raw + post-processed LLM responses
├── evaluation/                # Accuracy, cognitive bias, product preference, bootstrap
├── notebooks/                 # Demo notebooks
│   ├── demo_simple_simulation.ipynb  # Quick start: simulate responses to new questions
│   └── demo_full_pipeline.ipynb      # Full pipeline (alternative to shell scripts)
├── scripts/                   # Utility scripts
├── data/                      # Raw data
└── cache/                     # Cached data
```

## Pipeline Overview

The end-to-end flow is **persona → skill → simulation_input → simulation_output → evaluation**.

```
text_personas/pid_{N}.txt ──┐
                            ├──► skill_extraction (v1/v2/v3) ──► text_simulation/skills/pid_{N}/v{1,2,3}/{background,tools,decision_procedure}.txt
                            │
text_questions/pid_{N}.txt ─┴──► create_skill_inputs.py ──► text_simulation_input_skill_v{1,2,3}/
                                                                           │
                                                                           ▼
                                                          run_LLM_simulations.py
                                                                           │
                                                                           ▼
                                                          text_simulation_output_skill_v{1,2,3}/
                                                                           │
                                                                           ▼
                                                       evaluation/ (accuracy, bias, preference, bootstrap)
```

## Key Components

1. **Persona Processing**
   - `convert_persona_to_text.py`: Converts persona data to text format
   - `batch_convert_personas.py`: Batch processes multiple personas

2. **Question Processing**
   - `convert_question_json_to_text.py`: Converts question data to text format

3. **Skill Extraction** (see `../skill_extraction.md` for the full spec)
   - We currently extract **3 skills** per persona — `background`, `tools`, `decision_procedure` — and produce them in **3 versions** that vary how much inference reasoning we allow on top of the survey signal:
     - **v1_direct** — direct evidence only. Only questions that explicitly ask about a topic are used; demographics are not extrapolated. (~120 of ~450 items used)
     - **v2_inferred** — direct + labeled demographic / cross-scale inference. Inferences are explicitly tagged. (~200 of ~450 items used)
     - **v3_maximum** — every available signal: all psychometric scales, cognitive-test errors, economic-game behavior, intertemporal patience, word associations, social-desirability, full demographics. Aggressive inference, with explicit reasoning chains. (~450 of ~450 items used)
   - Each version writes the same three files into `text_simulation/skills/pid_{N}/v{1,2,3}_*/`:
     `background.txt`, `tools.txt`, `decision_procedure.txt`. The `decision_procedure.txt` ends with a `w_social` estimate in `[0.0, 1.0]` used downstream for social-influence experiments.
   - Drivers: `skill_extraction/extract_skills.py` (single persona) and `skill_extraction/batch_extract_skills_v2.py` (batch). Prompts are pinned in `skill_extraction/EXTRACTION_PROMPTS.md` and called against `claude-sonnet-4-6`.

4. **Simulation Input Assembly**
   - `text_simulation/create_skill_inputs.py` (and `create_skill_v2_inputs.py`, `create_skills_v4_inputs.py`, …) merge the extracted skill persona with `text_questions/pid_{N}.txt` to produce `text_simulation_input_skill_v{1,2,3}/pid_{N}.txt`.
   - For ablations, a family of `text_simulation_input_skills_v{1,2,3}_{bg, bg_ep, bg_dp, bg_tools, bg_dp_ep, …}/` directories isolate which skill components are exposed to the model.

5. **Simulation**
   - `text_simulation/run_LLM_simulations.py`: Runs the actual LLM simulations (single + batch).
   - `text_simulation/run_LLM_simulation_interaction.py`: Runs the social-interaction variant (uses `w_social`).
   - `text_simulation/llm_helper.py`: Helper functions for LLM interactions.
   - `text_simulation/postprocess_responses.py` / `run_postprocess.py`: Parses raw LLM JSON into the structured `answer_blocks_llm_imputed/` files used by the evaluator.
   - Outputs land in `text_simulation/text_simulation_output_skill_v{1,2,3}/` (and per-model variants such as `_gpt4o`, `_o4mini`, `_nano_temp0`).

6. **Evaluation** — three metrics, plus paired-bootstrap power analysis
   - **Overall accuracy** — Mean Absolute Deviation against wave-4 ground truth across **all** wave-4 questions. Computed by `evaluation/mad_accuracy_evaluation.py`, aggregated by `evaluation/eval_temp0_suite.py` / `eval_all_experiments.py`. Reported with 95% CI, random baseline, and human ceiling.
   - **Cognitive bias** — Accuracy restricted to the cognitive-bias question subset (all wave-4 columns *excluding* the pricing/product columns). Computed inside `evaluation/paired_bootstrap.py` as the `cognitive_bias` slice. Companion plot: `cog_bias_raw_vs_v3bg_scatter.png`.
   - **Product preference** — Accuracy restricted to the pricing/product-preference column set (the `pricing` slice in `paired_bootstrap.py`). This isolates willingness-to-pay / product-choice questions.
   - **Paired bootstrap (power analysis)** — `evaluation/paired_bootstrap.py`, `paired_bootstrap_50p.py`, `paired_bootstrap_v2_vs_raw.py`, `paired_bootstrap_v2_vs_v4.py`. Each persona is treated as its own pair across two conditions (e.g. `bg` vs `bg+ep`, `v2` vs `raw`, `v2` vs `v4`); the script resamples persona-level paired differences to produce CIs and p-values for each of the three metrics, giving the per-condition power analysis used to compare skill versions and ablations. `evaluation/bootstrap_ci.py` provides the underlying CI helper.

## Requirements

- Python 3.11.7 or higher
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd digital-twin-simulation
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Quick Start: Simulate Responses to New Questions

For a quick introduction to digital twin simulation, try our interactive demo notebook:

```bash
jupyter notebook notebooks/demo_simple_simulation.ipynb
```

This notebook demonstrates:
- Loading persona summaries directly from Hugging Face dataset (no setup required!)
- Creating custom survey questions
- Simulating responses using GPT-4.1 mini
- Running batch simulations for multiple personas
- Automatic package installation and API key configuration
- Works seamlessly in both local environments and Google Colab

Perfect for researchers who want to quickly test new survey questions on digital twins without complex setup.

### Full Pipeline Demo (Interactive Alternative)

For those who prefer Jupyter notebooks over shell scripts, we provide a complete pipeline walkthrough:

```bash
jupyter notebook notebooks/demo_full_pipeline.ipynb
```

This notebook covers the entire workflow from data preparation to evaluation, making it an excellent alternative to the shell script approach described below.

### Full Pipeline (Command Line)

To run the complete digital twin simulation pipeline:

1.  **Prepare the Data**:
    First, download the necessary dataset by executing the following command:
    ```bash
    poetry run python download_dataset.py
    ```

2.  **Configure API Access**:
    Set the `OPENAI_API_KEY` environment variable to enable LLM interactions. Create a file named `.env` in the project's root directory and add your API key as follows:
    ```
    OPENAI_API_KEY=your_actual_api_key_here
    ```
    *Replace `your_actual_api_key_here` with your valid OpenAI API key.*

3.  **Extract Skill Personas (v1 / v2 / v3)**:
    Build the three-version skill profile per persona. `ANTHROPIC_API_KEY` must be set.
    ```bash
    # Single persona, all three versions
    poetry run python skill_extraction/extract_skills.py --pid 1

    # Batch extraction
    poetry run python skill_extraction/batch_extract_skills_v2.py
    ```
    Outputs land in `text_simulation/skills/pid_{N}/v{1_direct,2_inferred,3_maximum}/`.

4.  **Build Simulation Inputs**:
    Merge each skill version with the question file into the model-ready input directory.
    ```bash
    poetry run python text_simulation/create_skill_inputs.py        # → text_simulation_input_skill_v{1,2,3}/
    ```

5.  **Run the Simulation Pipeline**:
    Execute the main simulation pipeline using the provided shell scripts. You can run a small test with a limited number of personas or simulate all available personas.

    *   For a small test run (e.g., 5 personas):
        ```bash
        ./scripts/run_pipeline.sh --max_personas=5
        ```
    *   To run the simulation for all 2058 personas:
        ```bash
        ./scripts/run_pipeline.sh
        ```

6.  **Evaluate the Results**:
    After running the simulations, evaluate the results using:
    ```bash
    ./scripts/run_evaluation_pipeline.sh
    ```
    For the three-metric breakdown (overall accuracy, cognitive bias, product preference) and paired-bootstrap power analysis comparing skill versions or ablations:
    ```bash
    # Per-config accuracy table
    poetry run python evaluation/eval_temp0_suite.py --suite nano_temp0

    # Paired bootstrap across the three metrics
    poetry run python evaluation/paired_bootstrap.py
    poetry run python evaluation/paired_bootstrap_v2_vs_raw.py
    poetry run python evaluation/paired_bootstrap_v2_vs_v4.py
    ```
    See `evaluation/HOW_TO_EVALUATE.md` for the full per-suite walkthrough.
