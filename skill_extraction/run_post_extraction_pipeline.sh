#!/bin/bash
# run_post_extraction_pipeline.sh — Full pipeline after skills_v2 extraction completes.
#
# Steps:
#   1. Regenerate input dirs (skills_v2 now has pids 1-50)
#   2. Generate simulation configs (50 personas, high reasoning, v2+v3, 4 conditions)
#   3. Run simulation batches
#   4. Re-download with correct parser
#   5. Postprocess + evaluate (overall + by question type)
#
# Usage (from Digital-Twin-Simulation/):
#   bash skill_extraction/run_post_extraction_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== Step 1: Create input directories for pids 21-50 only ====="
python skill_extraction/create_ablation_50p_inputs.py

echo ""
echo "===== Step 2: Generate simulation configs ====="
python skill_extraction/generate_ablation_50p_configs.py

echo ""
echo "===== Step 3: Run simulation batches ====="
bash skill_extraction/run_ablation_50p_temp0.sh

echo ""
echo "===== Step 4: Re-download with correct parser ====="
python skill_extraction/redownload_ablation_50p_results.py

echo ""
echo "===== Step 5: Postprocess + Evaluate ====="
for version in v2 v3; do
    suite="nano_v2_ablation_50p_${version}_temp0"
    echo "--- $suite ---"
    python text_simulation/run_postprocess.py --suite "$suite" --reasoning high
    python evaluation/eval_temp0_suite.py     --suite "$suite" --reasoning high
    python evaluation/eval_by_question_type.py --suite "$suite" --reasoning high
done

echo ""
echo "===== Pipeline complete ====="
