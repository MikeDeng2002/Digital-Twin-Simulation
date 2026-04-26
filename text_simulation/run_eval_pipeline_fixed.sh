#!/bin/bash
# run_eval_pipeline_fixed.sh — Full evaluation pipeline for ablation_fixed suites.
#
# Steps (for each of v1/v2/v3):
#   1. Re-download batch results with correct parser (fixes empty response_text)
#   2. Postprocess (impute answers)
#   3. eval_temp0_suite  → overall accuracy + CSV
#   4. eval_by_question_type → cognitive bias + product preference accuracy + CSV
#
# Usage (from Digital-Twin-Simulation/):
#   bash text_simulation/run_eval_pipeline_fixed.sh
#   bash text_simulation/run_eval_pipeline_fixed.sh v2   # single version

set -euo pipefail

FILTER_VERSION="${1:-}"
VERSIONS="v1 v2 v3"

for version in $VERSIONS; do
    if [[ -n "$FILTER_VERSION" && "$version" != "$FILTER_VERSION" ]]; then continue; fi

    SUITE="nano_v2_ablation_fixed_${version}_temp0"
    echo ""
    echo "=========================================="
    echo "  Suite: $SUITE"
    echo "=========================================="

    # Step 1: re-download with correct parser
    echo "[1/4] Re-downloading batch results..."
    python text_simulation/redownload_ablation_fixed_results.py --version "$version"

    # Step 2: postprocess
    echo "[2/4] Postprocessing..."
    python text_simulation/run_postprocess.py --suite "$SUITE"

    # Step 3: overall accuracy
    echo "[3/4] Overall accuracy evaluation..."
    python evaluation/eval_temp0_suite.py --suite "$SUITE"

    # Step 4: by question type (cognitive + pricing)
    echo "[4/4] Cognitive & product preference accuracy..."
    python evaluation/eval_by_question_type.py --suite "$SUITE"

    echo "  Done: $SUITE"
done

echo ""
echo "===== Full evaluation pipeline complete ====="
