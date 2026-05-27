#!/bin/bash
# Run all BundledWebShopping Evolution Evaluation baselines sequentially.
# Results saved to: PROJECT_ROOT/main_result/BundledWebShopping/{game}/{baseline}/
# Usage: nohup bash run_all_evaleval.sh > logs/run_all_evaleval.log 2>&1 &

set -e

MODEL="${MODEL:-qwen3-32b}"
PYTHON="${PYTHON:-/home/jiahao_huang/miniconda3/envs/Fraud/bin/python}"
RUNNER_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$RUNNER_DIR/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/main_result/BundledWebShopping"
GAMES="baking beauty electronics grocery home"

mkdir -p "$RUNNER_DIR/logs"

run_baseline() {
    local baseline=$1
    local runner=$2
    shift 2
    local extra_args=("$@")

    echo "========================================"
    echo "BASELINE: $baseline"
    echo "========================================"
    for game in $GAMES; do
        local output_dir="$OUTPUT_ROOT/$game/$baseline"
        mkdir -p "$output_dir"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $baseline / $game"
        "$PYTHON" "$RUNNER_DIR/$runner" \
            --game "$game" \
            --model "$MODEL" \
            --distances 1 5 \
            --output_dir "$output_dir" \
            "${extra_args[@]}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $baseline / $game"
        echo
    done
}

run_baseline non_evolution  run_non_evolution.py
run_baseline react          run_react.py
run_baseline naive          run_naive.py
run_baseline reflexion      run_reflexion.py
run_baseline rag            run_rag.py
run_baseline remem          run_remem.py
run_baseline memrl          run_memrl.py
run_baseline autoskill      run_autoskill.py
run_baseline evotest        run_evotest.py  --evo_model "$MODEL"

echo "========================================"
echo "ALL DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
