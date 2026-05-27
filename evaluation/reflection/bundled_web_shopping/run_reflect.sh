#!/bin/bash
# run_reflect.sh — Start VLLM server, run BWS Reflection Task, then auto-shutdown.
#
# Usage:
#   bash run_reflect.sh [MODEL] [GAMES]
#
# Defaults: MODEL=qwen3-32b, GAMES=all (all 5 BWS categories)

set -euo pipefail

WORKDIR="/home/jiahao_huang/ROGUE/BundledWebShopping/reflect"
LOGDIR="$WORKDIR/output/logs"
MODEL="${1:-qwen3-32b}"
GAMES="${2:-all}"

PORT=8001
GPUS="4,5"
VLLM_SESSION="vllm_reflect_bws"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── VLLM helpers ──────────────────────────────────────────────────────────────

start_vllm() {
    if tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
        log "Session $VLLM_SESSION already exists, reusing."
        return
    fi
    log "Starting VLLM: GPU $GPUS, port $PORT"
    tmux new-session -d -s "$VLLM_SESSION" \
        "CUDA_VISIBLE_DEVICES=$GPUS conda run -n Fraud \
         python -m vllm.entrypoints.openai.api_server \
         --model Qwen/Qwen3-32B \
         --served-model-name $MODEL \
         --tensor-parallel-size 2 \
         --max-model-len 32768 \
         --port $PORT \
         --trust-remote-code \
         --override-generation-config '{\"enable_thinking\": false}' \
         2>&1 | tee $LOGDIR/vllm_reflect.log"
}

wait_for_server() {
    log "Waiting for VLLM on port $PORT..."
    for i in $(seq 1 90); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            log "VLLM on port $PORT is ready."
            return 0
        fi
        sleep 10
    done
    log "ERROR: VLLM did not start within 15 minutes."
    return 1
}

kill_vllm() {
    log "Shutting down VLLM server..."
    tmux kill-session -t "$VLLM_SESSION" 2>/dev/null && log "Killed $VLLM_SESSION" || true
}

trap kill_vllm EXIT

# ── Main ──────────────────────────────────────────────────────────────────────

mkdir -p "$LOGDIR"

log "============================================"
log " run_reflect.sh (BWS)  MODEL=$MODEL  GAMES=$GAMES"
log "============================================"

start_vllm
wait_for_server

log "Starting Reflection Task..."
OPENAI_BASE_URL="http://localhost:${PORT}/v1" \
OPENAI_API_KEY="local" \
conda run -n Fraud --no-capture-output \
    python "$WORKDIR/run_reflect_task.py" \
        --model "$MODEL" \
        --api openai \
        --games $GAMES \
        --output_dir "$WORKDIR/output" \
        --workers 5 \
    2>&1 | tee "$LOGDIR/reflect_${MODEL}.log"

log "Reflection Task complete."
log "Running scorer..."
conda run -n Fraud --no-capture-output \
    python "$WORKDIR/score_reflect_task.py" \
        --results "$WORKDIR/output/${MODEL}_results.json" \
        --skip_llm_judge \
    2>&1 | tee "$LOGDIR/score_${MODEL}.log"

log "============================================"
log " All done! Results in $WORKDIR/output/"
log "============================================"
# VLLM server killed automatically by trap
