#!/usr/bin/env bash
set -e

OPENAI_API_BASE="${OPENAI_API_BASE:-http://vllm-baseline-service:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-qwen-baseline}"
DOMAIN="${DOMAIN:-airline}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/gcs/benchmarks/tau2-bench/$(date +%Y%m%d_%H%M%S)}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"

export OPENAI_API_BASE
export OPENAI_API_KEY

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Starting tau2-bench evaluation run"
echo "Target Endpoint : ${OPENAI_API_BASE}"
echo "Model Name      : ${MODEL_NAME}"
echo "Domain          : ${DOMAIN}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=================================================="

# Check endpoint connectivity
until curl -s "${OPENAI_API_BASE}/models" > /dev/null; do
    echo "Waiting for LLM endpoint at ${OPENAI_API_BASE} to be ready..."
    sleep 5
done

echo "LLM Endpoint is reachable. Running tau2 benchmark..."

tau2 run \
    --agent-llm "${MODEL_NAME}" \
    --domain "${DOMAIN}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-concurrency "${MAX_CONCURRENCY}"

echo "tau2 benchmark run completed successfully."
echo "Results and trajectories saved to ${OUTPUT_DIR}."
