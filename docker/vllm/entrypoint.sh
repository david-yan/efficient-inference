#!/usr/bin/env bash
set -e

# Default configurations
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
PORT="${PORT:-8000}"
HOST="0.0.0.0"

EXTRA_ARGS=()

if [ -n "$LORA_MODULES" ]; then
    EXTRA_ARGS+=(--enable-lora --lora-modules $LORA_MODULES)
fi

if [ -n "$QUANTIZATION" ]; then
    EXTRA_ARGS+=(--quantization "$QUANTIZATION")
fi

if [ "$ENFORCE_EAGER" = "true" ]; then
    EXTRA_ARGS+=(--enforce-eager)
fi

echo "Starting vLLM engine for model: ${MODEL_PATH} (served as: ${SERVED_MODEL_NAME})"

exec python3 -m vllm.entrypoints.openai.api_server \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
