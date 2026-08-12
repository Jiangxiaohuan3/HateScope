#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ARGS=()
if [[ -n "${MODEL_API_BASE:-}" || -n "${MODEL_API_MODEL:-}" ]]; then
  : "${MODEL_API_BASE:?Set MODEL_API_BASE when using a candidate API.}"
  : "${MODEL_API_MODEL:?Set MODEL_API_MODEL when using a candidate API.}"
  if [[ -z "${MODEL_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set MODEL_API_KEY or OPENAI_API_KEY when using a candidate API." >&2
    exit 1
  fi
  MODEL_ARGS=(
    --model-api-base "${MODEL_API_BASE}"
    --model-api-model "${MODEL_API_MODEL}"
    --model-api-workers "${MODEL_API_WORKERS:-8}"
  )
else
  : "${MODEL_PATH:?Set MODEL_PATH to the local candidate model path or model id.}"
  MODEL_ARGS=(--model-path "${MODEL_PATH}")
fi

JUDGE_ARGS=()
if [[ -n "${JUDGE_API_BASE:-}" || -n "${JUDGE_API_MODEL:-}" ]]; then
  : "${JUDGE_API_BASE:?Set JUDGE_API_BASE when using an API judge.}"
  : "${JUDGE_API_MODEL:?Set JUDGE_API_MODEL when using an API judge.}"
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${JUDGE_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY or JUDGE_API_KEY when using an API judge." >&2
    exit 1
  fi
  JUDGE_ARGS=(
    --judge-api-base "${JUDGE_API_BASE}"
    --judge-api-model "${JUDGE_API_MODEL}"
    --judge-api-workers "${JUDGE_API_WORKERS:-8}"
  )
else
  : "${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH to the local DeepSeek-V4-Flash path or model id.}"
  JUDGE_ARGS=(
    --judge-model-path "${JUDGE_MODEL_PATH}"
    --judge-tensor-parallel-size "${JUDGE_TP_SIZE:-8}"
  )
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python3 "${REPO_ROOT}/evaluation/evaluate_hatescope.py" \
  --data "${REPO_ROOT}/data/hatescope_1330.jsonl" \
  --output-root "${OUTPUT_ROOT:-${REPO_ROOT}/outputs}" \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --dtype "${DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens "${MAX_TOKENS:-4096}" \
  --guided-json \
  "${MODEL_ARGS[@]}" \
  "${JUDGE_ARGS[@]}" \
  "$@"
