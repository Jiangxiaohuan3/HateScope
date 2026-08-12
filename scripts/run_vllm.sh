#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the candidate model path or model id.}"
: "${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH to the local DeepSeek-V4-Flash path or model id.}"

python3 "${REPO_ROOT}/evaluation/evaluate_hatescope.py" \
  --model-path "${MODEL_PATH}" \
  --judge-model-path "${JUDGE_MODEL_PATH}" \
  --data "${REPO_ROOT}/data/hatescope_refined.jsonl" \
  --output-root "${OUTPUT_ROOT:-${REPO_ROOT}/outputs}" \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --judge-tensor-parallel-size "${JUDGE_TP_SIZE:-8}" \
  --dtype "${DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens "${MAX_TOKENS:-4096}" \
  --guided-json \
  "$@"
