#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COREFLOW_DIR="${WORKSPACE_DIR}/CoreFlow_"
PROFILER="${COREFLOW_DIR}/scripts/profile_spec_decode.py"

# Qwen2.5-32B target + Qwen2.5-7B draft on four 24 GiB GPUs.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/coreflow/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-2026}"
KS="${KS:-0 1 2 3 4 5 6 7 8}"
WORKFLOW_LIMIT="${WORKFLOW_LIMIT:-20}"
WORKFLOW_SELECTION="${WORKFLOW_SELECTION:-trace}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DRAFT_TENSOR_PARALLEL_SIZE="${DRAFT_TENSOR_PARALLEL_SIZE:-4}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_TASK_ATTEMPTS="${MAX_TASK_ATTEMPTS:-3}"
GPU_READY_ATTEMPTS="${GPU_READY_ATTEMPTS:-12}"
GPU_READY_INTERVAL_SECONDS="${GPU_READY_INTERVAL_SECONDS:-5}"
ENGINE_COOLDOWN_SECONDS="${ENGINE_COOLDOWN_SECONDS:-10}"

RESULTS_ROOT="${COREFLOW_DIR}/results/spec_decode/react_qwen25_32b/${RUN_ID}"
MANIFEST="${RESULTS_ROOT}/workflows.tsv"
mkdir -p "${RESULTS_ROOT}"
exec > >(tee -a "${RESULTS_ROOT}/sweep.log") 2>&1

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${PROFILER}" ]]; then
  echo "Profiler not found: ${PROFILER}" >&2
  exit 1
fi
if ! [[ "${WORKFLOW_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKFLOW_LIMIT must be a positive integer: ${WORKFLOW_LIMIT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if [[ "${#GPU_LIST[@]}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
  echo "GPU_IDS contains ${#GPU_LIST[@]} GPUs but TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}" >&2
  exit 1
fi

wait_for_gpus() {
  local attempt gpu_id ready
  for ((attempt = 1; attempt <= GPU_READY_ATTEMPTS; attempt++)); do
    ready=1
    for gpu_id in "${GPU_LIST[@]}"; do
      if ! nvidia-smi --id="${gpu_id}" >/dev/null 2>&1; then
        ready=0
        break
      fi
    done
    if [[ "${ready}" -eq 1 ]]; then
      return 0
    fi
    echo "GPU/NVML not ready (${attempt}/${GPU_READY_ATTEMPTS}); retrying in ${GPU_READY_INTERVAL_SECONDS}s" >&2
    sleep "${GPU_READY_INTERVAL_SECONDS}"
  done
  return 1
}

if ! wait_for_gpus; then
  echo "The requested GPUs remained unavailable; no k run was started" >&2
  exit 75
fi

cat > "${RESULTS_ROOT}/run_config.env" <<EOF
RUN_ID=${RUN_ID}
GPU_IDS=${GPU_IDS}
PYTHON_BIN=${PYTHON_BIN}
SEED=${SEED}
KS=${KS}
WORKFLOW_LIMIT=${WORKFLOW_LIMIT}
WORKFLOW_SELECTION=${WORKFLOW_SELECTION}
WARMUP_REQUESTS=${WARMUP_REQUESTS}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_TOKENS=${MAX_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
DRAFT_TENSOR_PARALLEL_SIZE=${DRAFT_TENSOR_PARALLEL_SIZE}
TARGET_MODEL=${TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
MAX_TASK_ATTEMPTS=${MAX_TASK_ATTEMPTS}
GPU_READY_ATTEMPTS=${GPU_READY_ATTEMPTS}
GPU_READY_INTERVAL_SECONDS=${GPU_READY_INTERVAL_SECONDS}
ENGINE_COOLDOWN_SECONDS=${ENGINE_COOLDOWN_SECONDS}
COREFLOW_DIR=${COREFLOW_DIR}
EOF

read -r -a K_LIST <<< "${KS}"
cd "${COREFLOW_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Generating the fixed ${WORKFLOW_LIMIT}-workflow manifest"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "${PYTHON_BIN}" "${PROFILER}" \
      --dry-run \
      --target-model "${TARGET_MODEL}" \
      --draft-model "${DRAFT_MODEL}" \
      --selection-unit workflow \
      --workflow-selection "${WORKFLOW_SELECTION}" \
      --workflow-limit "${WORKFLOW_LIMIT}" \
      --workflow-manifest-output "${MANIFEST}" \
      --seed "${SEED}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-tokens "${MAX_TOKENS}" \
      2>&1 | tee "${RESULTS_ROOT}/manifest.log"
else
  echo "Reusing workflow manifest: ${MANIFEST}"
fi

workflow_count=$(( $(wc -l < "${MANIFEST}") - 1 ))
expected_requests=$(awk -F '\t' 'NR > 1 { total += $2 } END { print total + 0 }' "${MANIFEST}")
if [[ "${workflow_count}" -ne "${WORKFLOW_LIMIT}" ]]; then
  echo "Expected ${WORKFLOW_LIMIT} workflows but manifest has ${workflow_count}" >&2
  exit 1
fi

echo "Results: ${RESULTS_ROOT}"
echo "GPUs: ${GPU_IDS}; target TP=${TENSOR_PARALLEL_SIZE}; draft TP=${DRAFT_TENSOR_PARALLEL_SIZE}"
echo "Fixed workload: ${workflow_count} workflows, ${expected_requests} requests"
echo "k sweep order: ${KS}"

for k in "${K_LIST[@]}"; do
  TASK_DIR="${RESULTS_ROOT}/k_${k}"
  STATUS_FILE="${TASK_DIR}/status.txt"
  mkdir -p "${TASK_DIR}"

  if [[ -f "${STATUS_FILE}" ]] && \
    [[ "$(head -n 1 "${STATUS_FILE}")" == "COMPLETE" ]]; then
    echo "Skipping completed k=${k}"
    continue
  fi

  task_succeeded=0
  exit_code=1
  for ((task_attempt = 1; task_attempt <= MAX_TASK_ATTEMPTS; task_attempt++)); do
    echo "===== Starting k=${k}, attempt=${task_attempt}/${MAX_TASK_ATTEMPTS} at $(date --iso-8601=seconds) ====="
    printf 'RUNNING\nattempt=%s\nstarted_at=%s\n' \
      "${task_attempt}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"

    if ! wait_for_gpus; then
      exit_code=75
      break
    fi

    printf '\n===== attempt %s/%s at %s =====\n' \
      "${task_attempt}" "${MAX_TASK_ATTEMPTS}" \
      "$(date --iso-8601=seconds)" >> "${TASK_DIR}/run.log"

    if CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      "${PYTHON_BIN}" "${PROFILER}" \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --num-speculative-tokens "${k}" \
        --selection-unit workflow \
        --workflow-selection "${WORKFLOW_SELECTION}" \
        --workflow-limit "${WORKFLOW_LIMIT}" \
        --seed "${SEED}" \
        --warmup-requests "${WARMUP_REQUESTS}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-tokens "${MAX_TOKENS}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --draft-tensor-parallel-size "${DRAFT_TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --temperature 0 \
        --enforce-eager \
        --output "${TASK_DIR}/metrics.jsonl" \
        2>&1 | tee -a "${TASK_DIR}/run.log"; then
      actual_requests=$(wc -l < "${TASK_DIR}/metrics.jsonl")
      if [[ "${actual_requests}" -eq "${expected_requests}" ]]; then
        task_succeeded=1
        break
      fi
      exit_code=65
      echo "k=${k} produced ${actual_requests}/${expected_requests} rows" >&2
    else
      pipeline_status=("${PIPESTATUS[@]}")
      exit_code="${pipeline_status[0]}"
      if [[ "${exit_code}" -eq 0 ]]; then
        exit_code="${pipeline_status[1]}"
      fi
    fi

    if [[ "${task_attempt}" -lt "${MAX_TASK_ATTEMPTS}" ]] && \
      rg -q "Can't initialize NVML|No platform detected|Device string must not be empty|No CUDA GPUs are available" "${TASK_DIR}/run.log"; then
      echo "Transient CUDA/NVML failure; cooling down before retry" >&2
      sleep "${ENGINE_COOLDOWN_SECONDS}"
      continue
    fi
    break
  done

  if [[ "${task_succeeded}" -eq 1 ]]; then
    printf 'COMPLETE\nrequests=%s\nfinished_at=%s\n' \
      "${expected_requests}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
    echo "===== Completed k=${k}: ${expected_requests} requests from ${workflow_count} workflows ====="
    sleep "${ENGINE_COOLDOWN_SECONDS}"
  else
    printf 'FAILED\nexit_code=%s\nfinished_at=%s\n' \
      "${exit_code}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
    echo "k=${k} failed with exit=${exit_code}" >&2
    exit "${exit_code}"
  fi
done

echo "Qwen2.5-32B sweep completed: ${RESULTS_ROOT}"
