#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COREFLOW_DIR="${WORKSPACE_DIR}/CoreFlow_"
PROFILER="${COREFLOW_DIR}/scripts/profile_spec_decode.py"

# Every setting can be overridden from the command line, for example:
#   GPU_ID=0 WORKFLOW_LIMIT=2 KS="0 1" ./run_react_sweep.sh
GPU_ID="${GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/coreflow/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-2026}"
KS="${KS:-0 1 2 3 4 5 6 7 8}"
WORKFLOW_LIMIT="${WORKFLOW_LIMIT:-}"
WORKFLOW_SELECTION="${WORKFLOW_SELECTION:-trace}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
GPU_READY_ATTEMPTS="${GPU_READY_ATTEMPTS:-12}"
GPU_READY_INTERVAL_SECONDS="${GPU_READY_INTERVAL_SECONDS:-5}"
MAX_TASK_ATTEMPTS="${MAX_TASK_ATTEMPTS:-3}"
ENGINE_COOLDOWN_SECONDS="${ENGINE_COOLDOWN_SECONDS:-5}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-unsloth/Llama-3.2-1B-Instruct}"

RESULTS_ROOT="${COREFLOW_DIR}/results/spec_decode/react/${RUN_ID}"
MANIFEST="${RESULTS_ROOT}/workflows.tsv"
mkdir -p "${RESULTS_ROOT}"

# Capture the driver-level output as well as each individual k run's log.
exec > >(tee -a "${RESULTS_ROOT}/sweep.log") 2>&1

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${PROFILER}" ]]; then
  echo "Profiler not found: ${PROFILER}" >&2
  exit 1
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer: ${GPU_ID}" >&2
  exit 1
fi
wait_for_gpu() {
  local attempt
  for ((attempt = 1; attempt <= GPU_READY_ATTEMPTS; attempt++)); do
    if nvidia-smi --id="${GPU_ID}" >/dev/null 2>&1; then
      return 0
    fi
    echo "GPU/NVML not ready (${attempt}/${GPU_READY_ATTEMPTS}); retrying in ${GPU_READY_INTERVAL_SECONDS}s" >&2
    sleep "${GPU_READY_INTERVAL_SECONDS}"
  done
  return 1
}

if ! wait_for_gpu; then
  echo "GPU ${GPU_ID} remained unavailable; no workflow was started" >&2
  exit 75
fi

cat > "${RESULTS_ROOT}/run_config.env" <<EOF
RUN_ID=${RUN_ID}
GPU_ID=${GPU_ID}
PYTHON_BIN=${PYTHON_BIN}
SEED=${SEED}
KS=${KS}
WORKFLOW_LIMIT=${WORKFLOW_LIMIT}
WORKFLOW_SELECTION=${WORKFLOW_SELECTION}
WARMUP_REQUESTS=${WARMUP_REQUESTS}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_TOKENS=${MAX_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
GPU_READY_ATTEMPTS=${GPU_READY_ATTEMPTS}
GPU_READY_INTERVAL_SECONDS=${GPU_READY_INTERVAL_SECONDS}
MAX_TASK_ATTEMPTS=${MAX_TASK_ATTEMPTS}
ENGINE_COOLDOWN_SECONDS=${ENGINE_COOLDOWN_SECONDS}
TARGET_MODEL=${TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
COREFLOW_DIR=${COREFLOW_DIR}
EOF

read -r -a K_LIST <<< "${KS}"

cd "${COREFLOW_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  MANIFEST_ARGS=(
    --dry-run
    --selection-unit workflow
    --workflow-selection "${WORKFLOW_SELECTION}"
    --workflow-manifest-output "${MANIFEST}"
    --seed "${SEED}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-tokens "${MAX_TOKENS}"
  )
  if [[ -n "${WORKFLOW_LIMIT}" ]]; then
    MANIFEST_ARGS+=(--workflow-limit "${WORKFLOW_LIMIT}")
  fi
  echo "Generating ordered workflow manifest"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" "${PROFILER}" "${MANIFEST_ARGS[@]}" \
    2>&1 | tee "${RESULTS_ROOT}/manifest.log"
else
  echo "Reusing workflow manifest: ${MANIFEST}"
fi

mapfile -t WORKFLOW_ROWS < <(tail -n +2 "${MANIFEST}")
if [[ "${#WORKFLOW_ROWS[@]}" -eq 0 ]]; then
  echo "Workflow manifest is empty: ${MANIFEST}" >&2
  exit 1
fi

echo "Results: ${RESULTS_ROOT}"
echo "GPU: ${GPU_ID}"
echo "Speculative-token order per workflow: ${KS}"
echo "Workflows: ${#WORKFLOW_ROWS[@]}"

workflow_number=0
for workflow_row in "${WORKFLOW_ROWS[@]}"; do
  workflow_number=$((workflow_number + 1))
  IFS=$'\t' read -r workflow_id request_count max_context <<< "${workflow_row}"
  printf -v workflow_index '%04d' "${workflow_number}"
  WORKFLOW_DIR="${RESULTS_ROOT}/workflows/${workflow_index}_${workflow_id}"
  WORKFLOW_STATUS="${WORKFLOW_DIR}/status.txt"
  mkdir -p "${WORKFLOW_DIR}"

  if [[ -f "${WORKFLOW_STATUS}" ]] && \
    [[ "$(head -n 1 "${WORKFLOW_STATUS}")" == "COMPLETE" ]]; then
    echo "Skipping completed workflow ${workflow_number}/${#WORKFLOW_ROWS[@]}: ${workflow_id}"
    continue
  fi

  cat > "${WORKFLOW_DIR}/workflow.env" <<EOF
WORKFLOW_INDEX=${workflow_index}
WORKFLOW_ID=${workflow_id}
REQUEST_COUNT=${request_count}
MAX_CONTEXT_TOKENS=${max_context}
EOF
  printf 'RUNNING\nstarted_at=%s\n' \
    "$(date --iso-8601=seconds)" > "${WORKFLOW_STATUS}"
  echo "===== Workflow ${workflow_number}/${#WORKFLOW_ROWS[@]}: ${workflow_id} (${request_count} requests) ====="

  for k in "${K_LIST[@]}"; do
    TASK_DIR="${WORKFLOW_DIR}/k_${k}"
    STATUS_FILE="${TASK_DIR}/status.txt"
    mkdir -p "${TASK_DIR}"

    if [[ -f "${STATUS_FILE}" ]] && \
      [[ "$(head -n 1 "${STATUS_FILE}")" == "COMPLETE" ]]; then
      echo "Skipping completed k=${k} for workflow=${workflow_id}"
      continue
    fi

    task_succeeded=0
    exit_code=1
    for ((task_attempt = 1; task_attempt <= MAX_TASK_ATTEMPTS; task_attempt++)); do
      echo "----- Starting workflow=${workflow_id}, k=${k}, attempt=${task_attempt}/${MAX_TASK_ATTEMPTS} at $(date --iso-8601=seconds) -----"
      printf 'RUNNING\nattempt=%s\nstarted_at=%s\n' \
        "${task_attempt}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"

      if ! wait_for_gpu; then
        exit_code=75
        echo "GPU ${GPU_ID} did not recover before workflow=${workflow_id}, k=${k}" >&2
        break
      fi

      printf '\n===== attempt %s/%s at %s =====\n' \
        "${task_attempt}" "${MAX_TASK_ATTEMPTS}" \
        "$(date --iso-8601=seconds)" >> "${TASK_DIR}/run.log"

      if CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        "${PYTHON_BIN}" "${PROFILER}" \
          --target-model "${TARGET_MODEL}" \
          --draft-model "${DRAFT_MODEL}" \
          --num-speculative-tokens "${k}" \
          --selection-unit workflow \
          --workflow-id "${workflow_id}" \
          --seed "${SEED}" \
          --warmup-requests "${WARMUP_REQUESTS}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-tokens "${MAX_TOKENS}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
          --temperature 0 \
          --enforce-eager \
          --output "${TASK_DIR}/metrics.jsonl" \
          2>&1 | tee -a "${TASK_DIR}/run.log"; then
        task_succeeded=1
        break
      fi

      pipeline_status=("${PIPESTATUS[@]}")
      exit_code="${pipeline_status[0]}"
      if [[ "${exit_code}" -eq 0 ]]; then
        exit_code="${pipeline_status[1]}"
      fi
      if [[ "${task_attempt}" -lt "${MAX_TASK_ATTEMPTS}" ]] && \
        rg -q "Can't initialize NVML|No platform detected|Device string must not be empty|No CUDA GPUs are available" "${TASK_DIR}/run.log"; then
        echo "Transient CUDA/NVML initialization failure; cooling down before retry" >&2
        sleep "${ENGINE_COOLDOWN_SECONDS}"
        continue
      fi
      break
    done

    if [[ "${task_succeeded}" -eq 1 ]]; then
      printf 'COMPLETE\nfinished_at=%s\n' \
        "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
      echo "----- Completed workflow=${workflow_id}, k=${k} -----"
      sleep "${ENGINE_COOLDOWN_SECONDS}"
    else
      printf 'FAILED\nexit_code=%s\nfinished_at=%s\n' \
        "${exit_code}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
      printf 'FAILED\nk=%s\nexit_code=%s\n' \
        "${k}" "${exit_code}" > "${WORKFLOW_STATUS}"
      echo "Run failed: workflow=${workflow_id}, k=${k}, exit=${exit_code}" >&2
      exit "${exit_code}"
    fi
  done

  printf 'COMPLETE\nfinished_at=%s\n' \
    "$(date --iso-8601=seconds)" > "${WORKFLOW_STATUS}"
  echo "===== Completed all k values for workflow=${workflow_id} ====="
done

echo "Sweep completed: ${RESULTS_ROOT}"
