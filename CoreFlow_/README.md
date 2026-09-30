# CoreFlow

CoreFlow is a context-centric resource management prototype for LLM-based
agent workflows. 

## Requirements

The artifact was prepared for Ubuntu 20.04 with Python 3.10, CUDA-capable
NVIDIA GPUs, and a conda environment. The main software dependencies are:

- vLLM 0.9.2
- FastAPI 0.121.0
- SciPy 1.8 or newer
- Ray 2.53.0
- Redis 7.4.0
- seaborn 0.13.2

Install CUDA-compatible PyTorch before installing vLLM if your environment does
not already provide it.

## Installation

Create and activate a clean conda environment:

```bash
conda create -n coreflow python=3.10 -y
conda activate coreflow
```

Install CoreFlow and the bundled vLLM source in editable mode:

```bash
cd /path/to/CoreFlow_
bash requirement.sh
```

The installation script runs:

```bash
python -m pip install -e .
python -m pip install -e ./src/vllm
```

If Ray is used across multiple hosts, start the Ray cluster separately on each
host before running the end-to-end serving experiment.

## Artifact Tasks

Run the following commands from the repository root.

T1: heterogeneous request interference analysis:

```bash
python scripts/interference_analyze.py \
  --kernel FlashAttention \
  --model Llama-3-8B \
  --device cuda:0 \
  --out-dir results/t1_interference
```

T2: cache reuse throughput benefit analysis:

```bash
bash scripts/reuse_benefits.sh
```

The T2 wrapper defaults to the full CUDA graph run with
`max_model_len=32768`, `gpu_memory_utilization=0.9`, `rate=5`,
`duration=300`, `output_tokens=106`, and queue statistics enabled in
`results/t2_throughput/instance.log`. Extra arguments are passed through to
`scripts/reuse_benefits.py`, for example:

```bash
bash scripts/reuse_benefits.sh \
  --out-dir results/t2_throughput_debug \
  --input-tokens 2000 \
  --ratios 0.5 \
  --duration 60
```

T3: offline profiling, resource allocation, controller startup, and workload
submission:

```bash
# 1. offline profiling
python scripts/model_profile.py --device cuda:0 --output-dir Profiler

# 2. resource allocation
python scripts/allocate.py \
  --gpus 32 \
  --data-dir data \
  --profile-dir Profiler \
  --output allocation.json

# 3. start controller and instances described by allocation.json, then
#    generate/submit workload.  GPU devices are declared with the standard
#    CUDA_VISIBLE_DEVICES environment variable.
CUDA_VISIBLE_DEVICES=1,2 \
T3_MODEL=/data/liaojianxiong/hf_config/Qwen2.5-7B \
bash scripts/t3_e2e.sh
```

## Speculative-decoding trace profiling

`scripts/profile_spec_decode.py` replays the saved semantic contexts from the
OWL-GAIA ReAct trace. It uses the target tokenizer to measure context length
and records request-level acceptance-position counters and latency metrics.
The default data location is `../datasets/owl_gaia_react/raw_logs`.

Inspect and validate the trace without loading a GPU model:

```bash
python scripts/profile_spec_decode.py \
  --dry-run \
  --max-model-len 8192 \
  --max-tokens 128
```

Run one speculative-decoding configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_spec_decode.py \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model unsloth/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 4 \
  --max-model-len 8192 \
  --max-tokens 128 \
  --gpu-memory-utilization 0.8 \
  --output results/spec_profile/react_k4.jsonl
```

Use `--num-speculative-tokens 0` with the same workflow selection for the
target-only baseline. Each value of `k` is run in a separate process because
vLLM sets the speculative configuration when it constructs the engine. The
default selection unit is a complete query workflow: all saved Plan and Task
calls are replayed in trace order, and a workflow is excluded if any call is
over capacity. A useful smoke test is
`--workflow-selection shortest --workflow-limit 1`. Request-level filters are
still available for micro-profiling with `--selection-unit request`.

Keep the initial single-GPU runs at the tested `max_model_len=8192`. With vLLM
V0, reducing this value can increase the number of allocated KV-cache blocks
and is not necessarily a lower-memory configuration. The tested 24 GiB setup
uses `gpu_memory_utilization=0.8`; `0.9` can leave too little headroom for the
1B draft model because V0 sizes the target KV cache without fully accounting
for the proposer's weights.

The output contains the original workflow identity, target-tokenizer context
length, generated length, per-position acceptance counts, weighted acceptance
components, TTFT, TPOT, decode time, and end-to-end latency. Generated text is
not retained; a SHA-256 hash of its token IDs is stored so baseline and
speculative runs can be checked for lossless equivalence.
