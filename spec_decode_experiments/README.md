# ReAct speculative-decoding sweep

The default sweep processes all eligible workflows in the stable
`test_<query_id>.log` order. For each workflow it evaluates every integer `k`
from 0 through 8 before moving to the next workflow. Every saved Plan and Task
call is replayed in trace order. Agent and context-length categories are
recorded for analysis but are not used to sample requests.

```bash
cd /root/nfs/sc_expansion/spec_decode_experiments
./run_react_sweep.sh
```

Results are organized as:

```text
CoreFlow_/results/spec_decode/react/<run_id>/
├── run_config.env
├── workflows.tsv
├── sweep.log
└── workflows/
    ├── 0001_<query_id>/
    │   ├── workflow.env
    │   ├── status.txt
    │   ├── k_0/{metrics.jsonl,run.log,status.txt}
    │   ├── k_1/{metrics.jsonl,run.log,status.txt}
    │   └── ...
    └── 0002_<query_id>/...
```

Common overrides:

```bash
GPU_ID=0 WORKFLOW_LIMIT=20 ./run_react_sweep.sh
RUN_ID=my_run KS="0 1 2 3 4" ./run_react_sweep.sh
```

Set the same `RUN_ID` to resume an interrupted sweep. Runs whose `status.txt`
starts with `COMPLETE` are skipped; an incomplete or failed combination is
rerun and its `metrics.jsonl` is replaced.

A workflow is eligible only if all of its requests fit within the configured
context capacity. This avoids reporting a partial workflow after silently
dropping an over-capacity agent call.

`WORKFLOW_LIMIT` is unset by default, so no workflow sampling is performed.
It is available only for short debugging runs. The outer loop is workflow and
the inner loop is `k`, so a stopped run retains complete per-workflow results.
This ordering reloads the target/draft engine for each `k` of each workflow;
that is intentionally slower but gives workflow-level checkpoint granularity.
Because repeated engine reloads can briefly make NVML unavailable, the driver
checks GPU readiness before every attempt, waits five seconds between engine
runs, and retries CUDA/NVML initialization failures up to three times.

## Four-GPU Qwen2.5-32B sweep

`run_qwen32_sweep.sh` uses `Qwen2.5-32B-Instruct` as the target and the locally
available `Qwen2.5-7B-Instruct` as the draft. Both use tensor parallelism 4.
It fixes the first 20 eligible workflows, runs all of them for one `k`, and
only then reloads the engine for the next `k`:

```bash
cd /root/nfs/sc_expansion/spec_decode_experiments
./run_qwen32_sweep.sh
```

Results are stored under
`CoreFlow_/results/spec_decode/react_qwen25_32b/<run_id>/k_<k>/`. Set the same
`RUN_ID` to resume and skip completed `k` values.

# 双卡并发 SD 工作流回放（新实验）

`run_two_agent_concurrent.py` 在两张独立 GPU 上各启动一个真实权重的
vLLM V0 实例：Plan 请求路由到第一张，Task 请求路由到第二张。每个实例在
`K>0` 时同时装载 target 和 draft。**这不是**旧的 dummy 权重／强制输出
T3 链路，也不是把 target 和 draft 分别放在两张卡上。

先运行不保留结果的双卡 smoke：

```bash
/opt/conda/envs/coreflow/bin/python spec_decode_experiments/run_two_agent_concurrent.py \
  --smoke --workflow-selection shortest --plan-k 0 --task-k 0 --max-tokens 128
/opt/conda/envs/coreflow/bin/python spec_decode_experiments/run_two_agent_concurrent.py \
  --smoke --workflow-selection shortest --plan-k 1 --task-k 1 --max-tokens 128
```

正式运行示例（GPU 0 和 1 必须空闲，脚本会在退出时停止自己启动的服务）：

```bash
/opt/conda/envs/coreflow/bin/python spec_decode_experiments/run_two_agent_concurrent.py \
  --plan-gpu 0 --task-gpu 1 --plan-k 1 --task-k 1 \
  --workflow-selection random --seed 2026 --workflow-limit 20 \
  --arrival-rate 0.12 --max-inflight-workflows 16 \
  --prefix-caching
```

结果按数据集、Plan/Task 的 K 和运行时间归档于
`CoreFlow_/results/spec_decode/concurrent_2gpu/`，包含固定 workflow manifest、
每次调用与每条 workflow 的 JSONL、服务端指标快照、配置及日志。可用
`analyze_capacity.py <结果目录>` 汇总。运行中断时，已完成请求仍保留在
JSONL 中；对照不同 K 时应固定相同的数据、随机种子、到达速率、输出上限
和服务配置，并重复运行。

回放器保存 OWL 日志中的真实语义 prompt，并允许多条 workflow 同时进行；
同一条 workflow 内部严格按原调用顺序等待上一请求完成。本次生成的文本不
会改写后续保存的 prompt，所以这验证的是固定轨迹下的推理服务资源行为，
而非重新执行智能体任务。`--arrival-rate 0` 会立即释放所有选中 workflow，
适合 smoke，不应据此声称最大稳定吞吐。测容量要扫描正的 workflow 到达率，
并结合完成率、P95、排队情况和实际调度 batch 判断是否达到稳态。

单实例 profiling 代码仍可用于独立测量 agent/context/K 的接受分布；
但它固定 `max_num_seqs=1`，不能替代本实验。两卡在 Plan/Task 各需一个
实例时只能形成 1:1 的整数 GPU 分配，因此当前实验也不能直接验证
“旧分配 vs 新分配”的实机收益。
