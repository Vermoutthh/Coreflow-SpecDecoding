# OWL-HumanEval ReAct semantic trace

This is a local working copy of the OWL-HumanEval trace for speculative-decoding experiments.

- `raw_logs/test_<id>.log`: one complete OWL execution log per HumanEval task, including agent labels, full saved model contexts, responses, and token usage.
- `reference_trace/node_list.json`: the original token-length node list.
- `reference_trace/node_dependency.json`: the original dependency graph.

Source: `/root/nfs/owl/humaneval_dataset/`. The copy contains 100 task logs and the reference trace has 798 LLM requests (287 `user_agent`, 511 `assistant_agent`). The source files were copied without transformation.

The current speculative-decoding profiler can read the logs with `--data-dir datasets/owl_humaneval_react/raw_logs` when invoked from the project root. Its default still points to OWL-GAIA; the existing sweep driver also uses that default until configured separately. As with OWL-GAIA, profiling replays each saved LLM context and does not feed locally generated responses into later workflow steps.
