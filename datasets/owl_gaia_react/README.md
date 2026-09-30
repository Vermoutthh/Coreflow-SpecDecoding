# OWL-GAIA ReAct semantic trace

This directory is a local working copy of the OWL-GAIA trace used to derive
the ReAct workload statistics in CoreFlow.

## Contents

- `raw_logs/test_<task_id>.log`: complete per-query OWL logs. Every LLM call
  contains its agent name, repeat index, full semantic context, model output,
  and token usage.
- `metadata.jsonl`: GAIA test metadata. The 100 task IDs represented by the
  logs all have a matching metadata row and none of them requires a local
  attachment.
- `reference_trace/node_list.json`: the token-length trace derived by the
  original OWL parser.
- `reference_trace/node_dependency.json`: dependencies derived by the original
  OWL parser.

## Provenance

- Logs: `/root/nfs/owl/gaia_dataset/test_*.log`
- Metadata: `/root/nfs/owl/gaia/metadata.jsonl`
- Reference trace: `/root/nfs/owl/gaia_dataset/{node_list,node_dependency}.json`

The source files are unchanged. The copied trace contains 100 queries and
2,685 LLM requests: 938 `user_agent` requests and 1,747 `assistant_agent`
requests. In the paper's ReAct terminology these correspond to the Plan and
Task agents, respectively.

## Intended speculative-decoding use

For each request, replay the saved context as an independent prompt. Generate
only the current response with the target/draft model pair. This keeps the
original semantic history fixed and avoids compounding changes from earlier
locally generated responses.

Prompt lengths must be recomputed with the target model tokenizer; the token
counts in the legacy trace were produced by Gemini 2.5 Flash and are retained
only for comparison.
