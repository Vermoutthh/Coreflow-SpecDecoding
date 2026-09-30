#!/usr/bin/env python3
"""Profile draft-model speculative decoding on saved OWL semantic traces.

The profiler intentionally replays every saved LLM context independently.  It
therefore measures the target/draft pair on the original agent state without
letting locally generated responses change later workflow steps.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from common import REPO_ROOT, ensure_dir


DEFAULT_DATA_DIR = REPO_ROOT.parent / "datasets/owl_gaia_react/raw_logs"
AGENT_NAMES = {
    "user_agent": "Plan",
    "assistant_agent": "Task",
}

HEADER_RE = re.compile(
    r" - INFO - Task: (?P<task>\d+), agent: (?P<agent>\w+), "
    r"repeat: (?P<repeat>\d+)\s*$"
)
CONTEXT_MARKER = " - INFO - Context: "
TOKENS_MARKER = " - INFO - Used tokens: "


@dataclass
class TraceRequest:
    query_id: str
    source_file: str
    logged_task: int
    agent_name: str
    repeat: int
    messages: list[dict[str, Any]]
    original_prompt_tokens: Optional[int]
    original_completion_tokens: Optional[int]
    prompt_text: str = ""
    prompt_token_ids: Optional[list[int]] = None

    @property
    def paper_agent(self) -> str:
        return AGENT_NAMES.get(self.agent_name, self.agent_name)

    @property
    def context_tokens(self) -> int:
        if self.prompt_token_ids is None:
            raise RuntimeError("The request has not been tokenized")
        return len(self.prompt_token_ids)


def _literal(line: str, marker: str, path: Path, lineno: int) -> Any:
    try:
        return ast.literal_eval(line.split(marker, 1)[1].rstrip("\n"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot parse {path}:{lineno}: {exc}") from exc


def iter_owl_requests(log_dir: Path) -> Iterable[TraceRequest]:
    """Yield complete LLM calls from OWL's four-line log records."""

    paths = sorted(log_dir.glob("test_*.log"))
    if not paths:
        raise FileNotFoundError(f"No test_*.log files found under {log_dir}")

    for path in paths:
        query_id = path.stem.removeprefix("test_")
        header: Optional[re.Match[str]] = None
        messages: Optional[list[dict[str, Any]]] = None

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, 1):
                match = HEADER_RE.search(line)
                if match:
                    header = match
                    messages = None
                    continue

                if CONTEXT_MARKER in line:
                    value = _literal(line, CONTEXT_MARKER, path, lineno)
                    if not isinstance(value, list):
                        raise ValueError(
                            f"Expected a message list at {path}:{lineno}"
                        )
                    messages = value
                    continue

                if TOKENS_MARKER not in line:
                    continue

                if header is None or messages is None:
                    raise ValueError(
                        f"Incomplete OWL record before {path}:{lineno}"
                    )
                usage = _literal(line, TOKENS_MARKER, path, lineno)
                yield TraceRequest(
                    query_id=query_id,
                    source_file=path.name,
                    logged_task=int(header.group("task")),
                    agent_name=header.group("agent"),
                    repeat=int(header.group("repeat")),
                    messages=messages,
                    original_prompt_tokens=_optional_int(
                        usage.get("prompt_tokens")
                    ),
                    original_completion_tokens=_optional_int(
                        usage.get("completion_tokens")
                    ),
                )
                header = None
                messages = None


def _optional_int(value: Any) -> Optional[int]:
    return int(value) if value is not None else None


def render_and_tokenize(
    requests: list[TraceRequest], tokenizer: Any, batch_size: int = 128
) -> None:
    """Render OpenAI-style histories with the target model's chat template."""

    for request in requests:
        request.prompt_text = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        encoded = tokenizer(
            [request.prompt_text for request in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        for request, token_ids in zip(batch, encoded):
            request.prompt_token_ids = list(token_ids)


def context_bin(length: int) -> str:
    limits = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    lower = 0
    for upper in limits:
        if length <= upper:
            return f"{lower}-{upper}"
        lower = upper
    return "131072+"


def select_requests(
    requests: list[TraceRequest],
    agents: set[str],
    min_context_tokens: int,
    max_context_tokens: Optional[int],
    selection: str,
    limit: Optional[int],
    samples_per_agent_bin: Optional[int],
    seed: int,
) -> list[TraceRequest]:
    selected = [
        request
        for request in requests
        if (not agents or request.agent_name in agents)
        and request.context_tokens >= min_context_tokens
        and (
            max_context_tokens is None
            or request.context_tokens <= max_context_tokens
        )
    ]
    if selection == "shortest":
        selected.sort(key=lambda request: request.context_tokens)
    elif selection == "longest":
        selected.sort(key=lambda request: request.context_tokens, reverse=True)

    if samples_per_agent_bin is not None:
        groups: dict[tuple[str, str], list[TraceRequest]] = {}
        for request in selected:
            key = (request.agent_name, context_bin(request.context_tokens))
            groups.setdefault(key, []).append(request)
        rng = random.Random(seed)
        sampled: list[TraceRequest] = []
        for key in sorted(groups):
            group = groups[key]
            count = min(samples_per_agent_bin, len(group))
            sampled.extend(rng.sample(group, count))
        # Keep request order identical across separate k processes.
        selected = sorted(
            sampled,
            key=lambda request: (
                request.agent_name,
                context_bin(request.context_tokens),
                request.query_id,
                request.logged_task,
                request.repeat,
            ),
        )
    if limit is not None:
        selected = selected[:limit]
    return selected


def select_complete_workflows(
    requests: list[TraceRequest],
    max_context_tokens: int,
    workflow_limit: Optional[int],
    workflow_selection: str,
    workflow_id: Optional[str],
    seed: int,
) -> tuple[list[TraceRequest], dict[str, Any]]:
    """Select whole query workflows and preserve every call within each one.

    A workflow is eligible only when every one of its saved LLM contexts fits
    the configured capacity. This prevents a nominally workflow-level run from
    silently dropping its longest agent calls.
    """

    grouped: dict[str, list[TraceRequest]] = {}
    for request in requests:
        grouped.setdefault(request.query_id, []).append(request)

    eligible: list[tuple[str, list[TraceRequest]]] = []
    excluded: list[tuple[str, list[TraceRequest]]] = []
    for query_id, workflow_requests in grouped.items():
        destination = (
            eligible
            if all(
                request.context_tokens <= max_context_tokens
                for request in workflow_requests
            )
            else excluded
        )
        destination.append((query_id, workflow_requests))

    if workflow_id is not None:
        if workflow_id not in grouped:
            raise ValueError(f"Unknown workflow ID: {workflow_id}")
        eligible_by_id = dict(eligible)
        if workflow_id not in eligible_by_id:
            longest = max(
                request.context_tokens for request in grouped[workflow_id]
            )
            raise ValueError(
                f"Workflow {workflow_id} is not complete-workflow eligible: "
                f"longest context {longest} exceeds {max_context_tokens}"
            )
        candidates = [(workflow_id, eligible_by_id[workflow_id])]
    else:
        candidates = list(eligible)

    if workflow_id is None:
        if workflow_selection == "random":
            rng = random.Random(seed)
            rng.shuffle(candidates)
        elif workflow_selection == "shortest":
            candidates.sort(
                key=lambda item: (
                    len(item[1]),
                    max(request.context_tokens for request in item[1]),
                    item[0],
                )
            )
        elif workflow_selection == "longest":
            candidates.sort(
                key=lambda item: (
                    len(item[1]),
                    max(request.context_tokens for request in item[1]),
                    item[0],
                ),
                reverse=True,
            )
        elif workflow_selection != "trace":
            raise ValueError(
                f"Unsupported workflow selection: {workflow_selection}"
            )

    if workflow_id is None and workflow_limit is not None:
        candidates = candidates[:workflow_limit]

    selected = [
        request
        for _, workflow_requests in candidates
        for request in workflow_requests
    ]
    details = {
        "selection_unit": "workflow",
        "workflow_selection": workflow_selection,
        "source_workflows": len(grouped),
        "eligible_complete_workflows": len(eligible),
        "excluded_over_capacity_workflows": len(excluded),
        "requests_in_excluded_workflows": sum(
            len(workflow_requests) for _, workflow_requests in excluded
        ),
        "selected_workflows": len(candidates),
        "selected_workflow_ids": [query_id for query_id, _ in candidates],
    }
    return selected, details


def write_workflow_manifest(
    path: Path, requests: list[TraceRequest]
) -> None:
    """Write the exact ordered workflow set used by a sweep driver."""

    grouped: dict[str, list[TraceRequest]] = {}
    for request in requests:
        grouped.setdefault(request.query_id, []).append(request)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\trequest_count\tmax_context_tokens\n")
        for query_id, workflow_requests in grouped.items():
            handle.write(
                f"{query_id}\t{len(workflow_requests)}\t"
                f"{max(r.context_tokens for r in workflow_requests)}\n"
            )


def summarize_requests(requests: list[TraceRequest]) -> dict[str, Any]:
    by_agent: dict[str, list[int]] = {}
    for request in requests:
        by_agent.setdefault(request.agent_name, []).append(
            request.context_tokens
        )

    summary: dict[str, Any] = {
        "num_requests": len(requests),
        "num_workflows": len({request.query_id for request in requests}),
        "agents": {},
    }
    for agent, values in sorted(by_agent.items()):
        ordered = sorted(values)
        summary["agents"][agent] = {
            "paper_agent": AGENT_NAMES.get(agent, agent),
            "count": len(ordered),
            "min": ordered[0],
            "p50": statistics.median(ordered),
            "p95": ordered[round((len(ordered) - 1) * 0.95)],
            "max": ordered[-1],
        }
    return summary


def _safe_delta(end: Optional[float], start: Optional[float]) -> Optional[float]:
    if end is None or start is None:
        return None
    return end - start


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def acceptance_metrics(
    counts: Optional[list[int]], num_speculative_tokens: int
) -> dict[str, Any]:
    """Convert V0 per-request acceptance-position counters into metrics."""

    if num_speculative_tokens <= 0:
        return {
            "draft_rounds": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "accepted_per_position": [],
            "accept_rate": None,
            "mean_accepted_length": None,
            "progress_per_round": None,
            "full_accept_rate": None,
        }

    if counts is None:
        raise RuntimeError(
            "vLLM did not return spec_token_acceptance_counts; "
            "ensure VLLM_USE_V1=0 and use the bundled CoreFlow vLLM"
        )
    if len(counts) != num_speculative_tokens + 1:
        raise RuntimeError(
            "Unexpected acceptance counter length: "
            f"got {len(counts)}, expected {num_speculative_tokens + 1}"
        )

    rounds = int(counts[0])
    accepted_per_position = [int(value) for value in counts[1:]]
    accepted = sum(accepted_per_position)
    drafted = rounds * num_speculative_tokens
    return {
        "draft_rounds": rounds,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "accepted_per_position": accepted_per_position,
        "accept_rate": accepted / drafted if drafted else None,
        "mean_accepted_length": accepted / rounds if rounds else None,
        "progress_per_round": (accepted + rounds) / rounds if rounds else None,
        "full_accept_rate": (
            accepted_per_position[-1] / rounds
            if rounds and accepted_per_position
            else None
        ),
    }


def build_llm(args: argparse.Namespace) -> Any:
    # CoreFlow's serving changes and draft-model implementation currently use
    # vLLM V0. Set this before importing vLLM so engine selection is stable.
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM

    speculative_config = None
    if args.num_speculative_tokens > 0:
        speculative_config = {
            "method": "draft_model",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        }

    return LLM(
        model=args.target_model,
        tokenizer=args.target_model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        speculative_config=speculative_config,
        disable_log_stats=True,
    )


def profile_one(
    llm: Any, request: TraceRequest, args: argparse.Namespace
) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    prompt = {"prompt_token_ids": request.prompt_token_ids}
    wall_start = time.perf_counter()
    output = llm.generate([prompt], params, use_tqdm=False)[0]
    wall_seconds = time.perf_counter() - wall_start

    completion = output.outputs[0]
    output_tokens = len(completion.token_ids)
    metrics = output.metrics
    if metrics is None:
        raise RuntimeError("vLLM returned no per-request metrics")

    first_token_seconds = _safe_delta(
        metrics.first_token_time, metrics.arrival_time
    )
    e2e_seconds = _safe_delta(metrics.finished_time, metrics.arrival_time)
    decode_seconds = _safe_delta(
        metrics.finished_time, metrics.first_token_time
    )
    tpot_seconds = (
        decode_seconds / (output_tokens - 1)
        if decode_seconds is not None and output_tokens > 1
        else None
    )

    row = {
        "query_id": request.query_id,
        "source_file": request.source_file,
        "logged_task": request.logged_task,
        "agent_name": request.agent_name,
        "paper_agent": request.paper_agent,
        "repeat": request.repeat,
        "context_tokens": request.context_tokens,
        "context_bin": context_bin(request.context_tokens),
        "original_prompt_tokens": request.original_prompt_tokens,
        "original_completion_tokens": request.original_completion_tokens,
        "target_model": args.target_model,
        "draft_model": (
            args.draft_model if args.num_speculative_tokens > 0 else None
        ),
        "num_speculative_tokens": args.num_speculative_tokens,
        "generated_tokens": output_tokens,
        # Lets k=0 and k>0 runs verify lossless equivalence without retaining
        # generated text or a potentially large token-id array.
        "output_token_ids_sha256": token_ids_sha256(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "ttft_seconds": first_token_seconds,
        "decode_seconds": decode_seconds,
        "tpot_seconds": tpot_seconds,
        "e2e_seconds": e2e_seconds,
        "wall_seconds": wall_seconds,
        "model_forward_seconds": metrics.model_forward_time,
        "model_execute_seconds": metrics.model_execute_time,
    }
    row.update(
        acceptance_metrics(
            metrics.spec_token_acceptance_counts,
            args.num_speculative_tokens,
        )
    )
    return row


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--target-model", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--draft-model", default="unsloth/Llama-3.2-1B-Instruct"
    )
    parser.add_argument("--num-speculative-tokens", "-k", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-context-tokens", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument(
        "--selection-unit",
        choices=("workflow", "request"),
        default="workflow",
        help=(
            "Select complete query workflows (default), or enable legacy "
            "request-level filtering for micro-profiling"
        ),
    )
    parser.add_argument(
        "--workflow-selection",
        choices=("trace", "random", "shortest", "longest"),
        default="trace",
        help="How to choose eligible complete workflows",
    )
    parser.add_argument(
        "--workflow-limit",
        type=int,
        help="Maximum number of complete workflows to replay",
    )
    parser.add_argument(
        "--workflow-id",
        help="Replay exactly one eligible complete workflow by query ID",
    )
    parser.add_argument(
        "--workflow-manifest-output",
        type=Path,
        help="Write the ordered selected-workflow manifest as TSV",
    )
    parser.add_argument(
        "--agents", nargs="*", default=[], choices=sorted(AGENT_NAMES)
    )
    parser.add_argument(
        "--selection",
        choices=("trace", "shortest", "longest"),
        default="trace",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--samples-per-agent-bin",
        type=int,
        help="Deterministically sample up to N requests per agent/context bin",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Run and discard this many selected requests before measurement",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    # Leave room for the draft model: vLLM V0's target-worker memory profile
    # does not fully account for the proposer's weights when sizing KV cache.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_speculative_tokens < 0:
        raise ValueError("--num-speculative-tokens must be >= 0")
    if args.num_speculative_tokens > 0 and not args.draft_model:
        raise ValueError("--draft-model is required when k > 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.workflow_limit is not None and args.workflow_limit <= 0:
        raise ValueError("--workflow-limit must be positive")
    if args.workflow_id is not None and args.workflow_limit is not None:
        raise ValueError("--workflow-id and --workflow-limit are exclusive")
    if args.workflow_id is not None and args.workflow_selection != "trace":
        raise ValueError(
            "--workflow-id cannot be combined with --workflow-selection"
        )
    if (
        args.samples_per_agent_bin is not None
        and args.samples_per_agent_bin <= 0
    ):
        raise ValueError("--samples-per-agent-bin must be positive")
    if args.limit is not None and args.samples_per_agent_bin is not None:
        raise ValueError(
            "--limit and --samples-per-agent-bin cannot be used together"
        )
    if args.selection_unit == "workflow":
        request_filters = []
        if args.agents:
            request_filters.append("--agents")
        if args.min_context_tokens != 0:
            request_filters.append("--min-context-tokens")
        if args.limit is not None:
            request_filters.append("--limit")
        if args.samples_per_agent_bin is not None:
            request_filters.append("--samples-per-agent-bin")
        if args.selection != "trace":
            request_filters.append("--selection")
        if request_filters:
            raise ValueError(
                "Request-level filters cannot be used with "
                "--selection-unit workflow: " + ", ".join(request_filters)
            )
    elif args.workflow_limit is not None or args.workflow_selection != "trace":
        raise ValueError(
            "--workflow-limit/--workflow-selection require "
            "--selection-unit workflow"
        )
    if (
        args.selection_unit != "workflow"
        and (
            args.workflow_id is not None
            or args.workflow_manifest_output is not None
        )
    ):
        raise ValueError(
            "--workflow-id/--workflow-manifest-output require "
            "--selection-unit workflow"
        )
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.max_tokens >= args.max_model_len:
        raise ValueError("--max-tokens must be smaller than --max-model-len")
    if args.max_context_tokens is not None and (
        args.max_context_tokens > args.max_model_len - args.max_tokens
    ):
        raise ValueError(
            "--max-context-tokens must leave room for --max-tokens within "
            "--max-model-len"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    # Avoid warnings while measuring the trace distribution. Actual requests
    # are filtered against max_model_len before reaching the engine.
    tokenizer.model_max_length = max(tokenizer.model_max_length, 10**9)

    requests = list(iter_owl_requests(args.data_dir))
    render_and_tokenize(requests, tokenizer)
    capacity_limit = args.max_model_len - args.max_tokens
    effective_max_context = (
        args.max_context_tokens
        if args.max_context_tokens is not None
        else capacity_limit
    )
    selection_details: dict[str, Any]
    if args.selection_unit == "workflow":
        selected, selection_details = select_complete_workflows(
            requests=requests,
            max_context_tokens=effective_max_context,
            workflow_limit=args.workflow_limit,
            workflow_selection=args.workflow_selection,
            workflow_id=args.workflow_id,
            seed=args.seed,
        )
    else:
        selected = select_requests(
            requests=requests,
            agents=set(args.agents),
            min_context_tokens=args.min_context_tokens,
            max_context_tokens=effective_max_context,
            selection=args.selection,
            limit=args.limit,
            samples_per_agent_bin=args.samples_per_agent_bin,
            seed=args.seed,
        )
        selection_details = {
            "selection_unit": "request",
            "source_workflows": len(
                {request.query_id for request in requests}
            ),
        }
    summary = summarize_requests(selected)
    summary.update(selection_details)
    summary["source_requests"] = len(requests)
    summary["effective_max_context_tokens"] = effective_max_context
    summary["skipped_over_capacity"] = sum(
        request.context_tokens > effective_max_context for request in requests
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.workflow_manifest_output is not None:
        write_workflow_manifest(args.workflow_manifest_output, selected)
    if args.dry_run:
        return
    if not selected:
        raise ValueError("No requests matched the selection filters")
    if args.output is None:
        raise ValueError("--output is required unless --dry-run is used")

    llm = None
    try:
        llm = build_llm(args)
        for index in range(args.warmup_requests):
            warmup_request = selected[index % len(selected)]
            profile_one(llm, warmup_request, args)
        if args.warmup_requests:
            print(
                f"Completed {args.warmup_requests} warmup request(s)",
                flush=True,
            )
        ensure_dir(args.output.parent)
        with args.output.open("w", encoding="utf-8") as handle:
            for index, request in enumerate(selected, 1):
                row = profile_one(llm, request, args)
                write_jsonl_row(handle, row)
                rate = row["accept_rate"]
                rate_text = "n/a" if rate is None else f"{rate:.3f}"
                e2e = row["e2e_seconds"]
                e2e_text = "n/a" if e2e is None else f"{e2e:.3f}s"
                print(
                    f"[{index}/{len(selected)}] "
                    f"workflow={request.query_id} {request.paper_agent} "
                    f"context={request.context_tokens} "
                    f"output={row['generated_tokens']} "
                    f"k={args.num_speculative_tokens} accept={rate_text} "
                    f"e2e={e2e_text}",
                    flush=True,
                )
    finally:
        if llm is not None:
            del llm
        # Avoid leaking NCCL process groups or GPU allocations when running a
        # k-sweep as multiple sequential profiler processes.
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
