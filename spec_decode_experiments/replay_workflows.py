#!/usr/bin/env python3
"""Concurrent, causally ordered replay of saved OWL workflow prompts.

Each saved prompt is sent unchanged.  The newly generated response is *not*
inserted into subsequent saved prompts, so this measures serving behavior on
the original semantic states rather than a newly evolving agent conversation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CoreFlow_" / "scripts"))
from profile_spec_decode import (  # noqa: E402
    TraceRequest,
    context_bin,
    iter_owl_requests,
    render_and_tokenize,
    select_complete_workflows,
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def prepare_workflows(args: argparse.Namespace) -> tuple[list[list[TraceRequest]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    requests = list(iter_owl_requests(args.data_dir))
    render_and_tokenize(requests, tokenizer)
    selected, details = select_complete_workflows(
        requests=requests,
        max_context_tokens=args.max_model_len - args.max_tokens,
        workflow_limit=args.workflow_limit,
        workflow_selection=args.workflow_selection,
        workflow_id=None,
        seed=args.seed,
    )
    grouped: dict[str, list[TraceRequest]] = defaultdict(list)
    for request in selected:
        grouped[request.query_id].append(request)
    return list(grouped.values()), details


async def send_request(
    client: httpx.AsyncClient,
    request: TraceRequest,
    endpoint: str,
    model: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": request.prompt_token_ids,
        "add_special_tokens": False,
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token: float | None = None
    output_parts: list[str] = []
    output_tokens: int | None = None
    finish_reason: str | None = None
    async with client.stream(
        "POST", endpoint.rstrip("/") + "/v1/completions", json=payload
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                output_tokens = event["usage"].get("completion_tokens")
            for choice in event.get("choices", []):
                chunk = choice.get("text", "")
                if chunk:
                    if first_token is None:
                        first_token = time.perf_counter()
                    output_parts.append(chunk)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    if finish_reason is None or output_tokens is None:
        raise RuntimeError("Completion stream ended without finish reason or usage")
    finished = time.perf_counter()
    return {
        "request_id": (
            f"{request.query_id}:{request.agent_name}:"
            f"{request.logged_task}:{request.repeat}"
        ),
        "workflow_id": request.query_id,
        "agent": request.paper_agent,
        "context_tokens": request.context_tokens,
        "context_bin": context_bin(request.context_tokens),
        "output_tokens": output_tokens,
        "output_chars": sum(map(len, output_parts)),
        "output_sha256": hashlib.sha256(
            "".join(output_parts).encode("utf-8")
        ).hexdigest(),
        "finish_reason": finish_reason,
        "start_monotonic": started,
        "end_monotonic": finished,
        "e2e_seconds": finished - started,
        "ttft_seconds": first_token - started if first_token is not None else None,
        "success": True,
    }


METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:coreflow_gpu_prefix_cache_hit_rate",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_emitted_tokens_total",
    "vllm:coreflow_scheduled_requests_per_step_count",
    "vllm:coreflow_scheduled_requests_per_step_sum",
)


def parse_metrics(body: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in METRICS:
            continue
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


async def poll_metrics(
    client: httpx.AsyncClient,
    endpoints: dict[str, str],
    path: Path,
    stop: asyncio.Event,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            for agent, endpoint in endpoints.items():
                try:
                    response = await client.get(endpoint.rstrip("/") + "/metrics")
                    response.raise_for_status()
                    row = {
                        "time": time.time(),
                        "agent": agent,
                        "metrics": parse_metrics(response.text),
                    }
                except (httpx.HTTPError, ValueError) as exc:
                    row = {"time": time.time(), "agent": agent, "error": str(exc)}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def run(args: argparse.Namespace) -> dict[str, Any]:
    workflows, selection = prepare_workflows(args)
    if not workflows:
        raise ValueError("No complete workflows fit the selected context limit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {"workflow_id": group[0].query_id, "calls": len(group),
         "max_context_tokens": max(r.context_tokens for r in group)}
        for group in workflows
    ]
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"selection": selection, "workflows": manifest}, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        return {"dry_run": True, "workflows": len(workflows),
                "requests": sum(len(group) for group in workflows),
                "selection": selection}

    endpoints = {"Plan": args.plan_url, "Task": args.task_url}
    timeout = httpx.Timeout(args.request_timeout, connect=30.0)
    limits = httpx.Limits(max_connections=max(16, 4 * args.max_inflight_workflows))
    semaphore = asyncio.Semaphore(args.max_inflight_workflows)
    write_lock = asyncio.Lock()
    stop_metrics = asyncio.Event()
    request_rows: list[dict[str, Any]] = []
    workflow_rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    release_offsets: list[float] = []
    offset = 0.0
    for index in range(len(workflows)):
        if index and args.arrival_rate > 0:
            offset += rng.expovariate(args.arrival_rate)
        release_offsets.append(offset)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        with (args.output_dir / "requests.jsonl").open("w", encoding="utf-8") as request_file, \
                (args.output_dir / "workflows.jsonl").open("w", encoding="utf-8") as workflow_file:
            poller = asyncio.create_task(poll_metrics(
                client, endpoints, args.output_dir / "server_metrics.jsonl", stop_metrics
            ))
            base = time.perf_counter()

            async def run_workflow(index: int, group: list[TraceRequest]) -> None:
                await asyncio.sleep(max(0.0, base + release_offsets[index] - time.perf_counter()))
                arrived = time.perf_counter()
                arrived_wall_time = time.time()
                async with semaphore:
                    admitted = time.perf_counter()
                    error: str | None = None
                    completed_calls = 0
                    for request in group:
                        try:
                            row = await send_request(
                                client, request, endpoints[request.paper_agent],
                                args.target_model, args.max_tokens, args.seed,
                            )
                        except (httpx.HTTPError, ValueError, KeyError, RuntimeError) as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            row = {
                                "request_id": (
                                    f"{request.query_id}:{request.agent_name}:"
                                    f"{request.logged_task}:{request.repeat}"
                                ),
                                "workflow_id": request.query_id,
                                "agent": request.paper_agent,
                                "context_tokens": request.context_tokens,
                                "context_bin": context_bin(request.context_tokens),
                                "success": False,
                                "error": error,
                            }
                        request_rows.append(row)
                        if row["success"]:
                            completed_calls += 1
                        async with write_lock:
                            request_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                            request_file.flush()
                        if error is not None:
                            break
                    finished = time.perf_counter()
                    summary = {
                        "workflow_id": group[0].query_id,
                        "calls_expected": len(group),
                        "calls_completed": completed_calls,
                        "arrival_monotonic": arrived,
                        "arrival_wall_time": arrived_wall_time,
                        "admitted_monotonic": admitted,
                        "finished_monotonic": finished,
                        "finished_wall_time": time.time(),
                        "admission_wait_seconds": admitted - arrived,
                        "e2e_seconds": finished - arrived,
                        "success": error is None,
                        "error": error,
                    }
                    workflow_rows.append(summary)
                    async with write_lock:
                        workflow_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
                        workflow_file.flush()

            try:
                await asyncio.gather(*(
                    run_workflow(index, group) for index, group in enumerate(workflows)
                ))
            finally:
                stop_metrics.set()
                await poller

    success_workflows = [row for row in workflow_rows if row["success"]]
    success_requests = [row for row in request_rows if row["success"]]
    elapsed = max(row["finished_monotonic"] for row in workflow_rows) - base
    summary = {
        "workflows_total": len(workflows),
        "workflows_completed": len(success_workflows),
        "requests_completed": len(success_requests),
        "requests_failed": len(request_rows) - len(success_requests),
        "wall_seconds": elapsed,
        "workflow_completions_per_second": len(success_workflows) / elapsed,
        "arrival_rate_workflows_per_second": args.arrival_rate,
        "p50_workflow_e2e_seconds": _percentile(
            [row["e2e_seconds"] for row in success_workflows], 0.5
        ),
        "p95_workflow_e2e_seconds": _percentile(
            [row["e2e_seconds"] for row in success_workflows], 0.95
        ),
        "p95_admission_wait_seconds": _percentile(
            [row["admission_wait_seconds"] for row in workflow_rows], 0.95
        ),
        "agents": {},
    }
    for agent in endpoints:
        rows = [row for row in success_requests if row["agent"] == agent]
        by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_context[row["context_bin"]].append(row)
        summary["agents"][agent] = {
            "requests": len(rows),
            "requests_per_second": len(rows) / elapsed,
            "generated_tokens": sum(row["output_tokens"] or 0 for row in rows),
            "p95_request_e2e_seconds": _percentile(
                [row["e2e_seconds"] for row in rows], 0.95
            ),
            "p95_ttft_seconds": _percentile(
                [row["ttft_seconds"] for row in rows
                 if row["ttft_seconds"] is not None], 0.95
            ),
            "context_bins": {
                bin_name: {
                    "requests": len(bin_rows),
                    "p95_request_e2e_seconds": _percentile(
                        [row["e2e_seconds"] for row in bin_rows], 0.95
                    ),
                }
                for bin_name, bin_rows in sorted(by_context.items())
            },
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--workflow-limit", type=int)
    parser.add_argument("--workflow-selection", choices=("trace", "random", "shortest", "longest"), default="trace")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plan-url", default="http://127.0.0.1:18101")
    parser.add_argument("--task-url", default="http://127.0.0.1:18102")
    parser.add_argument("--arrival-rate", type=float, default=0.0,
                        help="Poisson workflow arrivals/s; zero releases all at once")
    parser.add_argument("--max-inflight-workflows", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_tokens >= args.max_model_len:
        parser.error("max-tokens must be positive and smaller than max-model-len")
    if args.max_inflight_workflows <= 0 or args.arrival_rate < 0:
        parser.error("max-inflight-workflows must be positive and arrival-rate nonnegative")
    if args.workflow_limit is not None and args.workflow_limit <= 0:
        parser.error("workflow-limit must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), indent=2, ensure_ascii=False))
