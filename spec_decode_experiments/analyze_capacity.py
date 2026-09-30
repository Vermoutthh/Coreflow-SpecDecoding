#!/usr/bin/env python3
"""Summarize completed two-agent serving runs without treating smoke as capacity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metric_delta(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get("metrics", {}).get(key) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def summarize_run(path: Path) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    samples = load_jsonl(path / "server_metrics.jsonl")
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if "metrics" in sample:
            by_agent[sample["agent"]].append(sample)
    agent_metrics: dict[str, dict[str, Any]] = {}
    for agent, rows in by_agent.items():
        scheduled_count = metric_delta(
            rows, "vllm:coreflow_scheduled_requests_per_step_count"
        )
        scheduled_sum = metric_delta(
            rows, "vllm:coreflow_scheduled_requests_per_step_sum"
        )
        accepted = metric_delta(rows, "vllm:spec_decode_num_accepted_tokens_total")
        drafted = metric_delta(rows, "vllm:spec_decode_num_draft_tokens_total")
        waiting = [row["metrics"].get("vllm:num_requests_waiting", 0) for row in rows]
        running = [row["metrics"].get("vllm:num_requests_running", 0) for row in rows]
        kv_usage = [row["metrics"].get("vllm:gpu_cache_usage_perc", 0) for row in rows]
        prefix_hits = [
            row["metrics"]["vllm:coreflow_gpu_prefix_cache_hit_rate"]
            for row in rows
            if "vllm:coreflow_gpu_prefix_cache_hit_rate" in row["metrics"]
        ]
        agent_metrics[agent] = {
            "max_waiting_requests_sampled": max(waiting, default=None),
            "max_running_requests_sampled": max(running, default=None),
            "max_gpu_kv_usage_sampled": max(kv_usage, default=None),
            "mean_target_gpu_prefix_hit_rate_sampled": (
                sum(prefix_hits) / len(prefix_hits) if prefix_hits else None
            ),
            "mean_actual_scheduled_batch": (
                scheduled_sum / scheduled_count
                if scheduled_count and scheduled_sum is not None else None
            ),
            "draft_acceptance_rate": (
                accepted / drafted if drafted and accepted is not None else None
            ),
            "metrics_samples": len(rows),
        }
    return {
        "run_dir": str(path),
        "plan_k": config["plan_k"],
        "task_k": config["task_k"],
        "arrival_rate": config["arrival_rate"],
        "workflow_limit": config["workflow_limit"],
        "workflows_completed": summary["workflows_completed"],
        "workflows_total": summary["workflows_total"],
        "workflow_completions_per_second": summary["workflow_completions_per_second"],
        "p95_workflow_e2e_seconds": summary["p95_workflow_e2e_seconds"],
        "p95_admission_wait_seconds": summary.get("p95_admission_wait_seconds"),
        "agents": agent_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="Run directories or parent result directories")
    args = parser.parse_args()
    run_dirs: set[Path] = set()
    for path in args.paths:
        if (path / "summary.json").exists():
            run_dirs.add(path)
        else:
            run_dirs.update(summary.parent for summary in path.rglob("summary.json"))
    if not run_dirs:
        parser.error("No completed runs with summary.json found")
    for path in sorted(run_dirs):
        try:
            print(json.dumps(summarize_run(path), ensure_ascii=False))
        except (KeyError, OSError, ValueError) as exc:
            print(json.dumps({"run_dir": str(path), "error": str(exc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
