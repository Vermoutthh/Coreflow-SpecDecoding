#!/usr/bin/env python3
"""Run two real-weight vLLM instances and replay concurrent OWL workflows.

One GPU hosts the Plan target/draft pair; another hosts the Task pair.  The
script owns the server process groups and stops them even if replay fails.
Use --smoke for a two-workflow run whose temporary artifacts are discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "owl_gaia_react" / "raw_logs"
DEFAULT_RESULTS = ROOT / "CoreFlow_" / "results" / "spec_decode" / "concurrent_2gpu"


def gpu_memory_used_mib(index: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--id", str(index), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        # A just-stopped smoke run leaves TIME_WAIT sockets; those are not a
        # live listener and should not prevent reusing the fixed local ports.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def start_server(
    args: argparse.Namespace, agent: str, gpu: int, port: int, k: int,
    run_dir: Path,
) -> tuple[subprocess.Popen[bytes], object]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "VLLM_USE_V1": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join((
            str(ROOT / "CoreFlow_" / "src"),
            str(ROOT / "CoreFlow_" / "src" / "vllm"),
            env.get("PYTHONPATH", ""),
        )),
    })
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", args.target_model,
        "--tokenizer", args.target_model,
        "--dtype", "float16",
        "--tensor-parallel-size", "1",
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager", "--disable-log-requests",
    ]
    if args.prefix_caching:
        command.append("--enable-prefix-caching")
    if k:
        command.extend(("--speculative-config", json.dumps({
            "method": "draft_model", "model": args.draft_model,
            "num_speculative_tokens": k, "draft_tensor_parallel_size": 1,
        })))
    handle = (run_dir / f"{agent.lower()}_server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=ROOT / "CoreFlow_", env=env,
        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
    )
    return process, handle


def wait_health(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM on port {port} exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"vLLM did not become healthy on port {port}")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@contextmanager
def result_directory(args: argparse.Namespace) -> Iterator[Path]:
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="coreflow_sd_2gpu_smoke_") as path:
            yield Path(path)
        return
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = args.data_dir.parent.name
    path = args.results_root / dataset / f"plan_k{args.plan_k}_task_k{args.task_k}" / run_id
    if path.exists():
        raise FileExistsError(f"Results directory already exists: {path}")
    path.mkdir(parents=True)
    yield path


def run(args: argparse.Namespace) -> int:
    if args.plan_gpu == args.task_gpu:
        raise ValueError("Plan and Task must use distinct GPUs")
    for gpu in (args.plan_gpu, args.task_gpu):
        used = gpu_memory_used_mib(gpu)
        if used > args.max_preexisting_gpu_memory_mib:
            raise RuntimeError(
                f"GPU {gpu} already uses {used} MiB; refusing to launch"
            )
    for port in (args.plan_port, args.task_port):
        if not port_is_free(port):
            raise RuntimeError(f"Port {port} is already occupied")

    processes: list[subprocess.Popen[bytes]] = []
    handles: list[object] = []
    with result_directory(args) as run_dir:
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Results directory: {run_dir}", flush=True)
        try:
            launches = (
                ("Plan", args.plan_gpu, args.plan_port, args.plan_k),
                ("Task", args.task_gpu, args.task_port, args.task_k),
            )
            for agent, gpu, port, k in launches:
                print(f"Starting {agent} on GPU {gpu}, K={k}, port={port}", flush=True)
                process, handle = start_server(args, agent, gpu, port, k, run_dir)
                processes.append(process)
                handles.append(handle)
            for (_, _, port, _), process in zip(launches, processes):
                wait_health(process, port, args.server_startup_timeout)

            replay_command = [
                sys.executable, str(Path(__file__).with_name("replay_workflows.py")),
                "--data-dir", str(args.data_dir),
                "--target-model", args.target_model,
                "--max-model-len", str(args.max_model_len),
                "--max-tokens", str(args.max_tokens),
                "--workflow-limit", str(2 if args.smoke else args.workflow_limit),
                "--workflow-selection", args.workflow_selection,
                "--seed", str(args.seed),
                "--plan-url", f"http://127.0.0.1:{args.plan_port}",
                "--task-url", f"http://127.0.0.1:{args.task_port}",
                "--arrival-rate", str(0.0 if args.smoke else args.arrival_rate),
                "--max-inflight-workflows", str(2 if args.smoke else args.max_inflight_workflows),
                "--output-dir", str(run_dir),
            ]
            with (run_dir / "replay.log").open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    replay_command, cwd=ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode:
                print((run_dir / "replay.log").read_text(encoding="utf-8")[-5000:])
                raise RuntimeError(f"Replay failed with exit code {result.returncode}")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            if summary["workflows_completed"] != summary["workflows_total"]:
                raise RuntimeError("Some workflows did not complete; inspect replay.log")
            if args.smoke:
                samples = [json.loads(line) for line in
                           (run_dir / "server_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                for agent in ("Plan", "Task"):
                    agent_samples = [item.get("metrics", {}) for item in samples
                                     if item.get("agent") == agent]
                    if not any(
                        item.get("vllm:coreflow_scheduled_requests_per_step_count", 0) > 0
                        for item in agent_samples
                    ):
                        raise RuntimeError(f"No actual scheduled-batch metric for {agent}")
                    if args.prefix_caching and not any(
                        "vllm:coreflow_gpu_prefix_cache_hit_rate" in item
                        for item in agent_samples
                    ):
                        raise RuntimeError(f"No prefix-cache metric for {agent}")
            return 0
        except Exception:
            for agent in ("plan", "task"):
                path = run_dir / f"{agent}_server.log"
                if path.exists():
                    print(f"Last lines of {agent} server log:", flush=True)
                    print("\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-35:]), flush=True)
            raise
        finally:
            for process in reversed(processes):
                stop_process(process)
            for handle in handles:
                handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft-model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--plan-gpu", type=int, default=0)
    parser.add_argument("--task-gpu", type=int, default=1)
    parser.add_argument("--plan-port", type=int, default=18101)
    parser.add_argument("--task-port", type=int, default=18102)
    parser.add_argument("--plan-k", type=int, default=0)
    parser.add_argument("--task-k", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-preexisting-gpu-memory-mib", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--workflow-limit", type=int, default=20)
    parser.add_argument("--workflow-selection", choices=("trace", "random", "shortest", "longest"), default="trace")
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--max-inflight-workflows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--server-startup-timeout", type=float, default=240.0)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.plan_k < 0 or args.task_k < 0 or args.max_num_seqs < 2:
        parser.error("K must be nonnegative and max-num-seqs must be >= 2")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.plan_port == args.task_port:
        parser.error("Plan and Task ports must differ")
    if args.workflow_limit <= 0 or args.max_inflight_workflows <= 0:
        parser.error("workflow-limit and max-inflight-workflows must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
