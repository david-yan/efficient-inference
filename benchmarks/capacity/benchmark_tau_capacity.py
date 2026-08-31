#!/usr/bin/env python3
"""
Multi-Turn Agent Capacity & Efficiency Benchmark Harness (tau-bench / tau2-bench)

Evaluates vLLM serving limits, turn latency, TTFT, TPOT, GPU KV-cache saturation,
and task success rates across concurrency tiers for multi-turn agent workloads.
Includes per-task wallclock timeouts, full trajectory/transcript exports (.md and .json),
and timed-out task manifest logging.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional
import aiohttp
import numpy as np

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")

# Ensure tau2-bench src is in sys.path
TAU2_SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tau2-bench", "src")
)
if TAU2_SRC_DIR not in sys.path:
    sys.path.insert(0, TAU2_SRC_DIR)

# Ensure current directory is in sys.path for transcript_logger
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import run_tasks
    from tau2.runner.helpers import get_tasks
except ImportError as e:
    print(f"Error importing tau2: {e}")
    print(f"Ensure tau2-bench is installed or located at {TAU2_SRC_DIR}")
    sys.exit(1)

from transcript_logger import (
    export_simulation_artifacts,
    generate_tier_index_markdown,
    save_timed_out_manifest,
)


def scrape_vllm_metrics_instant(metrics_url: Optional[str]) -> Dict[str, float]:
    """Scrapes instantaneous Prometheus metrics from vLLM."""
    metrics = {
        "num_requests_running": 0.0,
        "num_requests_waiting": 0.0,
        "gpu_cache_usage_perc": 0.0,
        "prefix_cache_hit_rate": 0.0,
    }
    if not metrics_url:
        return metrics

    try:
        req = urllib.request.Request(metrics_url, headers={"User-Agent": "tau-bench-capacity"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                text = resp.read().decode("utf-8")
                for line in text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    if line.startswith("vllm:num_requests_running"):
                        metrics["num_requests_running"] = float(line.split()[-1])
                    elif line.startswith("vllm:num_requests_waiting"):
                        metrics["num_requests_waiting"] = float(line.split()[-1])
                    elif line.startswith("vllm:gpu_cache_usage_perc"):
                        metrics["gpu_cache_usage_perc"] = float(line.split()[-1]) * 100.0
                    elif line.startswith("vllm:cpu_prefix_cache_hit_rate") or line.startswith("vllm:gpu_prefix_cache_hit_rate"):
                        metrics["prefix_cache_hit_rate"] = max(
                            metrics["prefix_cache_hit_rate"],
                            float(line.split()[-1]) * 100.0,
                        )
    except Exception:
        pass
    return metrics


class ContinuousMetricsSampler(threading.Thread):
    """Background sampler that continuously measures active vLLM streams and KV cache during a test."""

    def __init__(self, metrics_url: Optional[str], sample_interval_sec: float = 0.5):
        super().__init__(daemon=True)
        self.metrics_url = metrics_url
        self.sample_interval = sample_interval_sec
        self.stop_event = threading.Event()
        self.running_samples: List[float] = []
        self.waiting_samples: List[float] = []
        self.kv_usage_samples: List[float] = []
        self.prefix_hits: List[float] = []

    def run(self):
        if not self.metrics_url:
            return
        while not self.stop_event.is_set():
            try:
                m = scrape_vllm_metrics_instant(self.metrics_url)
                self.running_samples.append(m["num_requests_running"])
                self.waiting_samples.append(m["num_requests_waiting"])
                self.kv_usage_samples.append(m["gpu_cache_usage_perc"])
                self.prefix_hits.append(m["prefix_cache_hit_rate"])
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def stop(self):
        self.stop_event.set()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "peak_active_streams": int(max(self.running_samples)) if self.running_samples else 0,
            "avg_active_streams": round(float(np.mean(self.running_samples)), 1) if self.running_samples else 0.0,
            "p50_active_streams": round(float(np.percentile(self.running_samples, 50)), 1) if self.running_samples else 0.0,
            "p90_active_streams": round(float(np.percentile(self.running_samples, 90)), 1) if self.running_samples else 0.0,
            "peak_waiting_queue": int(max(self.waiting_samples)) if self.waiting_samples else 0,
            "avg_waiting_queue": round(float(np.mean(self.waiting_samples)), 1) if self.waiting_samples else 0.0,
            "peak_kv_cache_perc": round(max(self.kv_usage_samples), 1) if self.kv_usage_samples else 0.0,
            "avg_kv_cache_perc": round(float(np.mean(self.kv_usage_samples)), 1) if self.kv_usage_samples else 0.0,
            "prefix_cache_hit_rate_perc": round(max(self.prefix_hits), 1) if self.prefix_hits else 0.0,
        }


def prepare_task_workload(domain: str, concurrency: int, requested_tasks: int, task_ids_override: Optional[List[str]] = None) -> List[Any]:
    """Prepares task workload, ensuring sufficient queue depth to sustain target concurrency."""
    domains = [d.strip() for d in domain.split(",") if d.strip()]
    all_domain_tasks = []
    for d in domains:
        all_domain_tasks.extend(get_tasks(d))

    if task_ids_override:
        filtered = [t for t in all_domain_tasks if str(t.id) in set(task_ids_override)]
        return filtered if filtered else all_domain_tasks

    target_count = max(requested_tasks, concurrency * 2)

    if len(all_domain_tasks) == 0:
        return []

    if target_count <= len(all_domain_tasks):
        return all_domain_tasks[:target_count]

    multiplier = (target_count + len(all_domain_tasks) - 1) // len(all_domain_tasks)
    return (all_domain_tasks * multiplier)[:target_count]


def run_tau_concurrency_tier(
    domain: str,
    concurrency: int,
    agent_llm: str,
    agent_args: dict,
    user_llm: str,
    user_args: dict,
    agent_name: str = "llm_agent",
    num_tasks: int = 50,
    num_trials: int = 1,
    max_steps: int = 30,
    task_timeout: float = 120.0,
    metrics_url: Optional[str] = None,
    output_dir: Optional[str] = None,
    task_ids_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute tau2-bench run for a specific concurrency tier with timeout enforcement and transcript logging."""
    tasks = prepare_task_workload(domain, concurrency, num_tasks, task_ids_override=task_ids_override)

    agent_args_clean = agent_args.copy() if agent_args else {}
    user_args_clean = user_args.copy() if user_args else {}
    agent_args_clean.setdefault("max_tokens", 1024)
    user_args_clean.setdefault("max_tokens", 1024)
    stop_tokens = [
        "<turn|>",
        "<|turn>",
        "<|turn>user",
        "<|turn>model>",
        "\n<|turn>",
        "<|im_end|>",
        "\nUser:",
        "\nAssistant:"
    ]
    if "gemma" in agent_llm.lower() or "qwen" in agent_llm.lower() or "vllm" in str(agent_args_clean.get("api_base", "")) or "8000" in str(agent_args_clean.get("api_base", "")) or "18000" in str(agent_args_clean.get("api_base", "")):
        agent_args_clean.setdefault("stop", stop_tokens)
    if "gemma" in user_llm.lower() or "qwen" in user_llm.lower() or "vllm" in str(user_args_clean.get("api_base", "")) or "8000" in str(user_args_clean.get("api_base", "")) or "18001" in str(user_args_clean.get("api_base", "")):
        user_args_clean.setdefault("stop", stop_tokens)


    primary_domain = domain.split(",")[0].strip() if "," in domain else domain
    config = TextRunConfig(
        domain=primary_domain,
        agent=agent_name,
        llm_agent=agent_llm,
        llm_args_agent=agent_args_clean,
        user="user_simulator",
        llm_user=user_llm,
        llm_args_user=user_args_clean,
        max_concurrency=concurrency,
        num_trials=num_trials,
        max_steps=max_steps,
        timeout=task_timeout,
        hallucination_retries=0,
    )

    sampler = ContinuousMetricsSampler(metrics_url, sample_interval_sec=0.5)
    sampler.start()

    start_time = time.perf_counter()
    results = run_tasks(config, tasks, console_display=False)
    elapsed = time.perf_counter() - start_time

    sampler.stop()
    sampler.join(timeout=2.0)
    server_metrics = sampler.get_summary()

    total_agent_tokens = 0
    total_user_tokens = 0
    agent_latencies = []
    user_latencies = []
    success_count = 0
    timed_out_count = 0
    total_turns = 0
    sim_details = []
    tier_timed_out_samples = []

    transcripts_base_dir = os.path.join(output_dir, "transcripts") if output_dir else "benchmarks/capacity/results/transcripts"

    for sim in results.simulations:
        is_success = False
        reward_val = 0.0
        term_reason = getattr(sim, "termination_reason", "unknown")
        if hasattr(term_reason, "value"):
            term_reason = term_reason.value

        duration_val = getattr(sim, "duration", getattr(sim, "duration_seconds", 0.0))
        is_timeout = (term_reason == "timeout" or duration_val >= task_timeout)

        if is_timeout:
            timed_out_count += 1

        if sim.reward_info:
            reward_val = float(getattr(sim.reward_info, "reward", 0.0))
            if reward_val == 1.0 and not is_timeout:
                success_count += 1
                is_success = True

        sim_agent_toks = 0
        sim_user_toks = 0
        sim_turns = 0

        for msg in sim.messages:
            gen_time = getattr(msg, "generation_time_seconds", None)
            usage = getattr(msg, "usage", None) or {}

            if msg.role == "assistant":
                sim_turns += 1
                if gen_time:
                    agent_latencies.append(gen_time)
                toks = usage.get("completion_tokens", 0)
                total_agent_tokens += toks
                sim_agent_toks += toks
            elif msg.role == "user":
                if gen_time:
                    user_latencies.append(gen_time)
                toks = usage.get("completion_tokens", 0)
                total_user_tokens += toks
                sim_user_toks += toks

        total_turns += sim_turns

        # Export individual task transcript and trajectory
        task_obj = getattr(sim, "task", None)
        artifacts = export_simulation_artifacts(sim, task_obj, concurrency, transcripts_base_dir)

        sim_detail_entry = {
            "task_id": str(getattr(sim.task, "id", "unknown") if hasattr(sim, "task") else "unknown"),
            "success": is_success,
            "reward": reward_val,
            "turns": sim_turns,
            "agent_tokens": sim_agent_toks,
            "user_tokens": sim_user_toks,
            "duration_sec": round(duration_val, 2),
            "termination_reason": term_reason,
            "transcript_md": artifacts["transcript_md"],
            "trajectory_json": artifacts["trajectory_json"],
        }
        sim_details.append(sim_detail_entry)

        if is_timeout:
            tier_timed_out_samples.append({
                "concurrency_tier": concurrency,
                "task_id": sim_detail_entry["task_id"],
                "domain": domain,
                "duration_sec": sim_detail_entry["duration_sec"],
                "turns_completed": sim_turns,
                "termination_reason": term_reason,
                "transcript_md": os.path.relpath(artifacts["transcript_md"], transcripts_base_dir),
                "transcript_json": os.path.relpath(artifacts["trajectory_json"], transcripts_base_dir),
            })

    # Write Concurrency Tier INDEX.md Catalog
    tier_index_md = generate_tier_index_markdown(sim_details, concurrency, elapsed)
    tier_index_path = os.path.join(transcripts_base_dir, f"tier_C{concurrency}", "INDEX.md")
    try:
        with open(tier_index_path, "w", encoding="utf-8") as f:
            f.write(tier_index_md)
    except Exception:
        pass

    total_simulations = len(results.simulations)
    pass_rate = success_count / total_simulations if total_simulations > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_simulations": total_simulations,
        "success_count": success_count,
        "timed_out_count": timed_out_count,
        "timed_out_rate_perc": round((timed_out_count / total_simulations) * 100.0, 1) if total_simulations > 0 else 0.0,
        "success_rate_perc": round(pass_rate * 100.0, 1),
        "total_elapsed_sec": round(elapsed, 2),
        "total_agent_tokens": total_agent_tokens,
        "total_user_tokens": total_user_tokens,
        "total_tokens": total_agent_tokens + total_user_tokens,
        "agent_throughput_tok_s": round(total_agent_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "user_throughput_tok_s": round(total_user_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "total_throughput_tok_s": round((total_agent_tokens + total_user_tokens) / elapsed, 1) if elapsed > 0 else 0.0,
        "agent_turn_latency_p50_ms": round(float(np.percentile(agent_latencies, 50) * 1000.0), 1) if agent_latencies else 0.0,
        "agent_turn_latency_p90_ms": round(float(np.percentile(agent_latencies, 90) * 1000.0), 1) if agent_latencies else 0.0,
        "agent_turn_latency_p99_ms": round(float(np.percentile(agent_latencies, 99) * 1000.0), 1) if agent_latencies else 0.0,
        "agent_turn_latency_mean_ms": round(float(np.mean(agent_latencies) * 1000.0), 1) if agent_latencies else 0.0,
        "user_turn_latency_p50_ms": round(float(np.percentile(user_latencies, 50) * 1000.0), 1) if user_latencies else 0.0,
        "user_turn_latency_p90_ms": round(float(np.percentile(user_latencies, 90) * 1000.0), 1) if user_latencies else 0.0,
        "total_turns": total_turns,
        "avg_turns_per_sim": round(total_turns / total_simulations, 1) if total_simulations > 0 else 0.0,
        "peak_active_streams": server_metrics["peak_active_streams"],
        "avg_active_streams": server_metrics["avg_active_streams"],
        "p50_active_streams": server_metrics["p50_active_streams"],
        "peak_waiting_queue": server_metrics["peak_waiting_queue"],
        "peak_kv_cache_perc": server_metrics["peak_kv_cache_perc"],
        "avg_kv_cache_perc": server_metrics["avg_kv_cache_perc"],
        "prefix_cache_hit_rate_perc": server_metrics["prefix_cache_hit_rate_perc"],
        "timed_out_samples": tier_timed_out_samples,
        "simulations_summary": sim_details[:50],
    }


async def run_tau_capacity_suite(
    domain: str,
    agent_llm: str,
    agent_args: dict,
    user_llm: str,
    user_args: dict,
    tiers: List[int],
    num_tasks: int,
    num_trials: int,
    agent_name: str = "llm_agent",
    max_steps: int = 30,
    task_timeout: float = 120.0,
    metrics_url: Optional[str] = None,
    output_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    task_ids_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs a full concurrency sweep for tau-bench capacity testing with transcript exporting and timed out manifests."""
    tier_results = []
    all_suite_timed_out_samples = []

    target_dir = output_dir or (os.path.dirname(os.path.abspath(output_json)) if output_json else "benchmarks/capacity/results")
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n==========================================================================")
    print(f" STARTING MULTI-TURN CAPACITY BENCHMARK ({domain.upper()})")
    print(f" Agent: {agent_llm} ({agent_name}) | User: {user_llm}")
    print(f" Tiers: {tiers} | Target Tasks: {num_tasks} | Max Steps: {max_steps} | Task Timeout: {task_timeout}s")
    print(f"==========================================================================\n")

    for concurrency in tiers:
        print(f"Running Concurrency Tier C={concurrency} ...")
        res = run_tau_concurrency_tier(
            domain=domain,
            concurrency=concurrency,
            agent_llm=agent_llm,
            agent_args=agent_args,
            user_llm=user_llm,
            user_args=user_args,
            agent_name=agent_name,
            num_tasks=num_tasks,
            num_trials=num_trials,
            max_steps=max_steps,
            task_timeout=task_timeout,
            metrics_url=metrics_url,
            output_dir=target_dir,
            task_ids_override=task_ids_override,
        )
        tier_results.append(res)
        all_suite_timed_out_samples.extend(res.get("timed_out_samples", []))

        print(
            f"  -> C={concurrency:3d} | Active Streams (Avg/Peak): {res['avg_active_streams']:4.1f}/{res['peak_active_streams']:2d} | "
            f"Tok/s: {res['total_throughput_tok_s']:7.1f} | "
            f"Agent P50: {res['agent_turn_latency_p50_ms']:6.1f}ms | "
            f"KV: {res['peak_kv_cache_perc']:4.1f}% | "
            f"Pass: {res['success_rate_perc']:5.1f}% | "
            f"Timed Out: {res['timed_out_count']:2d} | "
            f"Elapsed: {res['total_elapsed_sec']}s"
        )

    # Save Timed-Out Manifest
    if all_suite_timed_out_samples:
        manifest_res = save_timed_out_manifest(all_suite_timed_out_samples, target_dir, task_timeout)
        print(f"\nTimed-out manifest logged: {manifest_res['json_path']} ({len(all_suite_timed_out_samples)} tasks)")

    output_data = {
        "benchmark": "tau2-bench-capacity",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "domain": domain,
        "agent_llm": agent_llm,
        "agent_args": agent_args,
        "user_llm": user_llm,
        "user_args": user_args,
        "concurrency_tiers": tiers,
        "num_tasks_requested": num_tasks,
        "num_trials": num_trials,
        "max_steps": max_steps,
        "task_timeout_sec": task_timeout,
        "total_timed_out_tasks": len(all_suite_timed_out_samples),
        "results": tier_results,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults successfully saved to: {output_json}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Capacity Benchmark Harness (tau-bench)")
    parser.add_argument("--domain", default="airline", help="Domain name (e.g. airline, retail, mock)")
    parser.add_argument("--agent-llm", default="openai/gemma-3-4b", help="Agent LLM model name")
    parser.add_argument("--url-agent", default=None, help="Convenience flag for Agent API URL")
    parser.add_argument("--url-user", default=None, help="Convenience flag for User API URL")
    parser.add_argument("--agent-args", type=json.loads, default='{"api_base": "http://localhost:8000/v1", "api_key": "EMPTY", "max_tokens": 1024}')
    parser.add_argument("--user-llm", default="openai/gemma-3-4b", help="User Simulator LLM model name")
    parser.add_argument("--user-args", type=json.loads, default='{"api_base": "http://localhost:8000/v1", "api_key": "EMPTY", "max_tokens": 1024}')
    parser.add_argument("--tiers", nargs="+", type=int, default=[8, 16, 32, 64, 96, 128, 160])
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks to evaluate per tier")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of trials per task")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum number of turns per task")
    parser.add_argument("--task-timeout", type=float, default=120.0, help="Maximum execution time in seconds per task before timeout")
    parser.add_argument("--metrics-url", default=None, help="vLLM Prometheus metrics endpoint URL (e.g. http://localhost:8000/metrics)")
    parser.add_argument("--output-json", default="benchmarks/capacity/results/tau_capacity_results.json")
    parser.add_argument("--output-dir", default=None, help="Base output directory for transcripts and manifests")
    parser.add_argument("--rerun-manifest", default=None, help="Path to timed_out_tasks.json to re-execute only failed/timed-out tasks")

    args = parser.parse_args()

    agent_args = args.agent_args
    user_args = args.user_args
    if args.url_agent:
        agent_args["api_base"] = args.url_agent
    if args.url_user:
        user_args["api_base"] = args.url_user

    task_ids_override = None
    if args.rerun_manifest and os.path.exists(args.rerun_manifest):
        with open(args.rerun_manifest, "r") as f:
            manifest_json = json.load(f)
            task_ids_override = [str(item["task_id"]) for item in manifest_json.get("timed_out_samples", [])]
            print(f"Rerunning {len(task_ids_override)} tasks from manifest: {task_ids_override}")

    asyncio.run(
        run_tau_capacity_suite(
            domain=args.domain,
            agent_llm=args.agent_llm,
            agent_args=agent_args,
            user_llm=args.user_llm,
            user_args=user_args,
            tiers=args.tiers,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            max_steps=args.max_steps,
            task_timeout=args.task_timeout,
            metrics_url=args.metrics_url,
            output_json=args.output_json,
            output_dir=args.output_dir,
            task_ids_override=task_ids_override,
        )
    )


if __name__ == "__main__":
    main()
