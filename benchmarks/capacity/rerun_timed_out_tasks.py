#!/usr/bin/env python3
"""
Rerun Timed-Out Tasks Utility

Loads a timed_out_tasks.json manifest produced by benchmark_tau_capacity.py or run_tau_comparison.py,
extracts the timed-out task IDs, and re-runs them with isolated concurrency (or custom concurrency)
and configurable extended timeouts for detailed trajectory debugging.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from benchmark_tau_capacity import run_tau_concurrency_tier, save_timed_out_manifest


def load_manifest_task_ids(manifest_path: str) -> tuple[str, List[str]]:
    """Loads timed out task IDs and domain from a timed_out_tasks.json file or directory."""
    if os.path.isdir(manifest_path):
        manifest_path = os.path.join(manifest_path, "timed_out_tasks.json")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        data = json.load(f)

    domain = data.get("domain", "retail")
    tasks = data.get("timed_out_tasks", [])

    task_ids = []
    for item in tasks:
        tid = str(item.get("task_id", ""))
        if tid and tid not in task_ids:
            task_ids.append(tid)

    return domain, task_ids


async def main_async():
    parser = argparse.ArgumentParser(description="Rerun Timed-Out tau-bench Tasks")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to timed_out_tasks.json or directory containing it",
    )
    parser.add_argument("--domain", default=None, help="Override domain from manifest (e.g. airline, retail)")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency for rerun (default: 1 for isolation)")
    parser.add_argument("--task-timeout", type=float, default=300.0, help="Extended timeout in seconds for rerun (default: 300)")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum turns per task")
    parser.add_argument("--agent-llm", default="openai/gemma-3-4b", help="Agent LLM model name")
    parser.add_argument("--user-llm", default="openai/gemma-3-4b", help="User Simulator LLM model name")
    parser.add_argument("--url-agent", default=None, help="Override agent vLLM endpoint URL")
    parser.add_argument("--url-user", default=None, help="Override user vLLM endpoint URL")
    parser.add_argument("--metrics-url", default=None, help="Override vLLM Prometheus metrics endpoint URL")
    parser.add_argument("--output-dir", default=None, help="Directory to save rerun transcripts and results")

    args = parser.parse_args()

    manifest_domain, task_ids = load_manifest_task_ids(args.manifest)
    domain = args.domain or manifest_domain

    if not task_ids:
        print(f"No timed out tasks found in manifest: {args.manifest}")
        return

    print(f"\n==========================================================================")
    print(f" RERUNNING TIMED OUT TASKS ({len(task_ids)} tasks)")
    print(f" Domain: {domain} | Task IDs: {', '.join(task_ids)}")
    print(f" Concurrency: {args.concurrency} | Timeout: {args.task_timeout}s | Max Steps: {args.max_steps}")
    print(f"==========================================================================\n")

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest)) if os.path.isfile(args.manifest) else args.manifest
    output_dir = args.output_dir or os.path.join(manifest_dir, "rerun_results")
    os.makedirs(output_dir, exist_ok=True)

    agent_args = {"max_tokens": 1024}
    user_args = {"max_tokens": 1024}

    if args.url_agent:
        agent_args["api_base"] = args.url_agent
        os.environ["OPENAI_API_BASE"] = args.url_agent
    if args.url_user:
        user_args["api_base"] = args.url_user

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    res = run_tau_concurrency_tier(
        domain=domain,
        concurrency=args.concurrency,
        agent_llm=args.agent_llm,
        agent_args=agent_args,
        user_llm=args.user_llm,
        user_args=user_args,
        num_tasks=len(task_ids),
        num_trials=1,
        max_steps=args.max_steps,
        task_timeout=args.task_timeout,
        metrics_url=args.metrics_url,
        output_dir=output_dir,
        task_ids_override=task_ids,
    )

    out_file = os.path.join(output_dir, f"rerun_c{args.concurrency}_results.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    # If some tasks still timed out, save an updated manifest
    if res.get("timed_out_samples"):
        save_timed_out_manifest(res["timed_out_samples"], output_dir, args.task_timeout)

    print(f"\n==========================================================================")
    print(f" RERUN SUMMARY")
    print(f" Total tasks rerun: {res['total_simulations']}")
    print(f" Succeeded: {res['successful_tasks']} ({res['success_rate_perc']}%)")
    print(f" Still Timed Out: {res['timed_out_count']}")
    print(f" Elapsed: {res['total_elapsed_sec']}s")
    print(f" Results & Transcripts saved to: {output_dir}")
    print(f"==========================================================================\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
