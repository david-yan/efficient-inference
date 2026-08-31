#!/usr/bin/env python3
"""
Experiment 4: Supervised Fine-Tuning (SFT) Evaluation Runner for tau-bench (Dual-Node Config).

Evaluates the SFT fine-tuned LoRA model ("gemma_sft" or "gemma-3-4b-sft") at concurrency C*=96
across 2 dedicated vLLM nodes to measure latency, throughput, and JSON format compliance.
"""

import argparse
import asyncio
import os
import sys

TAU2_SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tau2-bench", "src")
)
if TAU2_SRC_DIR not in sys.path:
    sys.path.insert(0, TAU2_SRC_DIR)

CAPACITY_DIR = os.path.abspath(os.path.dirname(__file__))
if CAPACITY_DIR not in sys.path:
    sys.path.insert(0, CAPACITY_DIR)

from benchmark_tau_capacity import run_tau_capacity_suite


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: SFT LoRA Model Evaluation Runner (Dual-Node)")
    parser.add_argument("--domain", default="airline,retail", help="Comma-separated domains")
    parser.add_argument("--agent-llm", default="openai/gemma_sft", help="Fine-tuned LoRA model name")
    parser.add_argument("--url-agent", default="http://vllm-service-gemma-3-4b-sft:8000/v1", help="Instance 1 URL (Agent)")
    parser.add_argument("--url-user", default="http://vllm-service-gemma-3-4b-2:8000/v1", help="Instance 2 URL (User Simulator)")
    parser.add_argument("--tiers", nargs="+", type=int, default=[96], help="Concurrency tiers (default: 96)")
    parser.add_argument("--num-tasks", type=int, default=192, help="Number of tasks per tier")
    parser.add_argument("--task-timeout", type=float, default=120.0, help="Per-task timeout in seconds")
    parser.add_argument("--output-dir", default="benchmarks/capacity/results/exp4_sft")

    args = parser.parse_args()

    agent_args = {
        "api_base": args.url_agent,
        "api_key": "EMPTY",
        "max_tokens": 1024,
    }
    user_args = {
        "api_base": args.url_user,
        "api_key": "EMPTY",
        "max_tokens": 1024,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "exp4_sft_results.json")

    asyncio.run(
        run_tau_capacity_suite(
            domain=args.domain,
            agent_llm=args.agent_llm,
            agent_args=agent_args,
            user_llm=args.agent_llm,
            user_args=user_args,
            tiers=args.tiers,
            num_tasks=args.num_tasks,
            num_trials=1,
            max_steps=30,
            task_timeout=args.task_timeout,
            metrics_url=args.url_agent.replace("/v1", "") + "/metrics",
            output_json=output_json,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
