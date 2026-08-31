#!/usr/bin/env python3
"""
Automated Multi-Turn Agent Capacity Comparison Suite for tau-bench

Executes comparative capacity benchmarks across 3 deployment scenarios:
1. Scenario 1: Same local vLLM model for Agent and User Simulator (Shared Port 8000)
2. Scenario 2: Local vLLM Agent (Port 8000) + Remote Vertex AI User Simulator (Gemini 2.5 Flash)
3. Scenario 3: Different local vLLM models (Port 8000 & Port 8001)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

# Import benchmark runner module
from benchmark_tau_capacity import run_tau_capacity_suite

SCENARIO_CONFIGS = {
    "scenario_1_same_local": {
        "label": "Scenario 1: Same Local Model (Shared GPU / Port 8000)",
        "agent_llm": "openai/gemma-3-4b",
        "agent_args": {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY", "max_tokens": 1024},
        "user_llm": "openai/gemma-3-4b",
        "user_args": {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY", "max_tokens": 1024},
        "metrics_url": "http://localhost:8000/metrics",
    },
    "scenario_2_local_agent_vertex_user": {
        "label": "Scenario 2: Local Agent (Port 8000) + Vertex AI User",
        "agent_llm": "openai/gemma-3-4b",
        "agent_args": {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY", "max_tokens": 1024},
        "user_llm": "vertex_ai/gemini-2.5-flash",
        "user_args": {"max_tokens": 1024},
        "metrics_url": "http://localhost:8000/metrics",
    },
    "scenario_3_different_local": {
        "label": "Scenario 3: Different Local Models (Port 8000 & 8001)",
        "agent_llm": "openai/gemma-3-4b",
        "agent_args": {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY", "max_tokens": 1024},
        "user_llm": "openai/qwen-2.5-7b",
        "user_args": {"api_base": "http://localhost:8001/v1", "api_key": "EMPTY", "max_tokens": 1024},
        "metrics_url": "http://localhost:8000/metrics",
    },
}


def format_tau_comparison_table(results_by_scenario: Dict[str, Dict[str, Any]], title: str) -> str:
    """Formats a side-by-side comparative markdown/text table of tau-bench runs."""
    lines = []
    lines.append(f"\n{'='*150}")
    lines.append(f" MULTI-TURN AGENT CAPACITY COMPARATIVE REPORT: {title}")
    lines.append(f"{'='*150}")

    scenario_keys = list(results_by_scenario.keys())
    scenario_labels = [SCENARIO_CONFIGS.get(k, {}).get("label", k) for k in scenario_keys]

    header_cols = ["Concurrency", "Simulations"]
    for s_label in scenario_labels:
        short_name = s_label.split(":")[0]
        header_cols.extend([
            f"{short_name} Streams (Avg/Pk)",
            f"{short_name} Tok/s",
            f"{short_name} P50 (ms)",
            f"{short_name} KV %",
            f"{short_name} Pass %",
            f"{short_name} Timeouts",
        ])

    lines.append(" | ".join([f"{col:<20}" for col in header_cols]))
    lines.append("-" * 150)

    # Collect all concurrencies across scenarios
    concurrencies = set()
    for k, sdata in results_by_scenario.items():
        for r in sdata.get("results", []):
            concurrencies.add(r["concurrency"])

    for c in sorted(concurrencies):
        # find matching result row
        sim_count = "N/A"
        for k in scenario_keys:
            s_res = {r["concurrency"]: r for r in results_by_scenario[k].get("results", [])}.get(c)
            if s_res:
                sim_count = str(s_res.get("total_simulations", "N/A"))
                break

        row_values = [f"C={c:<18}", f"{sim_count:<18}"]
        for k in scenario_keys:
            s_res = {r["concurrency"]: r for r in results_by_scenario[k].get("results", [])}.get(c, {})
            if s_res:
                streams = f"{s_res.get('avg_active_streams', 0.0):.1f}/{s_res.get('peak_active_streams', 0)}"
                tok_s = f"{s_res.get('total_throughput_tok_s', 0.0):.1f}"
                p50_ms = f"{s_res.get('agent_turn_latency_p50_ms', 0.0):.1f}"
                kv_perc = f"{s_res.get('peak_kv_cache_perc', 0.0):.1f}%"
                pass_rate = f"{s_res.get('success_rate_perc', 0.0):.1f}%"
                timeouts = f"{s_res.get('timed_out_count', 0)}"
            else:
                streams, tok_s, p50_ms, kv_perc, pass_rate, timeouts = "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

            row_values.extend([f"{streams:<20}", f"{tok_s:<20}", f"{p50_ms:<20}", f"{kv_perc:<20}", f"{pass_rate:<20}", f"{timeouts:<20}"])
        lines.append(" | ".join(row_values))

    lines.append(f"{'='*150}\n")
    return "\n".join(lines)


async def main_async():
    parser = argparse.ArgumentParser(description="Multi-Turn Agent Capacity Comparison Runner (tau-bench)")
    parser.add_argument("--domain", default="airline", help="Domain name (e.g. airline, retail, mock)")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_CONFIGS.keys()), choices=list(SCENARIO_CONFIGS.keys()))
    parser.add_argument("--tiers", nargs="+", type=int, default=[8, 16, 32, 64, 96, 128, 160])
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks per tier")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum turns per task")
    parser.add_argument("--task-timeout", type=float, default=120.0, help="Per-sample timeout in seconds (default: 120)")
    parser.add_argument("--url-agent", default=None, help="Override agent vLLM endpoint URL")
    parser.add_argument("--url-user", default=None, help="Override user vLLM endpoint URL")
    parser.add_argument("--metrics-url", default=None, help="Override vLLM Prometheus metrics endpoint URL")
    parser.add_argument("--output-dir", default="benchmarks/capacity/results")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results_by_scenario = {}

    for sc_key in args.scenarios:
        sc_config = SCENARIO_CONFIGS[sc_key]
        print(f"\n=====================================================================")
        print(f" EXECUTION: {sc_config['label']}")
        print(f"=====================================================================")

        agent_args = sc_config["agent_args"].copy()
        user_args = sc_config["user_args"].copy()

        if args.url_agent and "api_base" in agent_args:
            agent_args["api_base"] = args.url_agent
            os.environ["OPENAI_API_BASE"] = args.url_agent
        if args.url_user and "api_base" in user_args:
            user_args["api_base"] = args.url_user

        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        if "api_key" in agent_args:
            os.environ["OPENAI_API_KEY"] = str(agent_args["api_key"])

        metrics_url = args.metrics_url
        if not metrics_url and args.url_agent:
            if "/v1" in args.url_agent:
                metrics_url = args.url_agent.split("/v1")[0] + "/metrics"
            else:
                metrics_url = args.url_agent.rstrip("/") + "/metrics"
        elif not metrics_url:
            metrics_url = sc_config.get("metrics_url")

        scenario_out_dir = os.path.join(args.output_dir, sc_key)
        os.makedirs(scenario_out_dir, exist_ok=True)
        output_json = os.path.join(scenario_out_dir, f"tau_capacity_{sc_key}.json")
        res = await run_tau_capacity_suite(
            domain=args.domain,
            agent_llm=sc_config["agent_llm"],
            agent_args=agent_args,
            user_llm=sc_config["user_llm"],
            user_args=user_args,
            tiers=args.tiers,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            max_steps=args.max_steps,
            task_timeout=args.task_timeout,
            metrics_url=metrics_url,
            output_json=output_json,
            output_dir=scenario_out_dir,
        )
        results_by_scenario[sc_key] = res

    table_str = format_tau_comparison_table(results_by_scenario, f"Domain: {args.domain.upper()}")
    print(table_str)

    summary_json_path = os.path.join(args.output_dir, "tau_comparison_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(results_by_scenario, f, indent=2)

    print(f"Comparison report complete! Summary JSON written to: {summary_json_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
