#!/usr/bin/env python3
"""
Automated Evaluation Comparison Suite for vLLM Serving Capacity
Runs comparative analysis across (1) Random Sample vs. (2) Random Slice.
Accurately measures true prefix caching acceleration and throughput curves.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Import the benchmark runner module
from benchmark_capacity import run_benchmark_suite


def format_comparative_table(results_a: Dict[str, Any], results_b: Dict[str, Any], label_a: str, label_b: str, title: str) -> str:
    """Formats a side-by-side comparative table of two benchmark runs."""
    lines = []
    lines.append(f"\n{'='*115}")
    lines.append(f" COMPARATIVE EFFICIENCY REPORT: {title}")
    lines.append(f" Strategy A: {label_a} vs. Strategy B: {label_b}")
    lines.append(f"{'='*115}")
    lines.append(f" Concurrency | Requests | {'Throughput (Tok/s)':<22} | {'TTFT P50 (ms)':<22} | {'Prefix Hit %':<16} | Peak KV %")
    lines.append(f"             |          | {label_a:<10} | {label_b:<10} | {label_a:<10} | {label_b:<10} | {label_a:<7} | {label_b:<7} | {label_a} / {label_b}")
    lines.append(f"{'-'*13}+" + f"{'-'*10}+" + f"{'-'*24}+" + f"{'-'*24}+" + f"{'-'*18}+" + f"{'-'*18}")

    tiers_a = {r["concurrency"]: r for r in results_a["results"]}
    tiers_b = {r["concurrency"]: r for r in results_b["results"]}
    all_concurrencies = sorted(set(list(tiers_a.keys()) + list(tiers_b.keys())))

    for c in all_concurrencies:
        ra = tiers_a.get(c, {})
        rb = tiers_b.get(c, {})
        req_cnt = ra.get("total_requests", rb.get("total_requests", 0))

        tok_a = f"{ra.get('throughput_tokens_per_sec', 0.0):.1f}" if ra else "N/A"
        tok_b = f"{rb.get('throughput_tokens_per_sec', 0.0):.1f}" if rb else "N/A"
        ttft_a = f"{ra.get('ttft_p50_ms', 0.0):.1f}" if ra else "N/A"
        ttft_b = f"{rb.get('ttft_p50_ms', 0.0):.1f}" if rb else "N/A"
        pref_a = f"{ra.get('prefix_hit_rate_perc', 0.0):.1f}%" if ra else "N/A"
        pref_b = f"{rb.get('prefix_hit_rate_perc', 0.0):.1f}%" if rb else "N/A"
        kv_a = f"{ra.get('peak_kv_cache_perc', 0.0):.1f}%" if ra else "N/A"
        kv_b = f"{rb.get('peak_kv_cache_perc', 0.0):.1f}%" if rb else "N/A"

        lines.append(f" {c:<11} | {req_cnt:<8} | {tok_a:<10} | {tok_b:<10} | {ttft_a:<10} | {ttft_b:<10} | {pref_a:<7} | {pref_b:<7} | {kv_a} / {kv_b}")

    lines.append(f"{'='*115}\n")
    return "\n".join(lines)


async def main_async():
    env_tiers = os.getenv("CONCURRENCY_TIERS")
    default_tiers = [int(x) for x in env_tiers.split()] if env_tiers else [1, 4, 8, 16, 32, 64, 96, 128, 192, 256]

    env_datasets = os.getenv("DATASETS")
    default_datasets = env_datasets.split() if env_datasets else ["gsm8k", "mmlu", "combined"]

    parser = argparse.ArgumentParser(description="Multi-Eval Capacity Comparison Runner")
    parser.add_argument("--url", default=os.getenv("ENDPOINT_URL", "http://localhost:8000"), help="vLLM URL")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gemma-3-4b"), help="Model Name")
    parser.add_argument("--tiers", nargs="+", type=int, default=default_tiers)
    parser.add_argument("--datasets", nargs="+", type=str, default=default_datasets)
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot", "cot"], default=os.getenv("PROMPT_MODE", "zero_shot"), help="Prompt formatting mode (zero_shot or few_shot)")
    parser.add_argument("--multiplier", type=int, default=int(os.getenv("CONCURRENCY_MULTIPLIER", "2")))
    parser.add_argument("--output-dir", default=os.getenv("GCS_OUTPUT_DIR", "benchmarks/capacity/results"))
    parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_labels = {
        "gsm8k": "GSM8K Math Reasoning",
        "mmlu": "MMLU Multi-Subject QA",
        "combined": "Combined Aggregated Evals",
    }

    all_comparison_data = {}

    for dataset_key in args.datasets:
        dataset_label = dataset_labels.get(dataset_key, dataset_key.upper())
        print(f"\n#####################################################################")
        print(f" EVALUATION: {dataset_label} (Tiers: {args.tiers} | Mode: {args.prompt_mode})")
        print(f"#####################################################################")

        # 1. Run Random Sample
        out_json_sample = os.path.join(args.output_dir, f"{dataset_key}_random_sample.json")
        res_sample = await run_benchmark_suite(
            url=args.url,
            model=args.model,
            profile="batch",
            dataset=dataset_key,
            strategy="random_sample",
            prompt_mode=args.prompt_mode,
            concurrency_tiers=args.tiers,
            multiplier=args.multiplier,
            isolate_cache=True,
            resume=args.resume,
            output_json=out_json_sample,
        )

        # 2. Run Random Slice
        out_json_slice = os.path.join(args.output_dir, f"{dataset_key}_random_slice.json")
        res_slice = await run_benchmark_suite(
            url=args.url,
            model=args.model,
            profile="batch",
            dataset=dataset_key,
            strategy="random_slice",
            prompt_mode=args.prompt_mode,
            concurrency_tiers=args.tiers,
            multiplier=args.multiplier,
            isolate_cache=True,
            resume=args.resume,
            output_json=out_json_slice,
        )

        table_str = format_comparative_table(
            res_sample, res_slice, "Random Sample", "Random Slice", dataset_label
        )
        print(table_str)

        all_comparison_data[dataset_key] = {
            "label": dataset_label,
            "sample": res_sample,
            "slice": res_slice,
            "table": table_str,
        }

    # Save comprehensive summary report
    summary_path = os.path.join(args.output_dir, "eval_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_comparison_data, f, indent=2)

    print(f"\nAll comparative runs completed! Full summary written to: {summary_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
