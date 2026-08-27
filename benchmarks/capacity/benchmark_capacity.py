#!/usr/bin/env python3
"""
vLLM Capacity Testing & Characterization Harness
Supports:
  - Evaluation strategies:
      (1) 'random_sample': Uniform random sampling across dataset (low prefix sharing)
      (2) 'random_slice': Contiguous window from prefix-sorted dataset (high prefix sharing)
  - Full academic datasets (GSM8K, MMLU, Combined)
  - Inter-tier cache isolation without intra-tier prefix destruction
  - Real-time Prometheus telemetry & dynamic concurrency scaling
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid
from typing import Any, Dict, List, Optional
import aiohttp
import numpy as np

WORKLOAD_PRESETS = {
    "batch": {
        "description": "Offline Batch / Academic Benchmarking (Max Throughput)",
        "input_len": 1024,
        "output_len": 256,
        "concurrency_tiers": [1, 4, 8, 16, 32, 64, 96, 128, 192, 256],
        "shared_prefix_len": 0,
        "multi_turn": 1,
        "multiplier": 2,
    },
    "agent": {
        "description": "Multi-Turn Agent Evals (Prefix Caching & TTFT)",
        "input_len": 512,
        "output_len": 100,
        "concurrency_tiers": [1, 2, 4, 8, 12, 16, 24, 32],
        "shared_prefix_len": 2500,
        "multi_turn": 3,
        "multiplier": 2,
    },
    "interactive": {
        "description": "Real-time Serving SLA Characterization",
        "input_len": 512,
        "output_len": 128,
        "concurrency_tiers": [1, 2, 4, 8, 12, 16, 24],
        "shared_prefix_len": 0,
        "multi_turn": 1,
        "multiplier": 3,
    },
}

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# Standard 8-shot Chain-of-Thought exemplars for GSM8K (Wei et al. / OpenAI Grade School Math)
GSM8K_8SHOT_COT = """The following are math word problems with step-by-step reasoning solutions.

Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Solution: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. The answer is 6.

Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Solution: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Solution: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Solution: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.

Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Solution: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then he got 2 + 2 = 4 toys. 5 + 4 = 9 toys. The answer is 9.

Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Solution: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Solution: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Solution: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

"""

# Standard 5-shot exemplars for MMLU multiple choice
MMLU_5SHOT_PREFIX = """Question: What is the primary function of mitochondria in eukaryotic cells?
A. Photosynthesis
B. ATP cellular respiration
C. Protein packaging
D. Lipid degradation
Answer: B

Question: If f(x) = 3x^2 - 4x + 1, what is f'(2)?
A. 8
B. 12
C. 10
D. 6
Answer: A

Question: Which treaty ended the Thirty Years' War in 1648?
A. Treaty of Utrecht
B. Peace of Westphalia
C. Treaty of Versailles
D. Treaty of Ghent
Answer: B

Question: What is the time complexity of searching in a balanced binary search tree with N elements?
A. O(1)
B. O(N)
C. O(log N)
D. O(N log N)
Answer: C

Question: Which component of a CPU directs the operation of the processor?
A. ALU
B. Control Unit
C. Register File
D. Cache
Answer: B

"""


def format_prompt(sample: Dict[str, Any], prompt_mode: str = "zero_shot", tier_salt: str = "") -> str:
    """Formats the prompt dynamically according to prompt_mode ('zero_shot' vs 'few_shot')."""
    raw_prompt = sample.get("prompt", "")
    prefix_str = f"[TierSalt: {tier_salt}] " if tier_salt else ""

    if prompt_mode in ("few_shot", "cot"):
        dataset_type = sample.get("dataset", "").lower()
        if "gsm8k" in dataset_type:
            question = sample.get("question")
            if not question and "Question: " in raw_prompt:
                q_part = raw_prompt.split("Question: ", 1)[-1]
                question = q_part.split("\nLet's think step by step.", 1)[0]
            if question:
                return f"{prefix_str}{GSM8K_8SHOT_COT}Problem: {question}\nSolution: Let's think step by step."
            return f"{prefix_str}{GSM8K_8SHOT_COT}{raw_prompt}"
        elif "mmlu" in dataset_type:
            subject = sample.get("subject", "")
            subject_name = subject.replace("_", " ") if subject else ""
            if subject_name:
                header = f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"
            else:
                header = ""
            core_prompt = raw_prompt.replace(header, "") if header and raw_prompt.startswith(header) else raw_prompt
            return f"{prefix_str}{header}{MMLU_5SHOT_PREFIX}{core_prompt}"
        else:
            return f"{prefix_str}System: You are an expert AI assistant solving academic evaluation questions with rigorous reasoning.\n\n{raw_prompt}"

    return f"{prefix_str}{raw_prompt}"



def load_dataset_prompts(dataset_name_or_path: str, strategy: str = "random_sample") -> List[Dict[str, Any]]:
    """Loads benchmark prompts from a JSONL file or built-in dataset directory."""
    path = dataset_name_or_path
    if not os.path.exists(path):
        subfolder = "sorted" if strategy == "random_slice" else "raw"
        suffix = "_sorted.jsonl" if strategy == "random_slice" else ".jsonl"
        candidates = [
            os.path.join(DEFAULT_DATASET_DIR, subfolder, f"{dataset_name_or_path}{suffix}"),
            os.path.join(DEFAULT_DATASET_DIR, "sorted", f"{dataset_name_or_path}_sorted.jsonl"),
            os.path.join(DEFAULT_DATASET_DIR, "raw", f"{dataset_name_or_path}.jsonl"),
            os.path.join(DEFAULT_DATASET_DIR, f"{dataset_name_or_path}.jsonl"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"Could not locate dataset '{dataset_name_or_path}'")

    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def select_tier_samples(
    dataset: List[Dict[str, Any]],
    num_requests: int,
    strategy: str = "random_sample",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Selects sample requests according to the chosen strategy:
      - 'random_sample': Uniformly randomly chosen without replacement
      - 'random_slice': Contiguous window from the prefix-sorted dataset
      - 'sequential': First N items
    """
    rng = random.Random(seed)
    n = len(dataset)
    if num_requests >= n:
        repeats = (num_requests // n) + 1
        expanded = (dataset * repeats)[:num_requests]
        return expanded

    if strategy == "random_slice":
        max_start = n - num_requests
        start_idx = rng.randint(0, max_start)
        return dataset[start_idx : start_idx + num_requests]
    elif strategy == "random_sample":
        return rng.sample(dataset, num_requests)
    else:
        return dataset[:num_requests]


def generate_prompt(input_len: int, shared_prefix: str = "", req_idx: int = 0, tier_salt: str = "") -> str:
    """Generates a synthetic prompt matching the requested token length."""
    salt_header = f"[Session: {tier_salt}] " if tier_salt else ""
    base_text = f"The quick brown fox jumps over the lazy dog. Artificial intelligence inference optimization on GPUs. "
    multiplier = max(1, (input_len * 4) // len(base_text))
    body = (base_text * multiplier)[: input_len * 4]
    if shared_prefix:
        return f"{salt_header}{shared_prefix}\n\nTask: Analyze text item {req_idx}.\n\nContext:\n{body}"
    return f"{salt_header}Task: Analyze text item {req_idx}.\n\nContext:\n{body}"


async def fetch_vllm_metrics(session: aiohttp.ClientSession, base_url: str) -> Dict[str, float]:
    """Scrapes /metrics from the vLLM server to inspect GPU KV cache and queue depth."""
    metrics_url = f"{base_url.rstrip('/v1')}/metrics"
    out = {
        "kv_cache_usage_perc": 0.0,
        "num_requests_running": 0.0,
        "num_requests_waiting": 0.0,
        "prefix_cache_hits_total": 0.0,
        "prefix_cache_queries_total": 0.0,
    }
    try:
        async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
            if resp.status == 200:
                text = await resp.text()
                for line in text.splitlines():
                    if line.startswith("#"):
                        continue
                    if line.startswith("vllm:kv_cache_usage_perc") or line.startswith("vllm:gpu_cache_usage_factor"):
                        out["kv_cache_usage_perc"] = float(line.split()[-1])
                    elif line.startswith("vllm:num_requests_running"):
                        out["num_requests_running"] = float(line.split()[-1])
                    elif line.startswith("vllm:num_requests_waiting"):
                        out["num_requests_waiting"] = float(line.split()[-1])
                    elif line.startswith("vllm:prefix_cache_hits_total"):
                        out["prefix_cache_hits_total"] = float(line.split()[-1])
                    elif line.startswith("vllm:prefix_cache_queries_total"):
                        out["prefix_cache_queries_total"] = float(line.split()[-1])
    except Exception:
        pass
    return out


async def monitor_metrics_loop(session: aiohttp.ClientSession, base_url: str, stop_event: asyncio.Event, stats: Dict[str, float]):
    """Polls vLLM metrics in the background while requests are actively executing."""
    while not stop_event.is_set():
        m = await fetch_vllm_metrics(session, base_url)
        stats["peak_kv_cache_perc"] = max(stats.get("peak_kv_cache_perc", 0.0), m.get("kv_cache_usage_perc", 0.0))
        stats["peak_waiting"] = max(stats.get("peak_waiting", 0.0), m.get("num_requests_waiting", 0.0))
        stats["peak_running"] = max(stats.get("peak_running", 0.0), m.get("num_requests_running", 0.0))
        await asyncio.sleep(0.05)


async def send_single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """Sends a single streaming completion request and captures TTFT and TPOT."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    endpoint = f"{base_url.rstrip('/')}/v1/completions"
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    token_chunks = 0

    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with session.post(endpoint, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                return {"error": f"HTTP {resp.status}: {err_text[:200]}"}

            async for line in resp.content:
                decoded = line.decode("utf-8", errors="ignore").strip()
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_chunks += 1

        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
        decode_duration = (end_time - first_token_time) if first_token_time else 0.0
        tpot = (decode_duration / max(1, token_chunks - 1)) if token_chunks > 1 else 0.0

        return {
            "success": True,
            "latency_sec": end_time - start_time,
            "ttft_sec": ttft,
            "tpot_sec": tpot,
            "output_tokens": token_chunks,
        }
    except Exception as e:
        return {"error": str(e)}


async def run_concurrency_tier(
    base_url: str,
    model: str,
    concurrency: int,
    samples: List[Dict[str, Any]],
    input_len: int,
    output_len: int,
    shared_prefix: str = "",
    multi_turn: int = 1,
    tier_salt: str = "",
    prompt_mode: str = "zero_shot",
) -> Dict[str, Any]:
    """Executes a load tier with fixed concurrency and selected samples."""
    semaphore = asyncio.Semaphore(concurrency)
    num_requests = len(samples)

    async with aiohttp.ClientSession() as session:
        m_start = await fetch_vllm_metrics(session, base_url)
        poll_stats = {"peak_kv_cache_perc": 0.0, "peak_waiting": 0.0, "peak_running": 0.0}
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_metrics_loop(session, base_url, stop_event, poll_stats))

        async def worker(idx: int):
            async with semaphore:
                results = []
                sample = samples[idx]
                current_prompt = format_prompt(sample, prompt_mode=prompt_mode, tier_salt=tier_salt)
                current_output_len = sample.get("expected_max_tokens", output_len)

                for turn in range(multi_turn):
                    res = await send_single_request(
                        session, base_url, model, current_prompt, current_output_len
                    )
                    results.append(res)
                    if not res.get("success"):
                        break
                    current_prompt += f"\nFollow-up question turn {turn + 2}: Provide further details."
                return results

        t_start = time.perf_counter()
        tasks = [worker(i) for i in range(num_requests)]
        raw_results = await asyncio.gather(*tasks)
        t_wall = time.perf_counter() - t_start

        stop_event.set()
        await monitor_task
        m_end = await fetch_vllm_metrics(session, base_url)

    d_hits = m_end.get("prefix_cache_hits_total", 0.0) - m_start.get("prefix_cache_hits_total", 0.0)
    d_queries = m_end.get("prefix_cache_queries_total", 0.0) - m_start.get("prefix_cache_queries_total", 0.0)
    hit_rate = (d_hits / max(1.0, d_queries)) * 100 if d_queries > 0 else 0.0

    all_turns = [turn for req_turns in raw_results for turn in req_turns]
    valid = [r for r in all_turns if r.get("success")]
    errors = [r for r in all_turns if not r.get("success")]

    total_output_tokens = sum(r["output_tokens"] for r in valid)
    approx_input_tokens = len(valid) * input_len
    total_tokens = approx_input_tokens + total_output_tokens

    ttfts = [r["ttft_sec"] * 1000 for r in valid]
    tpots = [r["tpot_sec"] * 1000 for r in valid]
    latencies = [r["latency_sec"] for r in valid]

    return {
        "concurrency": concurrency,
        "total_requests": len(all_turns),
        "completed": len(valid),
        "errors": len(errors),
        "wall_time_sec": round(t_wall, 2),
        "throughput_tokens_per_sec": round(total_tokens / max(0.001, t_wall), 2),
        "throughput_requests_per_sec": round(len(valid) / max(0.001, t_wall), 2),
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0.0,
        "ttft_p95_ms": round(float(np.percentile(ttfts, 95)), 2) if ttfts else 0.0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 2) if tpots else 0.0,
        "tpot_p95_ms": round(float(np.percentile(tpots, 95)), 2) if tpots else 0.0,
        "latency_p50_sec": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        "latency_p95_sec": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "peak_kv_cache_perc": round(poll_stats["peak_kv_cache_perc"] * 100, 1),
        "prefix_hit_rate_perc": round(hit_rate, 1),
        "peak_requests_waiting": int(poll_stats["peak_waiting"]),
    }


def print_summary_table(results: List[Dict[str, Any]], profile_name: str, model_name: str, dataset_name: Optional[str] = None, strategy: str = "random_sample", prompt_mode: str = "zero_shot"):
    """Prints a formatted ASCII table of the benchmark sweep."""
    ds_str = f" | Dataset: {dataset_name}" if dataset_name else ""
    pm_str = f" | Mode: {prompt_mode}"
    header = (
        f"\n===================================================================================================================\n"
        f" CAPACITY BENCHMARK REPORT: {model_name} (Profile: {profile_name}{ds_str}{pm_str} | Strategy: {strategy})\n"
        f"===================================================================================================================\n"
        f" Concurrency | Requests | Tok/s     | Req/s  | TTFT P50 (ms) | TPOT P50 (ms) | Lat P95 (s) | Peak KV % | Prefix % | Wait\n"
        f"-------------+----------+-----------+--------+---------------+---------------+-------------+-----------+----------+------"
    )
    print(header)
    for r in results:
        print(
            f" {r['concurrency']:<11} | "
            f"{r['total_requests']:<8} | "
            f"{r['throughput_tokens_per_sec']:<9} | "
            f"{r['throughput_requests_per_sec']:<6} | "
            f"{r['ttft_p50_ms']:<13} | "
            f"{r['tpot_p50_ms']:<13} | "
            f"{r['latency_p95_sec']:<11} | "
            f"{r['peak_kv_cache_perc']:<9}% | "
            f"{r['prefix_hit_rate_perc']:<8}% | "
            f"{r['peak_requests_waiting']}"
        )
    print("===================================================================================================================\n")


async def run_benchmark_suite(
    url: str,
    model: str,
    profile: str,
    dataset: Optional[str] = None,
    strategy: str = "random_sample",
    prompt_mode: str = "zero_shot",
    concurrency_tiers: Optional[List[int]] = None,
    requests_per_tier: Optional[int] = None,
    multiplier: int = 2,
    isolate_cache: bool = True,
    resume: bool = False,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs a complete capacity benchmark suite with incremental resume support."""
    cfg = WORKLOAD_PRESETS.get(profile, WORKLOAD_PRESETS["batch"]).copy()
    all_tiers = sorted(concurrency_tiers or cfg["concurrency_tiers"])

    # Load dataset based on strategy
    if dataset:
        all_samples = load_dataset_prompts(dataset, strategy=strategy)
    else:
        all_samples = [{"prompt": generate_prompt(cfg["input_len"], "", i), "expected_max_tokens": cfg["output_len"]} for i in range(1000)]

    completed_tier_map: Dict[int, Dict[str, Any]] = {}
    if resume and output_json and os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                existing = json.load(f)
                for r in existing.get("results", []):
                    completed_tier_map[r["concurrency"]] = r
            if completed_tier_map:
                print(f"Resuming {output_json}: Found {len(completed_tier_map)} existing completed tiers {sorted(completed_tier_map.keys())}.")
        except Exception as e:
            print(f"Warning loading existing output {output_json}: {e}")

    tiers_to_run = [c for c in all_tiers if c not in completed_tier_map]

    print(f"\n=======================================================")
    print(f"Capacity Suite: {dataset or 'Synthetic'} [{strategy.upper()}] (Mode: {prompt_mode})")
    print(f"Endpoint: {url} | Model: {model}")
    print(f"Tiers Target: {all_tiers} | Pending Tiers: {tiers_to_run}")
    print(f"=======================================================")

    for c in tiers_to_run:
        req_count = requests_per_tier or max(32, c * multiplier)
        selected_samples = select_tier_samples(all_samples, req_count, strategy=strategy, seed=c + 42)
        tier_salt = f"tier-c{c}-{uuid.uuid4().hex[:6]}" if isolate_cache else ""

        print(f"--> Tier {c}: Dispatching {req_count} requests [{strategy} | {prompt_mode}] (Salt: {tier_salt or 'None'})...")
        res = await run_concurrency_tier(
            base_url=url,
            model=model,
            concurrency=c,
            samples=selected_samples,
            input_len=cfg["input_len"],
            output_len=cfg["output_len"],
            shared_prefix="",
            multi_turn=cfg.get("multi_turn", 1),
            tier_salt=tier_salt,
            prompt_mode=prompt_mode,
        )
        completed_tier_map[c] = res
        print(f"    Done: {res['completed']}/{res['total_requests']} | Tok/s: {res['throughput_tokens_per_sec']} | TTFT: {res['ttft_p50_ms']}ms | Peak KV: {res['peak_kv_cache_perc']}% | Prefix: {res['prefix_hit_rate_perc']}%")

        if output_json:
            sorted_res = [completed_tier_map[k] for k in sorted(completed_tier_map.keys())]
            report_data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model,
                "endpoint": url,
                "profile": profile,
                "dataset": dataset,
                "strategy": strategy,
                "prompt_mode": prompt_mode,
                "total_requests": sum(r["total_requests"] for r in sorted_res),
                "cache_isolated": isolate_cache,
                "results": sorted_res,
            }
            os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(report_data, f, indent=2)

    final_results = [completed_tier_map[k] for k in sorted(completed_tier_map.keys())]
    print_summary_table(final_results, profile, model, dataset, strategy, prompt_mode=prompt_mode)

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "endpoint": url,
        "profile": profile,
        "dataset": dataset,
        "strategy": strategy,
        "prompt_mode": prompt_mode,
        "total_requests": sum(r["total_requests"] for r in final_results),
        "cache_isolated": isolate_cache,
        "results": final_results,
    }


async def main_async():
    parser = argparse.ArgumentParser(description="vLLM Capacity & Stress Benchmark Runner")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM Base URL")
    parser.add_argument("--model", default="gemma-3-4b", help="Model Name")
    parser.add_argument("--profile", choices=["batch", "agent", "interactive", "custom"], default="batch")
    parser.add_argument("--dataset", default=None, help="Dataset name ('gsm8k', 'mmlu', 'combined') or JSONL path")
    parser.add_argument("--strategy", choices=["random_sample", "random_slice", "sequential"], default="random_sample")
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot", "cot"], default="zero_shot", help="Prompt formatting mode")
    parser.add_argument("--concurrency-tiers", nargs="+", type=int, default=None)
    parser.add_argument("--requests-per-tier", type=int, default=None)
    parser.add_argument("--concurrency-multiplier", type=int, default=2)
    parser.add_argument("--no-isolate-cache", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Enable incremental resume")
    parser.add_argument("--output-json", default=None)

    args = parser.parse_args()

    await run_benchmark_suite(
        url=args.url,
        model=args.model,
        profile=args.profile,
        dataset=args.dataset,
        strategy=args.strategy,
        prompt_mode=args.prompt_mode,
        concurrency_tiers=args.concurrency_tiers,
        requests_per_tier=args.requests_per_tier,
        multiplier=args.concurrency_multiplier,
        isolate_cache=not args.no_isolate_cache,
        resume=args.resume,
        output_json=args.output_json,
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
