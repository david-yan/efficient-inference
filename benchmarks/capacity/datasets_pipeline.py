#!/usr/bin/env python3
"""
Datasets Pipeline for LLM Capacity Testing & Evaluation
Pulls pure, authentic academic benchmark datasets (GSM8K, MMLU) in clean raw format.
Few-shot CoT prefixes are dynamically prepended at runtime via benchmark_capacity.py.
"""

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List

GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MMLU_HF_API = "https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=all&split=test"


def pull_gsm8k() -> List[Dict[str, Any]]:
    """Pulls full authentic GSM8K test dataset in clean zero-shot format."""
    print("Downloading full GSM8K test dataset from OpenAI repository...")
    req = urllib.request.Request(GSM8K_TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
    samples = []
    with urllib.request.urlopen(req, timeout=30) as resp:
        for idx, line in enumerate(resp):
            line = line.decode("utf-8").strip()
            if not line:
                continue
            item = json.loads(line)
            question = item["question"]
            prompt = f"Question: {question}\nLet's think step by step."
            samples.append({
                "id": f"gsm8k-{idx:04d}",
                "dataset": "gsm8k",
                "question": question,
                "prompt": prompt,
                "expected_max_tokens": 256,
                "reference_answer": item.get("answer", ""),
            })
    print(f"Loaded {len(samples)} clean GSM8K test samples.")
    return samples


def pull_mmlu(max_samples: int = 1500) -> List[Dict[str, Any]]:
    """Pulls authentic MMLU test samples in clean zero-shot format."""
    print("Downloading MMLU test dataset from Hugging Face cais/mmlu...")
    samples = []
    offset = 0
    limit = 100
    while len(samples) < max_samples:
        url = f"{MMLU_HF_API}&offset={offset}&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("rows", [])
                if not rows:
                    break
                for row_data in rows:
                    row = row_data.get("row", {})
                    raw_subject = row.get("subject", "general")
                    subject_name = raw_subject.replace("_", " ")
                    question = row.get("question", "")
                    choices = row.get("choices", [])
                    choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                    prompt = (
                        f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"
                        f"Question: {question}\n"
                        f"{choices_str}\n"
                        f"Answer:"
                    )
                    samples.append({
                        "id": f"mmlu-{len(samples):04d}",
                        "dataset": "mmlu",
                        "subject": raw_subject,
                        "question": question,
                        "prompt": prompt,
                        "expected_max_tokens": 16,
                        "reference_answer": row.get("answer", ""),
                    })
                offset += len(rows)
                if len(rows) < limit:
                    break
        except Exception as e:
            print(f"Warning during MMLU download at offset {offset}: {e}")
            break
    print(f"Loaded {len(samples)} clean MMLU test samples.")
    return samples


def sort_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sorts samples by prompt prefix and length."""
    return sorted(samples, key=lambda x: (x.get("subject", x["dataset"]), len(x["prompt"])))


def save_jsonl(samples: List[Dict[str, Any]], filepath: str):
    """Saves records to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(samples)} samples to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Ingestion and Sorting Pipeline")
    parser.add_argument("--output-dir", default="benchmarks/capacity/datasets", help="Output directory")
    args = parser.parse_args()

    raw_dir = os.path.join(args.output_dir, "raw")
    sorted_dir = os.path.join(args.output_dir, "sorted")

    # 1. Pull Clean Datasets
    gsm8k_samples = pull_gsm8k()
    mmlu_samples = pull_mmlu(max_samples=1500)
    combined_samples = gsm8k_samples + mmlu_samples

    # 2. Save Raw Datasets
    save_jsonl(gsm8k_samples, os.path.join(raw_dir, "gsm8k.jsonl"))
    save_jsonl(mmlu_samples, os.path.join(raw_dir, "mmlu.jsonl"))
    save_jsonl(combined_samples, os.path.join(raw_dir, "combined.jsonl"))

    # 3. Save Sorted Datasets
    gsm8k_sorted = sort_samples(gsm8k_samples)
    mmlu_sorted = sort_samples(mmlu_samples)
    combined_sorted = sort_samples(combined_samples)

    save_jsonl(gsm8k_sorted, os.path.join(sorted_dir, "gsm8k_sorted.jsonl"))
    save_jsonl(mmlu_sorted, os.path.join(sorted_dir, "mmlu_sorted.jsonl"))
    save_jsonl(combined_sorted, os.path.join(sorted_dir, "combined_sorted.jsonl"))

    print("\nDataset preparation completed successfully with clean raw samples!")


if __name__ == "__main__":
    main()
