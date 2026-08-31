#!/usr/bin/env python3
"""
Trajectory Inspector for tau-bench experiments.

Downloads trajectory files from GCS and inspects turn-by-turn assistant messages
to identify failing tasks where the model started with valid JSON in early turns
and degenerated into repeating / malformed JSON in later turns.
"""

import json
import os
import subprocess
import sys


def download_sample_trajectories(gcs_prefix: str, local_dir: str, limit: int = 30) -> list[str]:
    """Downloads a sample of trajectory JSON files from GCS."""
    os.makedirs(local_dir, exist_ok=True)
    cmd = f"gcloud storage ls {gcs_prefix}*_trajectory.json | head -n {limit}"
    output = subprocess.check_output(cmd, shell=True, text=True)
    gcs_urls = [line.strip() for line in output.strip().split("\n") if line.strip()]

    local_files = []
    for url in gcs_urls:
        fname = os.path.basename(url)
        local_path = os.path.join(local_dir, fname)
        if not os.path.exists(local_path):
            dl_cmd = f"gcloud storage cp {url} {local_path}"
            subprocess.run(dl_cmd, shell=True, check=False)
        if os.path.exists(local_path):
            local_files.append(local_path)
    return local_files


def analyze_trajectory(filepath: str) -> dict:
    """Analyzes a trajectory file to trace turn-by-turn assistant outputs."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    task_id = data.get("task_id", os.path.basename(filepath))
    messages = data.get("messages", [])

    assistant_turns = []
    for idx, msg in enumerate(messages):
        role = msg.get("role") or msg.get("from_role")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "assistant" or role == "Role.AGENT":
            is_valid_json = False
            is_repetition = False
            if content and content.strip():
                text = content.strip()
                try:
                    json.loads(text)
                    is_valid_json = True
                except Exception:
                    is_valid_json = False

                if "content" in text and text.count("content") > 2:
                    is_repetition = True
                if text.count("```") > 1:
                    is_repetition = True

            assistant_turns.append({
                "turn_index": len(assistant_turns) + 1,
                "msg_idx": idx,
                "content": content[:150] + ("..." if len(content) > 150 else ""),
                "full_content": content,
                "tool_calls": tool_calls,
                "is_valid_json": is_valid_json,
                "is_repetition": is_repetition,
            })

    return {
        "task_id": task_id,
        "filepath": filepath,
        "total_messages": len(messages),
        "total_assistant_turns": len(assistant_turns),
        "assistant_turns": assistant_turns,
    }


def main():
    exp1_gcs = "gs://efficient-inference-506713-models/benchmarks/capacity/exp1_strict_prompt/transcripts/tier_C96/"
    local_dir = "/tmp/exp1_trajectories"

    print("Downloading sample trajectory JSONs from GCS...")
    files = download_sample_trajectories(exp1_gcs, local_dir, limit=30)
    print(f"Downloaded {len(files)} trajectory files.")

    degenerated_examples = []

    for f in files:
        analysis = analyze_trajectory(f)
        turns = analysis.get("assistant_turns", [])
        if len(turns) >= 2:
            # Check if early turns had valid JSON or normal outputs and later turns degenerated
            early_clean = any(t["is_valid_json"] or len(t["tool_calls"]) > 0 for t in turns[:2])
            later_degenerated = any(t["is_repetition"] or not t["is_valid_json"] for t in turns[2:])
            if early_clean and later_degenerated:
                degenerated_examples.append(analysis)

    print(f"\nFound {len(degenerated_examples)} tasks exhibiting early valid -> later degenerated turns:\n")

    for ex in degenerated_examples[:5]:
        print("==========================================================================")
        print(f"TASK ID: {ex['task_id']} ({os.path.basename(ex['filepath'])})")
        print(f"Total Assistant Turns: {ex['total_assistant_turns']}")
        print("--------------------------------------------------------------------------")
        for t in ex['assistant_turns']:
            status = "VALID JSON" if t['is_valid_json'] else ("DEGENERATED/REPETITION" if t['is_repetition'] else "INVALID JSON / RAW TEXT")
            print(f" [Turn {t['turn_index']}] {status}")
            print(f"   Snippet: {t['content']}")
        print("==========================================================================\n")


if __name__ == "__main__":
    main()
