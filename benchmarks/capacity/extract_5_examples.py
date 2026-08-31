#!/usr/bin/env python3
"""
Extract 5 concrete trajectory examples from GCS showing early valid JSON turns -> later degenerated turns.
"""

import json
import subprocess
import sys

TASK_IDS = ["0", "10", "36", "43", "44", "60", "76", "89", "104", "108", "109"]
GCS_BASE = "gs://efficient-inference-506713-models/benchmarks/capacity/exp1_strict_prompt/transcripts/tier_C96"

found_examples = []

for tid in TASK_IDS:
    url = f"{GCS_BASE}/task_{tid}_trial_1_trajectory.json"
    cmd = f"gcloud storage cat {url}"
    try:
        raw_json = subprocess.check_output(cmd, shell=True, text=True)
        data = json.loads(raw_json)
    except Exception:
        continue

    messages = data.get("messages", [])
    assistant_turns = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "assistant":
            assistant_turns.append({
                "content": content,
                "tool_calls": tool_calls,
            })

    if len(assistant_turns) >= 2:
        found_examples.append({
            "task_id": tid,
            "duration": data.get("duration"),
            "termination_reason": data.get("termination_reason"),
            "turns": assistant_turns,
        })
        if len(found_examples) >= 5:
            break

print(f"Extracted {len(found_examples)} trajectory examples:\n")
for ex in found_examples:
    print("=" * 80)
    print(f"TASK ID: {ex['task_id']} | Termination: {ex['termination_reason']} | Duration: {ex['duration']:.1f}s")
    print("=" * 80)
    for idx, t in enumerate(ex['turns'], 1):
        content_preview = repr(t['content'][:120])
        tc_preview = f" [Tool Calls: {len(t['tool_calls'])}]" if t['tool_calls'] else ""
        print(f" Turn {idx:2d}: {content_preview}{tc_preview}")
    print()
