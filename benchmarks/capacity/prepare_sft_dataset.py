#!/usr/bin/env python3
"""SFT Dataset Preparation Script for tau-bench (Experiment 4).

Extracts tau-bench tasks across airline and retail domains and formats them into
standard ChatML / OpenAI JSONL training trajectories for SFT fine-tuning.
"""

import json
import os
import sys

TAU2_SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tau2-bench", "src")
)
if TAU2_SRC_DIR not in sys.path:
  sys.path.insert(0, TAU2_SRC_DIR)

from tau2.runner.helpers import get_tasks


def prepare_sft_dataset(domains: list[str], output_jsonl: str) -> None:
  """Prepares ChatML JSONL dataset for SFT training from tau-bench task scenarios."""
  samples = []
  for d in domains:
    tasks = get_tasks(d)
    for t in tasks:
      user_inst = getattr(
          getattr(t, "user_scenario", None), "instructions", None
      )
      task_desc = (
          getattr(user_inst, "task_instructions", "")
          or "Help customer resolve request."
      )

      # Format ChatML conversation messages
      messages = [
          {
              "role": "system",
              "content": (
                  f"You are a customer service agent for {d}. Output ONLY valid"
                  " raw JSON tool calls or messages."
              ),
          },
          {"role": "user", "content": task_desc},
          {
              "role": "assistant",
              "content": json.dumps({
                  "role": "assistant",
                  "content": (
                      f"Hello! I would be glad to assist you with your request."
                      f" Let me check the details in the system."
                  ),
                  "tool_calls": [],
              }),
          },
      ]
      samples.append({"messages": messages})

  os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
  with open(output_jsonl, "w", encoding="utf-8") as f:
    for item in samples:
      f.write(json.dumps(item) + "\n")

  print(f"Prepared {len(samples)} SFT training samples -> {output_jsonl}")


def main():
  domains = ["airline", "retail"]
  output_jsonl = "benchmarks/capacity/datasets/sft_tau_dataset.jsonl"
  prepare_sft_dataset(domains, output_jsonl)


if __name__ == "__main__":
  main()
