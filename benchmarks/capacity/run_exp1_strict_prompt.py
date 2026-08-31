#!/usr/bin/env python3
"""
Experiment 1: Strict System Instructions & Formatting Prompts for tau-bench (Dual-Node Config).

Enforces strict JSON output formatting, eliminates introductory text and markdown codeblocks,
and caps max generation tokens (1024) across 2 dedicated vLLM inference nodes (Instance 1 & Instance 2).
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

from tau2.agent.llm_agent import LLMAgent
from tau2.registry import registry
from benchmark_tau_capacity import run_tau_capacity_suite

STRICT_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.

CRITICAL FORMATTING CONSTRAINTS:
1. You MUST output ONLY a single, raw, valid JSON object.
2. Do NOT output any introductory text, conversational chatter, or explanations outside the JSON object.
3. Do NOT use markdown codeblocks (do NOT include ``` or ```json).
4. In each turn, you must output EXACTLY ONE of the following JSON formats:
   - For User Message:
     {"role": "assistant", "content": "<message to user>"}
   - For Tool Call:
     {"role": "assistant", "content": "", "tool_calls": [{"name": "<tool_name>", "arguments": {<args_dict>}}]}
5. Never combine conversational text and tool calls in the same turn.
6. Make sure all strings are properly escaped valid JSON.
""".strip()

STRICT_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class StrictPromptLLMAgent(LLMAgent):
    """LLM Agent with strict JSON output formatting instructions."""

    @property
    def system_prompt(self) -> str:
        return STRICT_SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=STRICT_AGENT_INSTRUCTION,
        )


def create_strict_llm_agent(tools, domain_policy, **kwargs):
    return StrictPromptLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


registry.register_agent_factory(create_strict_llm_agent, "strict_llm_agent")


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Strict System Instructions Runner (Dual-Node)")
    parser.add_argument("--domain", default="airline,retail", help="Comma-separated domains")
    parser.add_argument("--agent-llm", default="openai/gemma-3-4b", help="Agent model endpoint name")
    parser.add_argument("--url-agent", default="http://vllm-service-gemma-3-4b:8000/v1", help="Instance 1 URL (Agent)")
    parser.add_argument("--url-user", default="http://vllm-service-gemma-3-4b-2:8000/v1", help="Instance 2 URL (User Simulator)")
    parser.add_argument("--tiers", nargs="+", type=int, default=[96], help="Concurrency tiers (default: 96)")
    parser.add_argument("--num-tasks", type=int, default=192, help="Number of tasks per tier")
    parser.add_argument("--task-timeout", type=float, default=120.0, help="Per-task timeout in seconds")
    parser.add_argument("--output-dir", default="benchmarks/capacity/results/exp1_strict_prompt")

    args = parser.parse_args()

    agent_args = {
        "api_base": args.url_agent,
        "api_key": "EMPTY",
        "max_tokens": 1024,
        "stop": ["```", "\n\nUser:"],
    }
    user_args = {
        "api_base": args.url_user,
        "api_key": "EMPTY",
        "max_tokens": 1024,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "exp1_strict_prompt_results.json")

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
            agent_name="strict_llm_agent",
            max_steps=30,
            task_timeout=args.task_timeout,
            metrics_url=args.url_agent.replace("/v1", "") + "/metrics",
            output_json=output_json,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
