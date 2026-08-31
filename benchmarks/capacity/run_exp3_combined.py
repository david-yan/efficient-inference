#!/usr/bin/env python3
"""
Experiment 3: Combined Strict Prompting + Self-Correction Loop for tau-bench (Dual-Node Config).

Combines strict JSON system prompt instructions, token limit constraints (1024 max_tokens),
stop sequences, AND a 1-chance error feedback self-correction loop across 2 dedicated vLLM nodes.
"""

import argparse
import asyncio
import copy
import json
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

from tau2.agent.llm_agent import LLMAgent, LLMAgentState, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.registry import registry
from benchmark_tau_capacity import run_tau_capacity_suite
from run_exp1_strict_prompt import STRICT_AGENT_INSTRUCTION, STRICT_SYSTEM_PROMPT


class CombinedStrictSelfCorrectAgent(LLMAgent):
    """LLM Agent combining strict system prompting and 1-chance self-correction loop."""

    @property
    def system_prompt(self) -> str:
        return STRICT_SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=STRICT_AGENT_INSTRUCTION,
        )

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        try:
            assistant_msg = super()._generate_next_message(message, state)
            self._validate_json_formatting(assistant_msg)
            return assistant_msg
        except (json.JSONDecodeError, ValueError, AssertionError) as error:
            error_feedback = (
                f"ERROR: Your previous response failed JSON validation: {error}.\n"
                f"Please fix your response. Output ONLY a single raw JSON object matching the required schema."
            )
            feedback_user_msg = UserMessage(role="user", content=error_feedback)

            correction_state = copy.deepcopy(state)
            correction_state.messages.append(feedback_user_msg)

            corrected_msg = super()._generate_next_message(feedback_user_msg, correction_state)
            return corrected_msg

    def _validate_json_formatting(self, assistant_msg: AssistantMessage) -> None:
        if assistant_msg.content and assistant_msg.content.strip():
            text = assistant_msg.content.strip()
            if text.startswith("```"):
                raise ValueError("Response contains markdown codeblock indicators (```). Return raw JSON only.")
            if not (text.startswith("{") or text.startswith("[")):
                raise ValueError("Response text does not start with JSON object or array character.")
            json.loads(text)


def create_combined_agent(tools, domain_policy, **kwargs):
    return CombinedStrictSelfCorrectAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


registry.register_agent_factory(create_combined_agent, "combined_strict_self_correct_agent")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Combined Strict Prompt + Self-Correction Runner (Dual-Node)")
    parser.add_argument("--domain", default="airline,retail", help="Comma-separated domains")
    parser.add_argument("--agent-llm", default="openai/gemma-3-4b", help="Agent model endpoint name")
    parser.add_argument("--url-agent", default="http://vllm-service-gemma-3-4b:8000/v1", help="Instance 1 URL (Agent)")
    parser.add_argument("--url-user", default="http://vllm-service-gemma-3-4b-2:8000/v1", help="Instance 2 URL (User Simulator)")
    parser.add_argument("--tiers", nargs="+", type=int, default=[96], help="Concurrency tiers (default: 96)")
    parser.add_argument("--num-tasks", type=int, default=192, help="Number of tasks per tier")
    parser.add_argument("--task-timeout", type=float, default=120.0, help="Per-task timeout in seconds")
    parser.add_argument("--output-dir", default="benchmarks/capacity/results/exp3_combined")

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
    output_json = os.path.join(args.output_dir, "exp3_combined_results.json")

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
            agent_name="combined_strict_self_correct_agent",
            max_steps=30,
            task_timeout=args.task_timeout,
            metrics_url=args.url_agent.replace("/v1", "") + "/metrics",
            output_json=output_json,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
