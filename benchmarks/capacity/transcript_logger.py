#!/usr/bin/env python3
"""
Trajectory and Transcript Exporter for tau-bench Multi-Turn Capacity Benchmarks.

Exports complete simulation runs to:
1. Human-readable Markdown transcripts (.md)
2. Raw structured JSON trajectory data models (.json)
3. Concurrency tier catalogs (INDEX.md)
4. Timed-out task failure manifests (timed_out_tasks.json / .md)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional


def format_message_markdown(msg: Any, turn_number: int) -> str:
    """Formats an Assistant, User, or Tool message into readable markdown with metrics."""
    role = getattr(msg, "role", "unknown")
    content = getattr(msg, "content", "") or ""
    gen_time = getattr(msg, "generation_time_seconds", None)
    usage = getattr(msg, "usage", None) or {}
    comp_toks = usage.get("completion_tokens", 0)
    prompt_toks = usage.get("prompt_tokens", 0)

    perf_info = []
    if comp_toks:
        perf_info.append(f"{comp_toks} comp tokens")
    if prompt_toks:
        perf_info.append(f"{prompt_toks} prompt tokens")
    if gen_time:
        perf_info.append(f"{gen_time:.2f}s latency")

    perf_badge = f" *({', '.join(perf_info)})*" if perf_info else ""

    lines = []
    if role == "user":
        lines.append(f"### Turn {turn_number}: 👤 **User Simulator**{perf_badge}")
        lines.append(f"> {content.strip()}")
    elif role == "assistant":
        lines.append(f"### Turn {turn_number}: 🤖 **Agent Assistant**{perf_badge}")
        if content.strip():
            lines.append(f"{content.strip()}")
        # Check tool calls
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            lines.append("\n**Tool Calls:**")
            for tc in tool_calls:
                func_name = getattr(tc, "name", None) or getattr(getattr(tc, "function", None), "name", "tool")
                func_args = getattr(tc, "arguments", {}) or getattr(getattr(tc, "function", None), "arguments", {})
                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except Exception:
                        pass
                lines.append(f"- **`{func_name}`**:")
                lines.append("```json")
                lines.append(json.dumps(func_args, indent=2))
                lines.append("```")
    elif role == "tool":
        lines.append(f"### Turn {turn_number}: ⚙️ **Environment Tool Result**{perf_badge}")
        requestor = getattr(msg, "requestor", "assistant")
        name = getattr(msg, "name", "tool_result")
        lines.append(f"*Returned for `{name}` (requestor: `{requestor}`):*")
        try:
            parsed_content = json.loads(content) if isinstance(content, str) else content
            lines.append("```json")
            lines.append(json.dumps(parsed_content, indent=2))
            lines.append("```")
        except Exception:
            lines.append(f"```\n{content}\n```")
    else:
        lines.append(f"### Turn {turn_number}: 💬 **{role.capitalize()}**{perf_badge}")
        lines.append(f"{content}")

    return "\n".join(lines)


def export_simulation_transcript_markdown(sim: Any, task: Optional[Any], concurrency: int) -> str:
    """Generates a complete markdown document for a simulation trajectory."""
    task_id = getattr(sim, "task_id", getattr(task, "id", "unknown"))
    duration = getattr(sim, "duration", getattr(sim, "duration_seconds", 0.0))
    termination_reason = getattr(sim, "termination_reason", "unknown")
    if hasattr(termination_reason, "value"):
        termination_reason = termination_reason.value

    reward = 0.0
    if hasattr(sim, "reward_info") and sim.reward_info:
        reward = float(getattr(sim.reward_info, "reward", 0.0))

    is_success = (reward == 1.0)
    is_timeout = (termination_reason == "timeout")

    if is_success:
        status_badge = "🟢 **PASSED (Reward: 1.0)**"
    elif is_timeout:
        status_badge = "⏳ **TIMED OUT (Reward: 0.0)**"
    else:
        status_badge = f"🔴 **FAILED (Reward: {reward})**"

    scenario_instructions = ""
    domain = "unknown"
    if task and hasattr(task, "user_scenario") and task.user_scenario:
        instructions_obj = getattr(task.user_scenario, "instructions", None)
        domain = getattr(instructions_obj, "domain", "unknown") if instructions_obj else "unknown"
        if instructions_obj:
            scenario_instructions = getattr(instructions_obj, "task_instructions", "") or ""
            reason = getattr(instructions_obj, "reason_for_call", "")
            known = getattr(instructions_obj, "known_info", "")
            if reason:
                scenario_instructions += f"\n\n**Reason for Call:** {reason}"
            if known:
                scenario_instructions += f"\n\n**Known Info:** {known}"

    messages = getattr(sim, "messages", []) or []

    lines = [
        f"# Simulation Transcript: Task `{task_id}` ({domain.upper()})",
        "",
        "## Summary Metadata",
        f"- **Status**: {status_badge}",
        f"- **Task ID**: `{task_id}`",
        f"- **Domain**: `{domain}`",
        f"- **Concurrency Tier**: `C={concurrency}`",
        f"- **Termination Reason**: `{termination_reason}`",
        f"- **Total Duration**: `{duration:.2f}s`",
        f"- **Total Messages**: `{len(messages)}`",
        "",
    ]

    if scenario_instructions:
        lines.extend([
            "## User Scenario Instructions",
            "> " + scenario_instructions.replace("\n", "\n> "),
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Conversation Trajectory",
        "",
    ])

    for i, msg in enumerate(messages, 1):
        lines.append(format_message_markdown(msg, i))
        lines.append("")

    return "\n".join(lines)


def export_simulation_artifacts(sim: Any, task: Optional[Any], concurrency: int, base_dir: str) -> Dict[str, str]:
    """Saves both task transcript .md and trajectory .json under base_dir/tier_C{concurrency}/."""
    tier_dir = os.path.join(base_dir, f"tier_C{concurrency}")
    os.makedirs(tier_dir, exist_ok=True)

    task_id = str(getattr(sim, "task_id", getattr(task, "id", "unknown")))
    trial = getattr(sim, "trial", 1) or 1

    md_filename = f"task_{task_id}_trial_{trial}_transcript.md"
    json_filename = f"task_{task_id}_trial_{trial}_trajectory.json"

    md_path = os.path.join(tier_dir, md_filename)
    json_path = os.path.join(tier_dir, json_filename)

    # 1. Export Markdown Transcript
    md_content = export_simulation_transcript_markdown(sim, task, concurrency)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. Export Raw JSON Trajectory
    sim_data: Dict[str, Any] = {}
    if hasattr(sim, "model_dump"):
        sim_data = sim.model_dump()
    elif hasattr(sim, "dict"):
        sim_data = sim.dict()
    elif isinstance(sim, dict):
        sim_data = sim
    else:
        sim_data = {
            "task_id": task_id,
            "duration": getattr(sim, "duration", 0.0),
            "termination_reason": str(getattr(sim, "termination_reason", "")),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sim_data, f, indent=2, default=str)

    return {"transcript_md": md_path, "trajectory_json": json_path}


def generate_tier_index_markdown(
    sim_details: List[Dict[str, Any]], concurrency: int, tier_elapsed: float
) -> str:
    """Generates an INDEX.md summarizing all simulations in a concurrency tier."""
    lines = [
        f"# Concurrency Tier C={concurrency} Trajectory Catalog",
        "",
        f"- **Total Simulations**: {len(sim_details)}",
        f"- **Elapsed Wallclock Time**: {tier_elapsed:.2f}s",
        "",
        "| Task ID | Status | Reward | Duration | Turns | Agent Tok | User Tok | Termination Reason | Transcript Link |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for s in sim_details:
        tid = s.get("task_id", "unknown")
        success = s.get("success", False)
        reward = s.get("reward", 0.0)
        term_reason = s.get("termination_reason", "unknown")
        dur = s.get("duration_sec", 0.0)
        turns = s.get("turns", 0)
        a_tok = s.get("agent_tokens", 0)
        u_tok = s.get("user_tokens", 0)

        if success:
            status_str = "🟢 PASS"
        elif term_reason == "timeout":
            status_str = "⏳ TIMEOUT"
        else:
            status_str = "🔴 FAIL"

        link_str = f"[Transcript](task_{tid}_trial_1_transcript.md)"
        lines.append(
            f"| `{tid}` | {status_str} | `{reward}` | `{dur:.1f}s` | `{turns}` | `{a_tok}` | `{u_tok}` | `{term_reason}` | {link_str} |"
        )

    return "\n".join(lines)


def save_timed_out_manifest(
    timed_out_entries: List[Dict[str, Any]], output_dir: str, timeout_sec: float
) -> Dict[str, str]:
    """Writes timed_out_tasks.json and timed_out_tasks.md manifest."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "timed_out_tasks.json")
    md_path = os.path.join(output_dir, "timed_out_tasks.md")

    manifest_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timeout_threshold_sec": timeout_sec,
        "total_timed_out": len(timed_out_entries),
        "timed_out_samples": timed_out_entries,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    lines = [
        "# Timed-Out Tasks Manifest",
        "",
        f"- **Timestamp**: `{manifest_data['timestamp']}`",
        f"- **Timeout Threshold**: `{timeout_sec}s`",
        f"- **Total Timed-Out Tasks**: `{len(timed_out_entries)}`",
        "",
        "These tasks exceeded the execution deadline and were terminated to prevent straggler tail latency. They can be inspected and re-run independently using `rerun_timed_out_tasks.py`.",
        "",
        "| Tier | Task ID | Domain | Duration | Turns | Reason | Transcript |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in timed_out_entries:
        tier = item.get("concurrency_tier", "N/A")
        tid = item.get("task_id", "N/A")
        dom = item.get("domain", "N/A")
        dur = item.get("duration_sec", 0.0)
        turns = item.get("turns_completed", 0)
        reason = item.get("termination_reason", "timeout")
        rel_path = item.get("transcript_md", "")

        link = f"[View Transcript]({rel_path})" if rel_path else "N/A"
        lines.append(f"| `C={tier}` | `{tid}` | `{dom}` | `{dur:.1f}s` | `{turns}` | `{reason}` | {link} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json_path": json_path, "md_path": md_path}
