"""Agentic RAG — multi-turn tool-use agent.

The agent receives a question, decides which tools to call, observes results,
reasons about them, and iterates until confident in a final answer or until it
reaches the tool-call budget.

The system prompt, tool definitions, tool executor, client, and model are all
injected so the loop stays independent of any particular configuration.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 8


@dataclass
class ToolCall:
    """Record of a single tool call made by the agent."""
    tool_name: str
    tool_input: dict
    tool_output: dict
    timestamp: float = 0.0


@dataclass
class AgentTrace:
    """Complete trace of an agent's execution on a single question."""
    question: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    total_tool_calls: int = 0
    stop_reason: str = ""
    total_time_s: float = 0.0
    error: str = ""


def run_agent(
    question: str,
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    tool_definitions: list[dict],
    execute_tool: Callable[[str, dict], dict],
    max_tokens: int = 2048,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> AgentTrace:
    """Run the agentic loop on a single question.

    The agent calls tools iteratively until it produces a final text answer or
    hits the maximum tool-call budget.

    Args:
        question: The user's natural-language question.
        client: Anthropic client instance.
        model: Model identifier string.
        system_prompt: The agent system prompt.
        tool_definitions: Tool schemas passed to the model.
        execute_tool: Callable (tool_name, tool_input) -> tool_output dict.
        max_tokens: Per-call generation budget.
        max_tool_calls: Maximum number of tool calls before forcing an answer.

    Returns:
        AgentTrace with the full execution history.
    """
    trace = AgentTrace(question=question)
    start_time = time.time()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    tool_call_count = 0

    try:
        while tool_call_count < max_tool_calls:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tool_definitions,
                messages=messages,
            )

            assistant_content = response.content
            has_tool_use = any(block.type == "tool_use" for block in assistant_content)

            if not has_tool_use:
                text_blocks = [block.text for block in assistant_content if block.type == "text"]
                trace.final_answer = "\n".join(text_blocks).strip()
                trace.stop_reason = "final_answer"
                break

            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_call_count += 1
                tool_start = time.time()
                tool_output = execute_tool(block.name, block.input)
                tool_elapsed = time.time() - tool_start

                trace.tool_calls.append(ToolCall(
                    tool_name=block.name,
                    tool_input=block.input,
                    tool_output=tool_output,
                    timestamp=tool_elapsed,
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(tool_output, ensure_ascii=False, default=str),
                })

                if tool_call_count >= max_tool_calls:
                    break

            messages.append({"role": "user", "content": tool_results})

        else:
            # Budget exhausted — request a final answer from what was gathered.
            messages.append({
                "role": "user",
                "content": "You have used all available tool calls. Based on the information gathered so far, provide your final answer now.",
            })
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tool_definitions,
                messages=messages,
            )
            text_blocks = [block.text for block in response.content if block.type == "text"]
            trace.final_answer = "\n".join(text_blocks).strip()
            trace.stop_reason = "budget_exhausted"

    except Exception as e:
        logger.error("Agent error on question '%s': %s", question[:50], e)
        trace.error = str(e)
        trace.stop_reason = "error"

    trace.total_tool_calls = tool_call_count
    trace.total_time_s = time.time() - start_time
    return trace


def trace_to_dict(trace: AgentTrace) -> dict:
    """Convert an AgentTrace to a serialisable dictionary."""
    return {
        "question": trace.question,
        "final_answer": trace.final_answer,
        "total_tool_calls": trace.total_tool_calls,
        "stop_reason": trace.stop_reason,
        "total_time_s": round(trace.total_time_s, 2),
        "error": trace.error,
        "tool_calls": [
            {
                "tool_name": tc.tool_name,
                "tool_input": tc.tool_input,
                "tool_output": tc.tool_output,
                "duration_s": round(tc.timestamp, 3),
            }
            for tc in trace.tool_calls
        ],
    }
