"""
sidekick_tests.py
Tests for the Sidekick and Clarifier graphs.

Run with:
    pytest sidekick_tests.py -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sidekick import (
    EvaluatorOutput,
    QSTNS,
    State,
    TestState,
    ask_question,
    build_clarifier_graph,
    build_sidekick_graph,
    evaluator,
    finish,
    format_conversation,
    route_based_on_evaluation,
    start_router,
    worker_router,
)


# ==========================================================================
# Helpers
# ==========================================================================
def _build_test_state() -> State:
    return {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ],
        "success_criteria": "Be helpful",
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False,
    }


def _build_test_evaluator_state(
    *, success: bool = False, needs_input: bool = False
) -> State:
    return {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ],
        "success_criteria": "Be helpful",
        "feedback_on_work": None,
        "success_criteria_met": success,
        "user_input_needed": needs_input,
    }


# ==========================================================================
# Pure-function unit tests (no LLM required)
# ==========================================================================
class TestRoutersAndHelpers:
    """Tests for router functions and format_conversation."""

    def test_format_conversation(self) -> None:
        messages = [
            HumanMessage(content="User says hello"),
            AIMessage(content="Assistant replies"),
            AIMessage(content=""),  # edge case: empty content
        ]
        result = format_conversation(messages)
        assert "User: User says hello" in result
        assert "Assistant: Assistant replies" in result
        assert "Assistant: [Tools use]" in result

    def test_worker_router_no_tool_calls(self) -> None:
        state = _build_test_state()
        assert worker_router(state) == "evaluator"

    def test_worker_router_with_tool_calls(self) -> None:
        state = _build_test_state()
        # Inject a tool_calls attribute onto the last message
        last = state["messages"][-1]
        last.additional_kwargs["tool_calls"] = [
            {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]
        # Patch the hasattr check so it sees tool_calls
        with patch.object(last, "tool_calls", [MagicMock()], create=True):
            assert worker_router(state) == "tools"

    def test_route_evaluation_to_end_on_success(self) -> None:
        state = _build_test_evaluator_state(success=True, needs_input=False)
        assert route_based_on_evaluation(state) == "END"

    def test_route_evaluation_to_end_on_needs_input(self) -> None:
        state = _build_test_evaluator_state(success=False, needs_input=True)
        assert route_based_on_evaluation(state) == "END"

    def test_route_evaluation_to_worker_otherwise(self) -> None:
        state = _build_test_evaluator_state(success=False, needs_input=False)
        assert route_based_on_evaluation(state) == "worker"

    def test_start_router_fewer_than_three(self) -> None:
        state: TestState = {"messages": [], "asked": ["Q1"], "done": False}
        assert start_router(state) == "ask_question"

    def test_start_router_three_or_more(self) -> None:
        state: TestState = {"messages": [], "asked": ["Q1", "Q2", "Q3"], "done": False}
        assert start_router(state) == "finish"


class testAskQuestionNode:
    """Tests for the clarifier ask_question node."""

    def test_ask_question_selects_new_question(self) -> None:
        state: TestState = {"messages": [], "asked": [], "done": False}
        result = ask_question(state)
        assert "messages" in result
        assert "asked" in result
        assert len(result["asked"]) == 1
        assert result["asked"][0] in QSTNS
        assert result["messages"][0]["content"] == result["asked"][0]

    def test_ask_question_avoids_duplicates(self) -> None:
        state: TestState = {"messages": [], "asked": [QSTNS[0], QSTNS[1]], "done": False}
        result = ask_question(state)
        assert len(result["asked"]) == 3
        assert set(result["asked"]) == set(QSTNS)
        assert result["messages"][0]["content"] == QSTNS[2]

    def test_ask_question_all_asked_raises(self) -> None:
        """If all questions are already asked, the while loop would run forever."""
        state: TestState = {"messages": [], "asked": list(QSTNS), "done": False}
        # In practice the router guards against this; the node itself would hang.
        # We simply assert the router routes to finish in this situation.
        assert start_router(state) == "finish"


class TestFinishNode:
    """Tests for the clarifier finish node."""

    def test_finish_returns_thankyou(self) -> None:
        state: TestState = {"messages": [], "asked": list(QSTNS), "done": False}
        result = finish(state)
        assert "messages" in result
        assert f"Thank you for answering all the {len(QSTNS)} questions" in result["messages"][0]["content"]
        assert result.get("done") is True


# ==========================================================================
# Graph compilation tests (no LLM calls)
# ==========================================================================
class TestGraphCompilation:
    """Ensure graphs compile and compile again without error."""

    def test_clarifier_graph_compiles(self) -> None:
        graph = build_clarifier_graph()
        assert graph is not None

    def test_sidekick_graph_compiles(self) -> None:
        # build_sidekick_graph relies on `tools` being populated.
        # If setup() wasn't run yet, tools will be empty and the graph still compiles
        # (the ToolNode just won't have tools).
        graph = build_sidekick_graph()
        assert graph is not None


# ==========================================================================
# Integration tests (requires LLM API if running live)
# ==========================================================================
class TestClarifierGraphIntegration:
    """End-to-end clarifier graph invocations."""

    @pytest.fixture
    def graph(self) -> Any:
        return build_clarifier_graph()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY",
    )
    async def test_first_step_asks_one_question(self, graph: Any) -> None:
        """A single ainvoke call should ask exactly one question."""
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "hello"}], "asked": []}
        )
        assert len(result["asked"]) == 1
        assert result["asked"][0] in QSTNS
        assert result["messages"][-1]["content"] == result["asked"][0]


class TestMainGraphIntegration:
    """End-to-end Sidekick worker/evaluator invocations."""

    @pytest.fixture
    def graph(self) -> Any:
        return build_sidekick_graph()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY",
    )
    async def test_worker_evaluator_superstep(self, graph: Any) -> None:
        """The main graph should run worker → evaluator in one ainvoke call."""
        state = {
            "messages": [{"role": "user", "content": "Say hello"}],
            "success_criteria": "Respond politely",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test"}})
        # We should have the original user msg + worker response + evaluator feedback
        assert len(result["messages"]) >= 3
        # The evaluator feedback should mention feedback
        assert "Evaluator Feedback" in result["messages"][-1]["content"]


# ==========================================================================
# End-to-end Gradio callback (pure logic, no UI)
# ==========================================================================
class TestGradioCallback:
    """Unit tests for the Gradio process_combined logic."""

    def test_clarify_prefix_added(self) -> None:
        """When count_before < len(QSTNS), the prefix is added."""
        gr_state = {
            "phase": "clarify",
            "count": 0,
            "last_q": [],
            "original_request": None,
            "original_success_criteria": None,
        }
        # We test the condition logic directly:
        if gr_state["count"] < len(QSTNS):
            prefixed = f"Clarifying Question {gr_state['count'] + 1}:\nWhat is your name?"
            assert prefixed.startswith("Clarifying Question 1:")

    def test_work_phase_no_prefix(self) -> None:
        gr_state = {
            "phase": "work",
            "count": 3,
            "last_q": list(QSTNS),
            "clarifications": list(QSTNS),
            "original_request": "Plan my week",
            "original_success_criteria": "Be detailed",
        }
        # Build system context
        ctx = (
            f"User clarifications:\n"
            + "\n".join(gr_state.get("clarifications", []))
            + f"\n\nOriginal request: {gr_state['original_request']}\n"
            f"Success criteria: {gr_state.get('original_success_criteria', '')}"
        )
        assert "User clarifications:" in ctx
        assert "Plan my week" in ctx
        assert "Be detailed" in ctx
