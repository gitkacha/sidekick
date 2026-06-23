"""
sidekick.py
Core agent logic for the Sidekick Personal Co-worker.

This module contains:
  - State definitions (State, TestState)
  - Pydantic schemas (EvaluatorOutput)
  - LangGraph node functions (worker, evaluator, ask_question, finish,…)
  - Graph builders for the main Sidekick loop and the clarifier sub-graph
  - Lazy setup() so imports stay side-effect free.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1.  Lazy imports that depend on sidekick_tools
# ---------------------------------------------------------------------------
_sidekick_tools_loaded = False
_playwright_tools = None
_other_tools = None


def _load_tools():
    """Import sidekick_tools lazily so this module can be imported without env."""
    global _sidekick_tools_loaded, _playwright_tools, _other_tools
    if _sidekick_tools_loaded:
        return
    from sidekick_tools import other_tools, playwright_tools  # type: ignore[import]

    import nest_asyncio

    nest_asyncio.apply()
    _playwright_tools = playwright_tools
    _other_tools = other_tools
    _sidekick_tools_loaded = True


# ---------------------------------------------------------------------------
# 2.  Pydantic / State definitions
# ---------------------------------------------------------------------------
class EvaluatorOutput(BaseModel):
    feedback: str = Field(
        description="Feedback on the assistant's response"
    )
    success_criteria_met: bool = Field(
        description="Whether the success criteria have been met"
    )
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    iteration_count: Optional[int]


class TestState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    asked: List[Any]
    done: bool


# ---------------------------------------------------------------------------
# 3.  Globals that are initialised in setup()
# ---------------------------------------------------------------------------
worker_llm_with_tools = None
evaluator_llm_with_output = None
tools: list[Any] = []
sidekick_graph: Any = None
clarifier_graph: Any = None

QSTNS = [
    "What is your name?",
    "What is your favourite colour?",
    "What is your current goal?",
]

MAX_WORKER_CALLS = 20

# ---------------------------------------------------------------------------
# 4.  setup()
# ---------------------------------------------------------------------------
def setup(model: str = "gpt-4o-mini") -> None:
    """
    Initialise LLMs, tools, and compile both graphs.
    Must be called before any graph invocation.
    """
    global worker_llm_with_tools, evaluator_llm_with_output, tools
    global sidekick_graph, clarifier_graph

    load_dotenv(override=True)

    # ------------------------------------------------------------------
    # Initialise tools
    # ------------------------------------------------------------------
    _load_tools()

    import asyncio

    # Playwright tools are async – create a one-off event loop to grab them
    loop = asyncio.get_event_loop()
    pw_tools, _, _ = loop.run_until_complete(_playwright_tools())  # type: ignore[misc]
    other = loop.run_until_complete(_other_tools())  # type: ignore[misc]
    tools = pw_tools + other

    # ------------------------------------------------------------------
    # Initialise LLMs – both with error handling
    # ------------------------------------------------------------------
    global _setup_error
    # Worker LLM
    try:
        worker_llm = ChatOpenAI(model=model)
        worker_llm_with_tools = worker_llm.bind_tools(tools)
    except Exception as e:
        worker_llm_with_tools = None
        _setup_error = str(e)

    # Evaluator LLM
    try:
        evaluator_llm = ChatOpenAI(model=model)
        evaluator_llm_with_output = evaluator_llm.with_structured_output(
            EvaluatorOutput
        )
    except Exception as e:
        evaluator_llm_with_output = None
        if _setup_error:
            _setup_error += f"\nEvaluator init failed: {e}"
        else:
            _setup_error = f"Evaluator init failed: {e}"

    # ------------------------------------------------------------------
    # Build main Sidekick graph
    # ------------------------------------------------------------------
    sidekick_graph = build_sidekick_graph()

    # ------------------------------------------------------------------
    # Build clarifier graph
    # ------------------------------------------------------------------
    clarifier_graph = build_clarifier_graph()


# ---------------------------------------------------------------------------
# 5.  Node functions – main Sidekick flow
# ---------------------------------------------------------------------------
def worker(state: State) -> Dict[str, Any]:
    system_message = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
This is the success criteria:
{state['success_criteria']}
You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""

    if state.get("feedback_on_work"):
        system_message += f"""
Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state['feedback_on_work']}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

    found_system_message = False
    messages = state["messages"]
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = system_message
            found_system_message = True

    if not found_system_message:
        messages = [SystemMessage(content=system_message)] + messages

    if worker_llm_with_tools is None:
        # Return a clear error message indicating LLM setup failure
        error_msg = globals().get('_setup_error', 'LLM not configured')
        return {"messages": [{"role": "assistant", "content": f"Setup error: {error_msg}"}]}
    response = worker_llm_with_tools.invoke(messages)
    return {"messages": [response]}


def worker_router(state: State) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "evaluator"


def format_conversation(messages: List[Any]) -> str:
    conversation = "Conversation history:\n\n"
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            text = message.content or "[Tools use]"
            conversation += f"Assistant: {text}\n"
    return conversation


def evaluator(state: State) -> Dict[str, Any]:
    last_response = state["messages"][-1].content

    system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
and whether more input is needed from the user."""

    user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

The entire conversation with the assistant, with the user's original request and all replies, is:
{format_conversation(state['messages'])}

The success criteria for this assignment is:
{state['success_criteria']}

And the final response from the Assistant that you are evaluating is:
{last_response}

Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.
"""
    if state.get("feedback_on_work"):
        user_message += (
            f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."
        )

    evaluator_messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ]

    if evaluator_llm_with_output is None:
        # Return a clear error message indicating LLM setup failure
        error_msg = globals().get('_setup_error', 'Evaluator LLM not configured')
        return {
            "messages": [{"role": "assistant", "content": f"Setup error: {error_msg}"}],
            "feedback_on_work": error_msg,
            "success_criteria_met": False,
            "user_input_needed": True,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    eval_result = evaluator_llm_with_output.invoke(evaluator_messages)
    return {
        "messages": [
            {"role": "assistant", "content": f"Evaluator Feedback on this answer: {eval_result.feedback}"}
        ],
        "feedback_on_work": eval_result.feedback,
        "success_criteria_met": eval_result.success_criteria_met,
        "user_input_needed": eval_result.user_input_needed,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def route_based_on_evaluation(state: State) -> str:
    if state["success_criteria_met"] or state["user_input_needed"]:
        return "END"
    if state.get("iteration_count", 0) >= MAX_WORKER_CALLS:
        return "END"
    return "worker"


# ---------------------------------------------------------------------------
# 6.  Node functions – clarifier
# ---------------------------------------------------------------------------
def ask_question(state: TestState) -> Dict[str, Any]:
    """Pick a random question that hasn't been asked yet."""
    import random

    while True:
        q = QSTNS[random.randint(0, len(QSTNS) - 1)]
        if q not in state["asked"]:
            break

    new_asked = state["asked"] + [q]
    return {
        "messages": [{"role": "assistant", "content": q}],
        "asked": new_asked,
    }


def finish(state: TestState) -> Dict[str, Any]:
    """All three questions have been asked – thank the user."""
    return {
        "messages": [
            {
                "role": "assistant",
                "content": f"Thank you for answering all the {len(QSTNS)} questions",
            }
        ],
        "done": True,
    }


def start_router(state: TestState) -> str:
    return "finish" if len(state["asked"]) >= len(QSTNS) else "ask_question"


# ---------------------------------------------------------------------------
# 7.  Graph builders
# ---------------------------------------------------------------------------
def build_sidekick_graph():
    """Compile the main Sidekick graph (worker → tools → evaluator → END|worker)."""
    graph_builder = StateGraph(State)
    graph_builder.add_node("worker", worker)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_node("evaluator", evaluator)

    graph_builder.add_conditional_edges(
        "worker", worker_router, {"tools": "tools", "evaluator": "evaluator"}
    )
    graph_builder.add_edge("tools", "worker")
    graph_builder.add_conditional_edges(
        "evaluator",
        route_based_on_evaluation,
        {"worker": "worker", "END": END},
    )
    graph_builder.add_edge(START, "worker")

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


def build_clarifier_graph():
    """Compile the turn-based clarifier graph (START → ask_question|finish → END).

    Each invocation asks exactly ONE question (or finishes if all 3 asked).
    The Gradio app drives the turn-by-turn flow and manages state via grState.
    """
    graph_builder = StateGraph(TestState)
    graph_builder.add_node("ask_question", ask_question)
    graph_builder.add_node("finish", finish)

    # Route from START based on how many questions have been asked
    graph_builder.add_conditional_edges(
        START,
        start_router,
        {"ask_question": "ask_question", "finish": "finish"},
    )
    # Both nodes go directly to END — no cycles, so one question per ainvoke
    graph_builder.add_edge("ask_question", END)
    graph_builder.add_edge("finish", END)

    # No checkpointer: app manages clarification state via grState
    return graph_builder.compile()


# ---------------------------------------------------------------------------
# 8.  Convenience helpers
# ---------------------------------------------------------------------------
def make_thread_id() -> str:
    return str(uuid.uuid4())
