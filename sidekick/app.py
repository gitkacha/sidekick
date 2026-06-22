"""
app.py
Gradio UI for the Sidekick Personal Co-worker.

Run with:
    python app.py

Dependencies:
    pip install gradio
    + everything listed in sidekick.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import gradio as gr
from langchain_core.messages import SystemMessage

from sidekick import (
    build_clarifier_graph,
    build_sidekick_graph,
    QSTNS,
    setup,
)

# ---------------------------------------------------------------------------
# Run setup once on import
# ---------------------------------------------------------------------------
setup()

# sidekick_graph = build_sidekick_graph()
# clarifier_graph = build_clarifier_graph()


# ---------------------------------------------------------------------------
# Gradio helpers
# ---------------------------------------------------------------------------
async def setup_combined() -> dict[str, Any]:
    return {
        "phase": "clarify",
        "count": 0,
        "last_q": [],
        "clarifications": [],
        "original_request": None,
        "original_success_criteria": None,
        "answers": [],
    }


async def process_combined(
    grState: dict[str, Any],
    message: str,
    success_criteria: str,
    chatbot: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """
    Orchestrate the two phases:
      1. CLARIFY – turn-based clarifier (3 questions, one per turn).
      2. WORK    – main Sidekick worker/evaluator loop.
    """
    user_msg = {"role": "user", "content": message}
    cfg = {"configurable": {"thread_id": "t2"}}

    # ------------------------------------------------------------------
    # PHASE 1 — CLARIFY
    # ------------------------------------------------------------------
    if grState.get("phase") == "clarify":
        # Capture the original request on the very first turn
        if grState.get("original_request") is None:
            grState["original_request"] = message
            grState["original_success_criteria"] = success_criteria
        else:
            # Record the user's answer to the previously asked question
            grState["answers"].append(message)

        count_before = grState["count"]

        result = await clarifier_graph.ainvoke(
            {"messages": [user_msg], "asked": grState["last_q"]},
            config=cfg,
        )

        assistant_msg = result["messages"][-1]
        reply = {"role": "assistant", "content": assistant_msg.content}

        # Sync checkpointed state
        asked_so_far = result.get("asked", grState["last_q"])
        grState["last_q"] = asked_so_far
        grState["count"] = len(grState["last_q"])

        # Prefix while we are still asking questions
        if count_before < len(QSTNS):
            reply["content"] = f"Clarifying Question {grState['count']}:\n{reply['content']}"

        # Transition once all 3 questions are on record
        if grState["count"] >= len(QSTNS):
            grState["phase"] = "work"
            grState["clarifications"] = [
                f"Q: {q}\nA: {a}"
                for q, a in zip(grState["last_q"], grState.get("answers", []))
            ]

        return chatbot + [user_msg, reply], grState

    # ------------------------------------------------------------------
    # PHASE 2 — WORK
    # ------------------------------------------------------------------
    system_context = (
        f"User clarifications:\n"
        + "\n".join(grState.get("clarifications", []))
        + f"\n\nOriginal request: {grState['original_request']}\n"
        f"Success criteria: {grState.get('original_success_criteria', '')}"
    )

    state = {
        "messages": [SystemMessage(content=system_context), user_msg],
        "success_criteria": success_criteria or grState.get("original_success_criteria", ""),
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False,
        "iteration_count": 0,
    }

    result = await sidekick_graph.ainvoke(state, config=cfg)

    user = {"role": "user", "content": message}
    reply = {"role": "assistant", "content": result["messages"][-2].content}
    feedback = {"role": "assistant", "content": result["messages"][-1].content}

    return chatbot + [user, reply, feedback], grState


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Sidekick", theme=gr.themes.Ocean()) as combined_ui:
    gr.Markdown("## Sidekick Personal Co-Worker (with Pre-Task Clarifier)")

    grState = gr.State(
        {
            "phase": "clarify",
            "count": 0,
            "last_q": [],
            "clarifications": [],
            "original_request": None,
            "original_success_criteria": None,
            "answers": [],
        }
    )

    with gr.Row():
        chatbot = gr.Chatbot(
            label="Sidekick", height=300, type="messages", layout="bubble"
        )

    with gr.Group():
        with gr.Row():
            message = gr.Textbox(
                show_label=False, placeholder="Your request to the Sidekick"
            )
        with gr.Row():
            success_criteria = gr.Textbox(
                show_label=False, placeholder="What are your success criteria?"
            )

    with gr.Row():
        go_button = gr.Button("Go!", variant="primary")

    # Wiring
    combined_ui.load(setup_combined, [], [grState])
    message.submit(
        process_combined,
        [grState, message, success_criteria, chatbot],
        [chatbot, grState],
    )
    success_criteria.submit(
        process_combined,
        [grState, message, success_criteria, chatbot],
        [chatbot, grState],
    )
    go_button.click(
        process_combined,
        [grState, message, success_criteria, chatbot],
        [chatbot, grState],
    )

if __name__ == "__main__":
    combined_ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
