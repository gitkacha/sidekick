# Systematic Debug Findings: Sidekick Pre-Task Clarifier
Critical Issue

Sidekick is skipping mandatory pre-task clarification questions, which violates its core design principle requiring 3 question-answer cycles before providing answers.

Root Cause

- Architectural State Handling: Questionnaire state isn't properly propagated between UI components
- Component Isolation: Subagent context not maintained during questioning

Resolution

Fixed in sidekick/sidekick.py and sidekick/app.py:

1. Removed the cyclic edge in `build_clarifier_graph()` - the graph now transitions directly to END after each question, preventing it from running all 3 questions in a single invocation
2. Removed the MemorySaver checkpointer - the Gradio app manages state via grState, so the checkpointer was causing state confusion
3. Updated `process_combined()` in app.py to track questions via `last_q` list length instead of a separate counter

The clarifier graph now correctly asks exactly ONE question per `ainvoke()` call, as expected by the turn-based UI flow.



Validation Steps

✅ All tests pass (16 passed, 2 skipped - skipped require OPENAI_API_KEY)

✅ Direct graph invocation test confirms:
- Turn 1: asks 1 question
- Turn 2: asks 1 question  
- Turn 3: asks 1 question
- Turn 4: returns done=True

The Sidekick now correctly enforces exactly 3 clarifying questions before moving to work phase.