from fastapi import FastAPI, Form, HTTPException
from typing import Optional
import json

app = FastAPI()

def validate_clarifier_state(state_json: Optional[str]) -> dict:
    """Validate that the state has the required structure."""
    if state_json is None:
        return {}
    try:
        state_dict = json.loads(state_json)
        if not isinstance(state_dict, dict):
            raise HTTPException(400, "State must be a JSON object")
        return state_dict
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in state parameter")
    except Exception as e:
        raise HTTPException(400, f"State validation error: {str(e)}")

@app.get("/answer", response_model=dict)
async def answer_question(question: str = Form(...), state: Optional[str] = Form(None)) -> dict:
    clarifier_state = {}
    if state is not None:
        try:
            clarifier_state = json.loads(state)
            # Validate state structure
            if not isinstance(clarifier_state, dict):
                raise HTTPException(400, "State must be a JSON object")
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON in state parameter")

    # Process the question with clarifier_state
    # Example: Add question to state tracking
    clarifier_state.setdefault('questions', []).append(question)

    return {
        "answer": "Processing completed",
        "state": clarifier_state,
        "status": "success"
    }