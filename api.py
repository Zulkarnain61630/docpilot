import uuid

from fastapi import FastAPI
from pydantic import BaseModel

import agent
from agent import run_agent

agent.confirm_with_user = lambda question: False

app = FastAPI(title="DocPilot")
sessions = {}


class Question(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question):
    session_id = body.session_id or str(uuid.uuid4())
    messages = sessions.setdefault(session_id, [])
    try:
        answer = run_agent(body.question, messages)
    except Exception as e:
        answer = f"Sorry, something went wrong ({type(e).__name__})."
    return {"session_id": session_id, "answer": answer}