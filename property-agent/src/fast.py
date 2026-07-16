"""FastAPI app exposing the property agent as a /chat endpoint for the Streamlit chat UI."""
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from main import agent_executor, extract_content

app = FastAPI(title="Berlin Property Agent Chat API")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Sends one user message through the agent and returns its reply.

    Args:
        request: user message plus a thread_id so concurrent chat sessions
            (e.g. separate Streamlit browser tabs) don't share agent memory.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    result = agent_executor.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    reply = extract_content(result["messages"][-1])
    return ChatResponse(reply=reply)
