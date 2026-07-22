"""Streamlit chat UI for the Berlin property agent (calls the agent's /chat endpoint).

Thin HTTP client only: no LangChain/LangGraph/Gemini deps needed here, those
stay server-side in property-agent/src/fast.py.
"""
import os
import uuid

import requests
import streamlit as st

AGENT_API_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8002/chat")

st.title("Berlin Property Agent")
st.caption(
    "Chat with the Property Agent — it can tell the current date/time, "
    "look up coordinates for an address, and estimate sale, "
    "new-construction, and rental prices for Berlin apartments."
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Your message..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                AGENT_API_URL,
                json={"message": prompt, "thread_id": st.session_state.thread_id},
                timeout=30,
            )
            response.raise_for_status()
            reply = response.json()["reply"]
        except requests.exceptions.RequestException as e:
            reply = f"Could not reach the agent ({e}). Is the agent API running at {AGENT_API_URL}?"
        st.markdown(reply)

    st.session_state.messages.append(("assistant", reply))
