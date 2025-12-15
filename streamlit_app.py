from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retriever.common_retriever import search_and_generate


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []


def _recent_history_for_retriever(max_messages: int = 5) -> List[Dict[str, Any]]:
    """
    Return up to the last `max_messages` messages BEFORE the current user prompt.

    We also lightly sanitize assistant messages because this UI stores a prefix line
    ("Retrieving from namespace: ...") inside the assistant content.
    """
    history = st.session_state.messages[:-1]  # exclude current user prompt (already appended)
    recent = history[-max_messages:] if max_messages > 0 else history

    cleaned: List[Dict[str, Any]] = []
    for msg in recent:
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        if role == "assistant":
            lines = [ln for ln in content.splitlines() if ln.strip()]
            if lines and lines[0].lower().startswith("retrieving from namespace:"):
                # Drop the UI prefix line, keep the answer content.
                content = "\n".join(lines[1:]).strip()

        cleaned.append({"role": role, "content": content})

    return cleaned


st.set_page_config(page_title="Financial RAG Retriever", layout="centered")
st.title("Financial RAG Retriever")
st.caption("Ask a question — the app will show which namespace it retrieves from.")

_init_state()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your query…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + reranking + generating…"):
            try:
                result = search_and_generate(
                    query=prompt,
                    chat_history=_recent_history_for_retriever(max_messages=5),
                    max_history_messages=5,
                    top_k=10,
                    rerank_top_n=5,
                )
            except Exception as exc:
                st.error(f"Retriever error: {exc}")
                st.stop()

        category = result.get("category", "unknown")
        st.markdown(f"Retrieving from namespace: `{category}`")

        # Stream the LLM response (after reranking) into the chat bubble.
        response_stream = result.get("response_stream")
        placeholder = st.empty()
        full_response = ""

        if response_stream is None:
            full_response = "Error: response stream was not returned."
            placeholder.markdown(full_response)
        else:
            try:
                for chunk in response_stream:
                    full_response += chunk
                    placeholder.markdown(full_response)
            except Exception as exc:
                full_response += f"\n\nError generating response: {exc}"
                placeholder.markdown(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"Retrieving from namespace: `{category}`\n\n{full_response}",
        }
    )
