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
                # Get last 5 messages for context (excluding the current message)
                last_messages = st.session_state.messages[-5:] if len(st.session_state.messages) > 0 else []
                result = search_and_generate(
                    query=prompt,
                    top_k=10,
                    rerank_top_n=5,
                    chat_history=last_messages
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
