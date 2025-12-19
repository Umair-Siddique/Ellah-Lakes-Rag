from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retriever.common_retriever import search_and_generate
from csv_agent.csv_common_retriever import (
    query_common_with_source as query_csv_router,
)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []


st.set_page_config(page_title="Financial RAG Retriever", layout="centered")
st.title("Financial RAG Retriever")
st.caption("Ask a question — the app will show which namespace it retrieves from.")

# Mode selector: default to Ellah Lakes retriever; optional CSV chat.
mode = st.sidebar.radio(
    "Retriever source",
    options=["Ellah Lakes RAG", "CSV (Monetary/GDP)"],
    index=0,
)

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
        if mode == "Ellah Lakes RAG":
            with st.spinner("Retrieving + reranking + generating…"):
                try:
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

            final_category = category
            final_response = full_response

        else:
            # CSV router path
            with st.spinner("Routing to CSV retriever…"):
                try:
                    answer, source_tag = query_csv_router(prompt)
                except Exception as exc:
                    st.error(f"CSV retriever error: {exc}")
                    st.stop()

            # Map source tag to friendly label
            source_label = (
                "Monetary aggregates CSV (1960–1980)"
                if source_tag == "monetary"
                else "GDP-by-sector CSV (1960+)"
            )
            st.markdown(f"Retrieving from: `{source_label}`")

            # Stream the final answer by chunks to mimic streaming.
            placeholder = st.empty()
            final_response = ""
            for chunk in answer.split():
                final_response += (chunk + " ")
                placeholder.markdown(final_response)

            final_category = source_label

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"Retrieving from: `{final_category}`\n\n{final_response}",
        }
    )
