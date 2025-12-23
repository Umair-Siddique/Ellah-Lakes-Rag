from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retriever.common_retriever import search_and_generate_with_tools
from csv_agent.csv_common_retriever import query_with_tools as query_csv_router


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    # Used to reduce cold-start latency on Streamlit Cloud by skipping reranking
    # for the first couple of RAG queries in a session.
    if "rag_query_count" not in st.session_state:
        st.session_state.rag_query_count = 0


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
            # Track how many RAG queries have been run in this Streamlit session.
            st.session_state.rag_query_count += 1
            enable_rerank = st.session_state.rag_query_count > 2

            spinner_label = (
                "Retrieving + reranking + generating…"
                if enable_rerank
                else "Retrieving + generating… (rerank warm-up skipped)"
            )

            with st.spinner(spinner_label):
                try:
                    last_messages = st.session_state.messages[-5:] if len(st.session_state.messages) > 0 else []
                    
                    # Use tool calling for intelligent routing
                    result = search_and_generate_with_tools(
                        query=prompt,
                        top_k=10,
                        rerank_top_n=5,
                        enable_rerank=enable_rerank,
                        chat_history=last_messages
                    )
                    
                except Exception as exc:
                    st.error(f"Retriever error: {exc}")
                    st.stop()

            category = result.get("category", "unknown")
            
            # Show which tool was used
            if result.get("tool_calls"):
                tool_info = result["tool_calls"][0]
                tool_name = tool_info['tool'].replace('_tool', '').replace('_', ' ').title()
                st.caption(f"🔧 Tool: `{tool_name}`")
            
            st.markdown(f"Namespace: `{category}`")

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
            # CSV router path - uses tool calling
            with st.spinner("Querying CSV data…"):
                try:
                    answer, source_tag = query_csv_router(prompt)
                except Exception as exc:
                    st.error(f"CSV retriever error: {exc}")
                    st.stop()

            # Map source tag to friendly label and tool name
            if source_tag == "monetary":
                source_label = "Monetary Aggregates (1960–1980)"
                tool_name = "Query Monetary Aggregates Tool"
            elif source_tag == "gdp":
                source_label = "GDP By Sector (1960+)"
                tool_name = "Query Gdp By Sector Tool"
            else:
                source_label = "CSV Data"
                tool_name = "Unknown Tool"
            
            st.caption(f"🔧 Tool: `{tool_name}`")
            st.markdown(f"Dataset: `{source_label}`")

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
            "content": f"{final_response}",
        }
    )
