"""
Streamlit UI for Financial RAG System

To run this application:
    streamlit run streamlit_app.py

Make sure you have set the following environment variables in your .env file:
    - OPENAI_API_KEY
    - COHERE_API_KEY
    - PINECONE_API_KEY
    - INDEX_NAME
"""
import streamlit as st
import sys
from pathlib import Path

# Add root directory to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retriever.common_retriever import (
    _classify_query,
    search,
    _generate_streaming_response
)
from config import Config

# Page configuration
st.set_page_config(
    page_title="Financial RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.title("📊 Financial RAG System")
st.markdown("### Intelligent Document Retrieval & Analysis")
st.markdown("Ask questions about financial statements, corporate disclosures, or director dealings.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key status
    st.subheader("API Keys Status")
    if Config.OPENAI_API_KEY:
        st.success("✓ OpenAI API Key")
    else:
        st.error("✗ OpenAI API Key Missing")
    
    if Config.COHERE_API_KEY:
        st.success("✓ Cohere API Key")
    else:
        st.error("✗ Cohere API Key Missing")
    
    if Config.PINECONE_API_KEY:
        st.success("✓ Pinecone API Key")
    else:
        st.error("✗ Pinecone API Key Missing")
    
    st.divider()
    
    # Advanced settings
    st.subheader("Advanced Settings")
    
    top_k = st.slider("Initial Results (top_k)", min_value=5, max_value=20, value=10, step=1)
    rerank_top_n = st.slider("Reranked Results", min_value=3, max_value=10, value=5, step=1)
    
    force_category = st.selectbox(
        "Force Category (Optional)",
        ["Auto-detect", "financial_statements", "corporate_disclosure", "director_dealing"]
    )
    
    if force_category == "Auto-detect":
        force_category = None
    
    st.divider()
    
    # Examples
    st.subheader("📝 Example Queries")
    st.markdown("""
    **Financial Statements:**
    - What were Ellah Lakes PLC's total assets in 2013?
    - Show me the revenue trends
    
    **Corporate Disclosures:**
    - What board meetings were scheduled in 2023?
    - Tell me about recent corporate announcements
    
    **Director Dealings:**
    - Which directors sold shares last year?
    - Show insider trading activity
    """)
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Main content area
query = st.text_input(
    "Enter your query:",
    placeholder="e.g., What were Ellah Lakes PLC's total assets as at 31 January 2013?",
    key="query_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

if search_button and query:
    # Validate API keys
    if not Config.OPENAI_API_KEY or not Config.COHERE_API_KEY or not Config.PINECONE_API_KEY:
        st.error("❌ Missing API keys! Please set OPENAI_API_KEY, COHERE_API_KEY, and PINECONE_API_KEY in your .env file.")
    else:
        # Create containers for status updates
        status_container = st.container()
        answer_container = st.container()
        sources_container = st.container()
        
        with status_container:
            try:
                # Step 1: Classification
                status_placeholder = st.empty()
                status_placeholder.info("🔍 **Step 1/4:** Classifying query using LLM...")
                
                # Classify the query
                if force_category:
                    category = force_category
                    classification = {
                        "category": category,
                        "confidence": 1.0,
                        "reasoning": "Category was manually forced",
                    }
                else:
                    classification = _classify_query(query, model="gpt-4o-mini")
                    category = classification["category"]
                
                confidence = classification['confidence']
                reasoning = classification['reasoning']
                category_display = category.replace('_', ' ').title()
                
                status_placeholder.success(
                    f"✅ **Step 1/4:** Query classified as **{category_display}** "
                    f"(Confidence: {confidence:.2%})"
                )
                
                # Step 2: Searching
                status_placeholder2 = st.empty()
                status_placeholder2.info(f"📚 **Step 2/4:** Searching from **{category_display}** documents...")
                
                # Perform search with reranking
                result = search(
                    query=query,
                    top_k=top_k,
                    classification_model="gpt-4o-mini",
                    filter_model=None,
                    force_category=category,
                    enable_rerank=True,
                    rerank_top_n=rerank_top_n,
                    rerank_model="rerank-english-v3.0"
                )
                
                status_placeholder2.success(
                    f"✅ **Step 2/4:** Retrieved {len(result['results'])} documents from **{category_display}**"
                )
                
                # Step 3: Reranking (already done in search function)
                status_placeholder3 = st.empty()
                if result['results']:
                    status_placeholder3.info("🔄 **Step 3/4:** Reranking documents using Cohere...")
                    status_placeholder3.success(
                        f"✅ **Step 3/4:** Documents reranked (Top {len(result['results'])} selected)"
                    )
                else:
                    status_placeholder3.warning("⚠️ **Step 3/4:** No documents to rerank")
                
                # Step 4: Generating
                status_placeholder4 = st.empty()
                status_placeholder4.info("✨ **Step 4/4:** Generating answer using AI...")
                
                # Display answer with streaming
                with answer_container:
                    st.markdown("---")
                    st.subheader("💡 Answer")
                    
                    if not result['results']:
                        st.warning("No relevant documents found to answer your query. Please try rephrasing your question.")
                    else:
                        answer_placeholder = st.empty()
                        full_response = ""
                        
                        # Generate streaming response
                        response_stream = _generate_streaming_response(
                            query=query,
                            documents=result['results'],
                            category=category,
                            model="command-a-03-2025"
                        )
                        
                        # Stream the response
                        for chunk in response_stream:
                            full_response += chunk
                            answer_placeholder.markdown(full_response + "▌")
                        
                        # Final answer without cursor
                        answer_placeholder.markdown(full_response)
                        
                        status_placeholder4.success("✅ **Step 4/4:** Answer generated successfully!")
                        
                        # Store in history
                        st.session_state.history.append({
                            'query': query,
                            'answer': full_response,
                            'category': category_display,
                            'confidence': confidence,
                            'num_docs': len(result['results'])
                        })
                
                # Display source documents
                with sources_container:
                    st.markdown("---")
                    with st.expander(f"📄 View Source Documents ({len(result['results'])} documents)", expanded=False):
                        for i, (doc, formatted) in enumerate(zip(result['results'], result['formatted_results']), 1):
                            st.markdown(f"**Document {i}** (Score: {doc.get('score', 0):.4f})")
                            st.text(formatted)
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

elif search_button and not query:
    st.warning("⚠️ Please enter a query to search.")

# Display history
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Query History")
    
    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"{i}. {item['query'][:100]}...", expanded=False):
            st.markdown(f"**Category:** {item['category']}")
            st.markdown(f"**Confidence:** {item['confidence']:.2%}")
            st.markdown(f"**Documents Used:** {item['num_docs']}")
            st.markdown("**Answer:**")
            st.markdown(item['answer'])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Financial RAG System | Powered by OpenAI, Cohere & Pinecone</p>
    </div>
    """,
    unsafe_allow_html=True
)

