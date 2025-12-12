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

# Custom CSS for ChatGPT-like UI
st.markdown("""
    <style>
    /* Set main container to fixed height */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
        max-width: 100%;
    }
    
    /* Message styling */
    .user-message {
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        background-color: #f7f7f8;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        max-width: 80%;
    }
    
    /* Chat messages container - scrollable with fixed height */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        height: calc(100vh - 320px);
        min-height: 400px;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Custom scrollbar for chat container */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Input area styling */
    .stTextInput {
        max-width: 900px;
        margin: 0 auto;
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem 0;
        z-index: 100;
    }
    
    /* Ensure text input takes full width */
    .stTextInput > div {
        width: 100%;
    }
    </style>
    <script>
    // Auto-scroll chat container to bottom
    function scrollToBottom() {
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    // Run on page load
    window.addEventListener('load', scrollToBottom);
    
    // Run after any changes (for Streamlit updates)
    const observer = new MutationObserver(scrollToBottom);
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
    </script>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ''

# Header (fixed at top)
st.title("📊 Financial RAG System")
st.markdown("### Intelligent Document Retrieval & Analysis")
st.markdown("---")

# Default settings (no longer exposed in UI)
top_k = 10
rerank_top_n = 5
force_category = None

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
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_query = ''
        st.rerun()

# Main content area - Chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
if st.session_state.history:
    for item in st.session_state.history:
        # User message
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong><br>
            {item['query']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="assistant-message">
            <strong>Assistant:</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(item['answer'])
        st.markdown("")
else:
    # Welcome message when no history
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem; color: #666;'>
        <h2>👋 Welcome to Financial RAG System</h2>
        <p style='font-size: 1.1rem; margin-top: 1rem;'>
            Ask me anything about financial statements, corporate disclosures, or director dealings.
        </p>
        <p style='font-size: 0.95rem; margin-top: 1rem; color: #888;'>
            I'll search through the documents and provide you with accurate answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input area at the bottom - fixed
st.markdown("---")
query = st.text_input(
    "Message Financial RAG",
    placeholder="Ask me anything about financial statements, corporate disclosures, or director dealings...",
    key="query_input",
    label_visibility="collapsed"
)

if query and query != st.session_state.get('last_query', ''):
    # Store the current query to prevent reprocessing
    st.session_state.last_query = query
    
    # Validate API keys
    if not Config.OPENAI_API_KEY or not Config.COHERE_API_KEY or not Config.PINECONE_API_KEY:
        st.error("❌ Missing API keys! Please set OPENAI_API_KEY, COHERE_API_KEY, and PINECONE_API_KEY in your .env file.")
    else:
        try:
            # Show loading spinner
            with st.spinner("Thinking..."):
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
            
            # User message
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong><br>
                {query}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message header
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Assistant:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display answer
            if not result['results']:
                answer = "I couldn't find relevant documents to answer your query. Please try rephrasing your question."
                st.markdown(answer)
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
                answer = full_response
            
            # Store in history
            st.session_state.history.append({
                'query': query,
                'answer': answer,
                'category': category_display,
                'confidence': confidence,
                'num_docs': len(result['results'])
            })
            
            # Rerun to show the new message in chat
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem;'>
        <p>Financial RAG System | Powered by OpenAI, Cohere & Pinecone</p>
    </div>
    """,
    unsafe_allow_html=True
)

