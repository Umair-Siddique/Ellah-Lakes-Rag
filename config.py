import os

# Always load .env for local development first
from dotenv import load_dotenv
load_dotenv()

# Check if running in Streamlit Cloud (not just if streamlit is installed)
_use_streamlit_secrets = False
try:
    import streamlit as st
    # Only use streamlit secrets if secrets are actually available
    if hasattr(st, 'secrets') and len(st.secrets) > 0:
        _use_streamlit_secrets = True
except (ImportError, FileNotFoundError, RuntimeError):
    pass


def _get_config_value(key: str, default=None):
    """
    Get configuration value from Streamlit secrets (cloud) or environment variables (local).
    Priority: Streamlit secrets > Environment variables > Default
    """
    if _use_streamlit_secrets:
        try:
            # Try Streamlit secrets first
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    
    # Fallback to environment variable
    return os.getenv(key, default)


class Config:
    OPENAI_API_KEY = _get_config_value("OPENAI_API_KEY")
    PINECONE_API_KEY = _get_config_value("PINECONE_API_KEY")
    INDEX_NAME = _get_config_value("INDEX_NAME") or _get_config_value("PINECONE_INDEX_NAME")
    COHERE_API_KEY = _get_config_value("COHERE_API_KEY")
    PINECONE_ENVIRONMENT = _get_config_value("PINECONE_ENVIRONMENT", "us-east-1")


# Debug: Print config loading status on import (only in development)
if os.getenv("DEBUG_CONFIG"):
    print(f"[Config Debug] OPENAI_API_KEY: {'✓' if Config.OPENAI_API_KEY else '✗'}")
    print(f"[Config Debug] COHERE_API_KEY: {'✓' if Config.COHERE_API_KEY else '✗'}")
    print(f"[Config Debug] PINECONE_API_KEY: {'✓' if Config.PINECONE_API_KEY else '✗'}")
    print(f"[Config Debug] Using Streamlit secrets: {_use_streamlit_secrets}")

