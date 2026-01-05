import os

# Try to import streamlit for secrets support
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def get_config_value(key, default=None):
    """
    Get configuration value from multiple sources in order of priority:
    1. Environment variables
    2. Streamlit secrets (if available)
    3. Default value
    """
    # Check environment variables first
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value
    
    # Check Streamlit secrets if available
    if HAS_STREAMLIT:
        try:
            return st.secrets.get(key, default)
        except Exception:
            pass
    
    return default


class Config:
    """Configuration for Azure OpenAI and RAG system
    
    Configuration can be set via:
    1. Environment variables (highest priority)
    2. Streamlit secrets (.streamlit/secrets.toml)
    3. Default values (lowest priority)
    """
    
    # Azure OpenAI Configuration
    # IMPORTANT: Set these in .streamlit/secrets.toml (local) or Streamlit Cloud Secrets (production)
    # DO NOT hardcode credentials here!
    AZURE_OPENAI_ENDPOINT = get_config_value(
        "AZURE_OPENAI_ENDPOINT",
        None  # No default - must be provided in secrets
    )
    AZURE_OPENAI_API_KEY = get_config_value(
        "AZURE_OPENAI_API_KEY",
        None  # No default - must be provided in secrets
    )
    AZURE_OPENAI_API_VERSION = get_config_value(
        "AZURE_OPENAI_API_VERSION",
        "2025-01-01-preview"
    )
    
    # Deployment names
    EMBEDDING_DEPLOYMENT = get_config_value("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    CHAT_DEPLOYMENT = get_config_value("CHAT_DEPLOYMENT", "gpt-4o-mini")
    
    # Embedding configuration
    EMBEDDING_DIMENSION = int(get_config_value("EMBEDDING_DIMENSION", "3072"))
    
    # Similarity metric configuration
    # Options: "cosine" (normalized dot product, range 0-1) or "semantic" (dot product similarity, unnormalized)
    SIMILARITY_METRIC = get_config_value("SIMILARITY_METRIC", "cosine")
    
    # Search configuration
    TOP_K_RESULTS = int(get_config_value("TOP_K_RESULTS", "10"))
    
    # Scoring configuration - Two main scores
    SEMANTIC_WEIGHT = float(get_config_value("SEMANTIC_WEIGHT", "0.7"))
    SIMILARITY_WEIGHT = float(get_config_value("SIMILARITY_WEIGHT", "0.3"))
    
    # RAG configuration
    MAX_CONTEXT_LENGTH = int(get_config_value("MAX_CONTEXT_LENGTH", "4000"))
    TEMPERATURE = float(get_config_value("TEMPERATURE", "0"))
    MAX_TOKENS = int(get_config_value("MAX_TOKENS", "1000"))
    QUERY_REFRAMING_TIMEOUT = float(get_config_value("QUERY_REFRAMING_TIMEOUT", "10.0"))
    
    # Data paths
    PRODUCTS_JSON_PATH = "../../output/products.json"
    EMBEDDINGS_CACHE_PATH = "data/embeddings.pkl"
    VECTOR_STORE_PATH = "data/vector_store.faiss"
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is set"""
        if not cls.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT not set in configuration")
        if not cls.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY not set in configuration")
        return True
