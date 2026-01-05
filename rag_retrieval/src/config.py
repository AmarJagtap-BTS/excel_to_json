import os

# Try to import Streamlit for secrets support
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def get_config_value(key, default=""):
    """Get configuration value from Streamlit secrets or environment variables"""
    # Priority: Streamlit secrets > Environment variables > Default
    if HAS_STREAMLIT:
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except (FileNotFoundError, KeyError):
            return os.getenv(key, default)
    return os.getenv(key, default)

class Config:
    """Configuration for Azure OpenAI and RAG system"""
    
    # Azure OpenAI Configuration - Use Streamlit secrets or environment variables
    AZURE_OPENAI_ENDPOINT = get_config_value("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = get_config_value("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = get_config_value("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    # Deployment names
    EMBEDDING_DEPLOYMENT = get_config_value("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    CHAT_DEPLOYMENT = get_config_value("CHAT_DEPLOYMENT", "gpt-4o-mini")
    
    # Embedding configuration
    EMBEDDING_DIMENSION = 3072  # text-embedding-3-large dimension
    
    SIMILARITY_METRIC = "cosine"  
    
    # Search configuration
    TOP_K_RESULTS = 10
    
    # Scoring configuration - Two main scores
    SEMANTIC_WEIGHT = 0.7
    SIMILARITY_WEIGHT = 0.3
    
    # RAG configuration
    MAX_CONTEXT_LENGTH = 4000
    TEMPERATURE = 0
    MAX_TOKENS = 1000
    QUERY_REFRAMING_TIMEOUT = 10.0  # Timeout for query reframing API calls
    
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
