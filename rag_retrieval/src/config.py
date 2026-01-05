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
            # Try to access Streamlit secrets
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    
    # Fallback to environment variables
    return os.getenv(key, default)

class ConfigMeta(type):
    """Metaclass to make Config class attributes dynamic"""
    
    _cache = {}
    
    def __getattribute__(cls, name):
        # Dynamic config values
        config_map = {
            'AZURE_OPENAI_ENDPOINT': ('AZURE_OPENAI_ENDPOINT', ''),
            'AZURE_OPENAI_API_KEY': ('AZURE_OPENAI_API_KEY', ''),
            'AZURE_OPENAI_API_VERSION': ('AZURE_OPENAI_API_VERSION', '2025-01-01-preview'),
            'EMBEDDING_DEPLOYMENT': ('EMBEDDING_DEPLOYMENT', 'text-embedding-3-large'),
            'CHAT_DEPLOYMENT': ('CHAT_DEPLOYMENT', 'gpt-4o-mini'),
        }
        
        if name in config_map:
            cache_key = f'config_{name}'
            if cache_key not in cls._cache:
                key, default = config_map[name]
                cls._cache[cache_key] = get_config_value(key, default)
            return cls._cache[cache_key]
        
        return super().__getattribute__(name)

class Config(metaclass=ConfigMeta):
    """Configuration for Azure OpenAI and RAG system"""
    
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
        errors = []
        
        if not cls.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT not set")
        if not cls.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY not set")
        
        if errors:
            error_msg = "Configuration errors:\n- " + "\n- ".join(errors)
            error_msg += "\n\nPlease configure secrets in Streamlit Cloud:"
            error_msg += "\n1. Go to your app settings"
            error_msg += "\n2. Click 'Secrets' in the sidebar"
            error_msg += "\n3. Add the required configuration values"
            error_msg += "\n\nSee secrets.toml.example for the format."
            raise ValueError(error_msg)
        
        return True
