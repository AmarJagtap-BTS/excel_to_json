import os

class Config:
    """Configuration for Azure OpenAI and RAG system"""
    
    # Azure OpenAI Configuration - Use environment variables
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    # Deployment names
    EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "gpt-4o-mini")
    
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
