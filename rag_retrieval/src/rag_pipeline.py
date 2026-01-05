import json
import os
from typing import List, Dict, Any, Optional

from openai import AzureOpenAI

from config import Config
from embeddings import EmbeddingsGenerator
from vector_store import VectorStore
from prompt_manager import PromptManager
from utils import setup_logging


# Setup logging
logger = setup_logging(__name__)


class RAGPipeline:
    """Complete RAG pipeline for product catalog search"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Initialized VectorStore instance
        """
        logger.info("Initializing RAG Pipeline")
        Config.validate()
        
        # Validate required credentials are set
        if not Config.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT not configured. Please set it in .streamlit/secrets.toml")
        if not Config.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY not configured. Please set it in .streamlit/secrets.toml")
        
        self.client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )
        self.embedding_model = Config.EMBEDDING_DEPLOYMENT
        self.chat_model = Config.CHAT_DEPLOYMENT
        self.vector_store = vector_store
        self.embeddings_generator = EmbeddingsGenerator()
        
        # Initialize prompt manager (singleton, loads prompts once)
        self.prompt_manager = PromptManager()
        
        # Build brand lookup for case-insensitive matching
        self._build_brand_lookup()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _build_brand_lookup(self):
        """Build a case-insensitive brand lookup dictionary"""
        self.brand_lookup = {}
        for meta in self.vector_store.metadata:
            brand = meta.get('brand', '')
            if brand:
                # Map lowercase brand to actual brand name
                self.brand_lookup[brand.lower()] = brand
        logger.info(f"Built brand lookup with {len(self.brand_lookup)} brands")
    
    def normalize_brand(self, brand: str) -> Optional[str]:
        """
        Normalize brand name to match database format (case-insensitive)
        
        Args:
            brand: Brand name from user query or LLM extraction
            
        Returns:
            Normalized brand name matching database, or None if not found
        """
        if not brand:
            return None
        
        # Direct case-insensitive lookup
        normalized = self.brand_lookup.get(brand.lower())
        if normalized:
            return normalized
        
        # Fallback: try partial matching for common variations
        brand_lower = brand.lower()
        for db_brand_lower, db_brand in self.brand_lookup.items():
            # Check if input is contained in database brand or vice versa
            if brand_lower in db_brand_lower or db_brand_lower in brand_lower:
                return db_brand
        
        # Return original if no match found
        return brand
    
    def is_greeting_or_casual(self, query: str) -> bool:
        """
        Check if query is just a greeting or casual message that doesn't need product search
        
        Args:
            query: User query
            
        Returns:
            True if it's a greeting/casual message, False if it needs product search
        """
        query_lower = query.lower().strip()
        
        # Common greetings and casual phrases
        greetings = [
            'hi', 'hello', 'hey', 'howdy', 'greetings',
            'good morning', 'good afternoon', 'good evening',
            'what\'s up', 'whats up', 'sup', 'yo',
            'thanks', 'thank you', 'ok', 'okay', 'bye', 'goodbye'
        ]
        
        # Check if query is just a greeting (exact match or very short)
        if query_lower in greetings or len(query.split()) <= 2 and any(g in query_lower for g in greetings):
            return True
        
        return False
    
    def extract_filters(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract brand and denomination filters from natural language query using LLM
        
        Args:
            query: User query
            
        Returns:
            Dict with 'brand' and 'denomination' keys (None if not found)
        """
        try:
            # Get prompts from manager
            prompts = self.prompt_manager.get_all_prompts()
            filter_extraction = prompts.get('filter_extraction', {})
            
            if not filter_extraction.get('enabled', False):
                return {'brand': None, 'denomination': None}
            
            system_prompt = filter_extraction.get('system_prompt', '')
            user_prompt_template = filter_extraction.get('user_prompt_template', '')
            user_prompt = user_prompt_template.replace('{query}', query)

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=100,
                timeout=10.0
            )
            
            content = response.choices[0].message.content
            if content:
                # Parse JSON response
                import json
                import re
                filters = json.loads(content.strip())
                brand = filters.get('brand')
                denomination = filters.get('denomination')
                
                # Convert "null" strings to None
                if brand == "null" or not brand:
                    brand = None
                if denomination == "null" or not denomination:
                    denomination = None
                
                # Normalize brand name using generic lookup
                if brand:
                    brand = self.normalize_brand(brand)
                
                # Normalize denomination format
                if denomination:
                    # Remove extra spaces
                    denomination = denomination.strip()
                    # Ensure Rs. prefix format
                    if not denomination.startswith('Rs.'):
                        # Extract number
                        numbers = re.findall(r'\d+', denomination)
                        if numbers:
                            denomination = f'Rs. {numbers[0]}'
                
                logger.info(f"Extracted filters - Brand: {brand}, Denomination: {denomination}")
                return {'brand': brand, 'denomination': denomination}
            
        except Exception as e:
            logger.debug(f"Filter extraction failed: {str(e)}")
        
        return {'brand': None, 'denomination': None}
    
    def reframe_query(self, query: str) -> str:
        """
        Reframe user query for better semantic search using LLM
        
        Args:
            query: Original user query
            
        Returns:
            Reframed query optimized for search
        """
        try:
            prompts = self.prompt_manager.get_all_prompts()
            query_reframing = prompts.get('query_reframing', {})
            
            if not query_reframing.get('enabled', False):
                return query
            
            system_prompt = query_reframing.get('system_prompt', '')
            instruction = query_reframing.get('instruction', '')
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{instruction}\n\nOriginal query: {query}"}
                ],
                temperature=0.3,
                max_tokens=100,
                timeout=10.0  # Add 10 second timeout
            )
            
            content = response.choices[0].message.content
            reframed = content.strip() if content else query
            logger.info(f"Query reframed: '{query}' -> '{reframed}'")
            return reframed
            
        except Exception as e:
            error_type = type(e).__name__
            logger.warning(f"Query reframing failed ({error_type}), using original query")
            logger.debug(f"Reframing error details: {str(e)}")
            return query
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for user query"""
        return self.embeddings_generator.generate_embedding(query)
    
    def search(self, query: str, top_k: Optional[int] = None, 
              brand: Optional[str] = None, denomination: Optional[str] = None,
              min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant products with separate semantic and keyword scores
        
        Args:
            query: User search query
            top_k: Number of results (default from config)
            brand: Optional brand filter
            denomination: Optional denomination filter
            min_score: Minimum similarity score
            
        Returns:
            List of relevant products with semantic_score, keyword_score, and combined_score
        """
        # Use config defaults if not specified
        top_k = top_k or Config.TOP_K_RESULTS
        min_score = min_score if min_score is not None else 0.0
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Use hybrid_search for filtering by brand/denomination
        results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            top_k=top_k,
            brand=brand,
            denomination=denomination,
            min_score=min_score,
            query_text=query
        )
        
        return results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into context for LLM with complete JSON data and separate scores"""
        if not results:
            return "No relevant products found."
        
        import json
        context_parts = ["Here are the relevant products with complete data and relevance scores:\n"]
        
        for i, result in enumerate(results, 1):
            # Create clean product data (remove internal fields)
            product_data = {
                'id': result.get('id'),
                'product_title': result.get('product_title'),
                'brand': result.get('brand'),
                'denomination': result.get('denomination'),
                'validity': result.get('validity'),
                'validity_duration': result.get('validity_duration'),
                # Two main scoring metrics
                'semantic_score': round(result.get('semantic_score', 0), 3),
                'similarity_score': round(result.get('similarity_score', 0), 3),
                'combined_score': round(result.get('combined_score', result.get('semantic_score', 0)), 3),
                'full_details': result.get('text', '')
            }
            
            # Format as JSON
            context_parts.append(f"\nProduct {i}:")
            context_parts.append(json.dumps(product_data, indent=2, ensure_ascii=False))
            context_parts.append("")
        
        # Add scoring explanation with current weights
        context_parts.append("\nScoring Explanation:")
        context_parts.append("- semantic_score: Vector embedding similarity based on meaning (0-1)")
        context_parts.append("- similarity_score: Keyword/lexical similarity based on exact term matching (0-1)")
        context_parts.append(f"- combined_score: Weighted combination ({int(Config.SEMANTIC_WEIGHT*100)}% semantic + {int(Config.SIMILARITY_WEIGHT*100)}% similarity)")
        
        return '\n'.join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate natural language response using GPT-4
        
        Args:
            query: User query
            context: Retrieved product context
            
        Returns:
            Generated response
        """
        try:
            # Get prompts from manager (cached, no file I/O)
            system_prompt = self.prompt_manager.get_system_prompt()
            user_prompt = self.prompt_manager.get_user_prompt(query, context)
            
            logger.debug(f"Generating response for query: {query[:50]}...")
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )
            
            logger.info("Response generated successfully")
            content = response.choices[0].message.content
            return content if content else "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def query(self, user_query: str, top_k: Optional[int] = None,
             brand: Optional[str] = None, denomination: Optional[str] = None,
             return_raw: bool = False) -> Dict[str, Any]:
        """
        Complete RAG query pipeline with query reframing
        
        Args:
            user_query: User's search query
            top_k: Number of results to retrieve
            brand: Optional brand filter
            denomination: Optional denomination filter
            return_raw: If True, return raw results without LLM response
            
        Returns:
            Dict with response and retrieved products
        """
        # Check if it's just a greeting - respond directly without product search
        if self.is_greeting_or_casual(user_query):
            greeting_response = (
                "Hello! 👋 I'm here to help you find the perfect products from our catalog. "
                "I can assist you with:\n"
                "- Credit cards (cashback, rewards, travel, fuel)\n"
                "- Gift vouchers (food, shopping, entertainment)\n"
                "- Subscriptions (streaming, lifestyle)\n"
                "- And more!\n\n"
                "What are you looking for today?"
            )
            return {
                'original_query': user_query,
                'reframed_query': None,
                'response': greeting_response,
                'results': [],
                'count': 0,
                'is_greeting': True,
                'extracted_filters': {}
            }
        
        # Step 1: Extract filters if not provided (auto-detect from query)
        extracted_filters = {}
        if brand is None or denomination is None:
            extracted_filters = self.extract_filters(user_query)
            brand = brand or extracted_filters.get('brand')
            denomination = denomination or extracted_filters.get('denomination')
            logger.info(f"Auto-extracted filters - Brand: {brand}, Denomination: {denomination}")
        
        # Step 2: Reframe query for better search
        reframed_query = self.reframe_query(user_query)
        
        # Step 3: Search for relevant products using reframed query and filters
        results = self.search(
            query=reframed_query,
            top_k=top_k,
            brand=brand,
            denomination=denomination
        )
        
        if return_raw:
            return {
                'original_query': user_query,
                'reframed_query': reframed_query,
                'results': results,
                'count': len(results),
                'extracted_filters': extracted_filters
            }
        
        # Step 4: Format context
        context = self.format_context(results)
        
        # Step 5: Generate response with nudges
        response = self.generate_response(user_query, context)
        
        return {
            'original_query': user_query,
            'reframed_query': reframed_query,
            'response': response,
            'results': results,
            'count': len(results),
            'is_greeting': False,
            'extracted_filters': extracted_filters
        }


if __name__ == "__main__":
    # Example usage
    from config import Config
    import os
    
    # Load vector store
    store = VectorStore(dimension=Config.EMBEDDING_DIMENSION)
    
    if os.path.exists(Config.VECTOR_STORE_PATH):
        store.load(
            Config.VECTOR_STORE_PATH,
            Config.VECTOR_STORE_PATH.replace('.faiss', '_metadata.pkl')
        )
    else:
        print("WARNING: Vector store not found. Run vector_store.py first to build the index.")
        exit(1)
    
    # Create RAG pipeline
    rag = RAGPipeline(store)
    
    # Example queries
    queries = [
        "Find me cashback credit cards",
        "Show me Zee5 subscription vouchers",
        "Pizza delivery gift cards under Rs. 500"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = rag.query(query)
        print(f"\n{result['response']}")
        print(f"\nFound {result['count']} relevant products")
