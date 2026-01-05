import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional


class VectorStore:
    """FAISS-based vector store with configurable similarity metric"""
    
    def __init__(self, dimension: int = 3072):
        """
        Initialize FAISS index with configurable similarity metric
        
        Args:
            dimension: Embedding dimension (3072 for text-embedding-3-large)
        """
        self.dimension = dimension
        
        # Import config for similarity metric and scoring weights
        from config import Config
        self.config = Config
        
        # Create index based on configured similarity metric
        self.similarity_metric = self.config.SIMILARITY_METRIC.lower()
        if self.similarity_metric == "cosine":
            # Use IndexFlatIP (Inner Product) with normalized vectors for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            print(f"Initialized FAISS index with COSINE similarity")
        elif self.similarity_metric == "semantic":
            # Use IndexFlatIP without normalization for semantic (dot product) similarity
            self.index = faiss.IndexFlatIP(dimension)
            print(f"Initialized FAISS index with SEMANTIC (dot product) similarity")
        else:
            raise ValueError(f"Unsupported similarity metric: {self.config.SIMILARITY_METRIC}. Use 'cosine' or 'semantic'")
        
        self.metadata = []  # Store product metadata
        
    def add_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        """
        Add embeddings to the vector store
        
        Args:
            embeddings_data: List of dicts with 'embedding' and 'metadata'
        """
        # Extract embeddings
        embeddings = np.array([item['embedding'] for item in embeddings_data], 
                             dtype=np.float32)
        
        # Normalize vectors only for cosine similarity
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(embeddings)
            print(f"Normalized embeddings for cosine similarity")
        else:
            print(f"Using unnormalized embeddings for dot product similarity")
        
        # Add to index
        self.index.add(embeddings)  # type: ignore
        
        # Store metadata
        self.metadata = [
            {
                'id': item['id'],
                'text': item['text'],
                **item['metadata']
            }
            for item in embeddings_data
        ]
        
        print(f"Added {len(embeddings_data)} vectors to FAISS index")
        print(f"   Index size: {self.index.ntotal} vectors")
        print(f"   Similarity metric: {self.similarity_metric}")
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
               min_score: float = 0.0, query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar products using configured similarity metric with separate scoring
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1 for cosine, unbounded for semantic)
            query_text: Optional query text for keyword matching score
            
        Returns:
            List of results with metadata and separate scores
        """
        # Prepare query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize query vector only for cosine similarity
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)  # type: ignore
        
        # Format results with separate scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= min_score:  # Valid result
                # Convert numpy float32 to Python float for JSON serialization
                semantic_score = float(score)
                
                result = {
                    'semantic_score': semantic_score,  # Semantic similarity (0-1 for cosine, unbounded for semantic)
                    'score': semantic_score,  # Keep for backward compatibility
                    'similarity_metric': self.similarity_metric,  # Include metric used
                    **self.metadata[idx]
                }
                
                # Calculate keyword similarity if query text provided
                if query_text:
                    similarity_score = self._calculate_keyword_similarity(
                        query_text, 
                        result.get('text', '')
                    )
                    result['similarity_score'] = float(similarity_score)
                    # Combined score using config weights
                    semantic_weight = self.config.SEMANTIC_WEIGHT
                    similarity_weight = self.config.SIMILARITY_WEIGHT
                    result['combined_score'] = float((semantic_weight * semantic_score) + (similarity_weight * similarity_score))
                
                results.append(result)
        
        return results
    
    def _calculate_keyword_similarity(self, query: str, text: str) -> float:
        """
        Calculate keyword-based similarity score
        
        Args:
            query: Search query
            text: Product text
            
        Returns:
            Keyword similarity score (0-1)
        """
        if not query or not text:
            return 0.0
        
        # Convert to lowercase and split into words
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        query_words = query_words - stop_words
        text_words = text_words - stop_words
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # Calculate coverage (how many query words are in text)
        coverage_score = intersection / len(query_words) if query_words else 0.0
        
        # Weighted combination (50% Jaccard + 50% Coverage)
        keyword_similarity = (0.5 * jaccard_score) + (0.5 * coverage_score)
        
        return float(keyword_similarity)
    
    def filter_by_metadata(self, brand: Optional[str] = None, 
                          denomination: Optional[str] = None) -> List[int]:
        """
        Get indices of products matching metadata filters
        
        Args:
            brand: Filter by brand name
            denomination: Filter by denomination
            
        Returns:
            List of matching indices
        """
        matching_indices = []
        
        for idx, meta in enumerate(self.metadata):
            match = True
            
            if brand and meta.get('brand') != brand:
                match = False
            
            if denomination and meta.get('denomination') != denomination:
                match = False
            
            if match:
                matching_indices.append(idx)
        
        return matching_indices
    
    def hybrid_search(self, query_embedding: List[float], top_k: int = 10,
                     brand: Optional[str] = None, denomination: Optional[str] = None,
                     min_score: float = 0.0, query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search: Vector similarity (using configured metric) + metadata filtering + keyword matching
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            brand: Optional brand filter
            denomination: Optional denomination filter
            min_score: Minimum similarity score (0-1 for cosine, unbounded for semantic)
            query_text: Optional query text for keyword similarity
            
        Returns:
            Filtered and ranked results with separate scores
        """
        # Get all results with separate scoring
        all_results = self.search(query_embedding, top_k=100, min_score=min_score, query_text=query_text)
        
        # Apply metadata filters
        filtered_results = []
        for result in all_results:
            match = True
            
            # Case-insensitive brand matching
            if brand:
                result_brand = result.get('brand', '').lower()
                filter_brand = brand.lower()
                if result_brand != filter_brand:
                    match = False
            
            # Case-insensitive denomination matching
            if denomination:
                result_denom = str(result.get('denomination', '')).lower()
                filter_denom = str(denomination).lower()
                if result_denom != filter_denom:
                    match = False
            
            if match:
                filtered_results.append(result)
            
            # Stop when we have enough results
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results[:top_k]
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Saved vector store to {index_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded vector store with {self.index.ntotal} vectors")
        print(f"Using similarity metric: {self.similarity_metric}")
        print(f"Note: Make sure the loaded index was created with the same similarity metric!")
    
    @property
    def size(self) -> int:
        """Get number of vectors in the store"""
        return self.index.ntotal


if __name__ == "__main__":
    # Example usage
    from embeddings import EmbeddingsGenerator
    from config import Config
    
    # Generate embeddings
    generator = EmbeddingsGenerator()
    embeddings_data = generator.process_and_cache(
        Config.PRODUCTS_JSON_PATH,
        Config.EMBEDDINGS_CACHE_PATH
    )
    
    # Create vector store
    store = VectorStore(dimension=Config.EMBEDDING_DIMENSION)
    store.add_embeddings(embeddings_data)
    
    # Save
    store.save(
        Config.VECTOR_STORE_PATH,
        Config.VECTOR_STORE_PATH.replace('.faiss', '_metadata.pkl')
    )
    
    print(f"\nVector store ready with {store.size} products!")
