import json
import pickle
import os
from typing import List, Dict, Any
from openai import AzureOpenAI
from config import Config
import numpy as np


class EmbeddingsGenerator:
    """Generate and manage embeddings for product catalog using Azure OpenAI"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        Config.validate()
        self.client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )
        self.embedding_model = Config.EMBEDDING_DEPLOYMENT
        
    def load_products(self, json_path: str) -> List[Dict[str, Any]]:
        """Load product catalog from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        print(f"Loaded {len(products)} products from {json_path}")
        return products
    
    def create_searchable_text(self, product: Dict[str, Any]) -> str:
        """Combine product fields into searchable text"""
        parts = []
        
        # Add title
        if product.get('Product Title'):
            parts.append(f"Title: {product['Product Title']}")
        
        # Add brand
        if product.get('Brand Name'):
            parts.append(f"Brand: {product['Brand Name']}")
        
        # Add short description
        if product.get('Product Short Description'):
            parts.append(f"Description: {product['Product Short Description']}")
        
        # Add long description (join array if it's a list)
        long_desc = product.get('Product Long Description')
        if long_desc:
            if isinstance(long_desc, list):
                long_desc = ' '.join(long_desc)
            parts.append(f"Details: {long_desc}")
        
        # Add denomination
        if product.get('denomination'):
            parts.append(f"Price: {product['denomination']}")
        
        # Add validity info
        if product.get('Validity') and product['Validity'] != 'na':
            parts.append(f"Validity: {product['Validity']}")
        
        return ' '.join(parts)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Azure OpenAI"""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def generate_embeddings_batch(self, products: List[Dict[str, Any]], 
                                  batch_size: int = 100) -> List[Dict[str, Any]]:
        """Generate embeddings for all products with metadata"""
        embeddings_data = []
        
        for i, product in enumerate(products):
            # Create searchable text
            text = self.create_searchable_text(product)
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Store embedding with metadata
            embeddings_data.append({
                'id': product.get('id'),
                'text': text,
                'embedding': embedding,
                'metadata': {
                    'product_title': product.get('Product Title'),
                    'brand': product.get('Brand Name'),
                    'denomination': product.get('denomination'),
                    'validity': product.get('Validity'),
                    'validity_duration': product.get('Validity Duration')
                }
            })
            
            if (i + 1) % 10 == 0:
                print(f"Generated embeddings for {i + 1}/{len(products)} products")
        
        print(f"Completed: Generated embeddings for all {len(products)} products")
        return embeddings_data
    
    def save_embeddings(self, embeddings_data: List[Dict[str, Any]], 
                       cache_path: str):
        """Save embeddings to pickle file for caching"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"Saved embeddings to {cache_path}")
    
    def load_embeddings(self, cache_path: str) -> List[Dict[str, Any]]:
        """Load embeddings from cache"""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Embeddings cache not found at {cache_path}")
        
        with open(cache_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        print(f"Loaded {len(embeddings_data)} embeddings from cache")
        return embeddings_data
    
    def process_and_cache(self, json_path: str, cache_path: str, 
                         force_regenerate: bool = False) -> List[Dict[str, Any]]:
        """Load products, generate embeddings, and cache them"""
        # Check if cache exists
        if os.path.exists(cache_path) and not force_regenerate:
            print("Using cached embeddings...")
            return self.load_embeddings(cache_path)
        
        # Generate new embeddings
        print("Generating new embeddings...")
        products = self.load_products(json_path)
        embeddings_data = self.generate_embeddings_batch(products)
        self.save_embeddings(embeddings_data, cache_path)
        
        return embeddings_data


if __name__ == "__main__":
    # Example usage
    generator = EmbeddingsGenerator()
    embeddings = generator.process_and_cache(
        json_path=Config.PRODUCTS_JSON_PATH,
        cache_path=Config.EMBEDDINGS_CACHE_PATH,
        force_regenerate=False
    )
    print(f"\nReady! {len(embeddings)} product embeddings available")
