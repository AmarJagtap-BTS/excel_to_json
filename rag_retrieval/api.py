from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import subprocess
import glob
import json

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from config import Config  # type: ignore
from vector_store import VectorStore  # type: ignore
from rag_pipeline import RAGPipeline  # type: ignore
from prompt_manager import PromptManager  # type: ignore
from embeddings import EmbeddingsGenerator  # type: ignore

# Initialize FastAPI app
app = FastAPI(
    title="Product Catalog RAG API",
    description="AI-Powered Product Search API using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG system
rag_system = None
vector_store = None
prompt_manager = None


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 30
    brand: Optional[str] = None
    return_raw: Optional[bool] = False


class ProductResult(BaseModel):
    id: Optional[int]
    product_title: Optional[str]
    brand: Optional[str]
    denomination: Optional[str]
    validity: Optional[str]
    semantic_score: float
    similarity_score: Optional[float]
    combined_score: Optional[float]


class QueryResponse(BaseModel):
    response: str
    original_query: str
    reframed_query: Optional[str]
    count: int
    results: List[Dict[str, Any]]


class StatusResponse(BaseModel):
    status: str
    message: str
    products_count: Optional[int] = None


def check_and_build_knowledge_base() -> tuple[bool, str, Optional[VectorStore]]:
    """
    Check if knowledge base exists, if not build it automatically
    Returns: (success, message, optional_prebuilt_store)
    """
    # Get paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    rag_folder = os.path.dirname(__file__)
    input_folder = os.path.join(project_root, 'input')
    json_path = os.path.join(project_root, 'output', 'products.json')
    vector_store_path = os.path.join(rag_folder, 'data', 'vector_store.faiss')
    metadata_path = vector_store_path.replace('.faiss', '_metadata.pkl')
    
    # Step 1: Check if vector database exists
    if os.path.exists(vector_store_path) and os.path.exists(metadata_path):
        return True, "Vector database found and loaded", None
    
    # Step 2: Vector DB not found, check for JSON file
    if not os.path.exists(json_path):
        # Step 3: JSON not found, check for Excel files in input folder
        excel_files = glob.glob(os.path.join(input_folder, '*.xlsx')) + glob.glob(os.path.join(input_folder, '*.xls'))
        
        if not excel_files:
            return False, "Knowledge Base not present. Please add an Excel file (.xlsx/.xls) to the 'input' folder.", None
        
        # Step 4: Excel file found, convert to JSON
        print(f"Found Excel file: {os.path.basename(excel_files[0])}")
        print("Converting Excel to JSON...")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            # Run the Excel to JSON converter
            main_script = os.path.join(project_root, 'src', 'main.py')
            result = subprocess.run(
                ['python', main_script, '--output', json_path],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode != 0:
                return False, f"Error converting Excel to JSON: {result.stderr}", None
            
            print(f"Excel converted to JSON successfully")
            
        except Exception as e:
            return False, f"Error converting Excel to JSON: {str(e)}", None
    
    # Step 5: JSON exists (or just created), build vector store
    print("Building vector store from JSON...")
    
    try:
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        if not products:
            return False, "JSON file is empty", None
        
        print(f"Loaded {len(products)} products from JSON")
        
        # Create embeddings generator
        embeddings_gen = EmbeddingsGenerator()
        
        # Generate embeddings for all products
        print("Generating embeddings (this may take a few minutes)...")
        
        embeddings_data = []
        for idx, product in enumerate(products):
            # Create searchable text
            text = embeddings_gen.create_searchable_text(product)
            
            # Generate embedding
            embedding = embeddings_gen.generate_embedding(text)
            
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
            
            # Update progress
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(products)} products...")
        
        print(f"Generated embeddings for all {len(products)} products")
        
        # Create vector store and add embeddings
        store = VectorStore(dimension=Config.EMBEDDING_DIMENSION)
        store.add_embeddings(embeddings_data)
        
        # Save vector store
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        store.save(vector_store_path, metadata_path)
        
        # Save embeddings cache
        embeddings_cache_path = os.path.join(os.path.dirname(vector_store_path), 'embeddings.pkl')
        embeddings_gen.save_embeddings(embeddings_data, embeddings_cache_path)
        
        # Copy JSON to data folder for reference
        data_json_path = os.path.join(rag_folder, 'data', 'products.json')
        with open(data_json_path, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        
        print(f"Vector store built successfully with {len(products)} products")
        return True, f"Knowledge base initialized with {len(products)} products", store
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error building vector store: {str(e)}")
        print(error_details)
        return False, f"Error building vector store: {str(e)}", None


def load_rag_system():
    """Load and initialize the RAG system"""
    global rag_system, vector_store, prompt_manager
    
    try:
        # Check and build knowledge base if needed
        success, message, pre_built_store = check_and_build_knowledge_base()
        
        if not success:
            raise Exception(message)
        
        print(message)
        
        # Load vector store
        if pre_built_store:
            vector_store = pre_built_store
        else:
            vector_store = VectorStore(dimension=Config.EMBEDDING_DIMENSION)
            vector_store_path = os.path.join(os.path.dirname(__file__), 'data', 'vector_store.faiss')
            metadata_path = vector_store_path.replace('.faiss', '_metadata.pkl')
            vector_store.load(vector_store_path, metadata_path)
        
        # Create RAG pipeline
        rag_system = RAGPipeline(vector_store)
        
        # Load prompts
        prompt_manager = PromptManager()
        
        return True
    
    except Exception as e:
        print(f"Error loading RAG system: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    print("Initializing RAG system...")
    success = load_rag_system()
    if success:
        print("RAG system initialized successfully!")
    else:
        print("WARNING: RAG system initialization failed. Check logs.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Product Catalog RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint"""
    if rag_system is None or vector_store is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return StatusResponse(
        status="healthy",
        message="RAG system is running",
        products_count=vector_store.size
    )


@app.post("/query", response_model=QueryResponse)
async def query_products(request: QueryRequest):
    """
    Query products using natural language
    
    - **query**: Natural language query
    - **top_k**: Number of results to return (default: 30)
    - **brand**: Optional brand filter
    - **return_raw**: Return raw results without LLM response
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Execute query
        result = rag_system.query(
            user_query=request.query,
            top_k=request.top_k,
            brand=request.brand,
            return_raw=request.return_raw
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get system statistics"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "total_products": vector_store.size,
        "embedding_dimension": Config.EMBEDDING_DIMENSION,
        "semantic_weight": Config.SEMANTIC_WEIGHT,
        "similarity_weight": Config.SIMILARITY_WEIGHT,
        "top_k_results": Config.TOP_K_RESULTS,
        "similarity_threshold": Config.SIMILARITY_THRESHOLD,
        "model": Config.CHAT_MODEL,
        "embedding_model": Config.EMBEDDING_MODEL
    }


@app.post("/rebuild", response_model=StatusResponse)
async def rebuild_knowledge_base():
    """Force rebuild the knowledge base from Excel/JSON"""
    try:
        # Delete existing vector store
        vector_store_path = os.path.join(os.path.dirname(__file__), 'data', 'vector_store.faiss')
        metadata_path = vector_store_path.replace('.faiss', '_metadata.pkl')
        
        if os.path.exists(vector_store_path):
            os.remove(vector_store_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Reload RAG system (will rebuild automatically)
        success = load_rag_system()
        
        if success:
            return StatusResponse(
                status="success",
                message="Knowledge base rebuilt successfully",
                products_count=vector_store.size if vector_store else 0
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to rebuild knowledge base")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
