import streamlit as st
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


# Page configuration
st.set_page_config(
    page_title="Product Catalog Search",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .brand-tag {
        background-color: #2196F3;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def check_and_build_knowledge_base():
    """
    Check if vector database exists, if not, build it from JSON or Excel.
    Returns: (success: bool, message: str, store: VectorStore or None)
    """
    # Get absolute paths
    rag_folder = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(rag_folder)
    vector_store_path = os.path.join(rag_folder, 'data', 'vector_store.faiss')
    metadata_path = vector_store_path.replace('.faiss', '_metadata.pkl')
    json_path = os.path.join(project_root, 'output', 'products.json')
    input_folder = os.path.join(project_root, 'input')
    
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
        st.info(f"Found Excel file: {os.path.basename(excel_files[0])}")
        st.info("Converting Excel to JSON...")
        
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
            
            st.success(f"Excel converted to JSON successfully")
            
        except Exception as e:
            return False, f"Error converting Excel to JSON: {str(e)}", None
    
    # Step 5: JSON exists (or just created), build vector store
    
    st.info("Building vector store from JSON...")
    
    try:
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        if not products:
            return False, "JSON file is empty", None
        
        st.info(f"Loaded {len(products)} products from JSON")
        
        # Create embeddings generator
        embeddings_gen = EmbeddingsGenerator()
        
        # Generate embeddings for all products
        st.info("Generating embeddings (this may take a few minutes)...")
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
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
            progress = (idx + 1) / len(products)
            progress_bar.progress(progress)
            progress_text.text(f"Processed {idx + 1}/{len(products)} products...")
        
        progress_text.text(f"Generated embeddings for all {len(products)} products")
        
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
        
        st.success(f"Vector store built successfully with {len(products)} products")
        return True, f"Knowledge base initialized with {len(products)} products", store
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error building vector store: {str(e)}")
        st.code(error_details)
        return False, f"Error building vector store: {str(e)}", None


@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    try:
        # Check and build knowledge base if needed
        success, message, pre_built_store = check_and_build_knowledge_base()
        
        if not success:
            st.error(message)
            return None
        
        st.success(message)
        
        # Load vector store
        if pre_built_store:
            store = pre_built_store
        else:
            store = VectorStore(dimension=Config.EMBEDDING_DIMENSION)
            vector_store_path = os.path.join(os.path.dirname(__file__), 'data', 'vector_store.faiss')
            metadata_path = vector_store_path.replace('.faiss', '_metadata.pkl')
            store.load(vector_store_path, metadata_path)
        
        # Create RAG pipeline
        rag = RAGPipeline(store)
        return rag, store
    
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None


# Removed unused display_product_card function


@st.cache_resource
def load_prompts():
    """Load prompts using PromptManager (singleton, cached)"""
    manager = PromptManager()
    return manager


def main():
    # Load prompts
    prompts = load_prompts()
    
    # Header
    st.markdown('<h1 class="main-header">Product Catalog Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Product Search Chatbot")
    st.markdown("---")
    
    # Load RAG system
    result = load_rag_system()
    if result is None:
        st.stop()
    
    rag, store = result
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Show reframed query for assistant messages if available
            if message["role"] == "assistant" and message.get("reframed_query"):
                if message.get("reframed_query") != st.session_state.messages[st.session_state.messages.index(message) - 1].get("content", ""):
                    st.info(f"**Optimized search:** {message['reframed_query']}")
            
            st.markdown(message["content"])
            
            # Show product count for assistant messages
            if message["role"] == "assistant" and message.get("count"):
                st.caption(f"Found {message['count']} relevant products")
            
            # Products hidden - only showing AI response
    
    # Chat input
    if prompt := st.chat_input(prompts.get_ui_message('chat_input_placeholder')):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching products..."):
                try:
                    # Execute query with reframing and auto filter extraction
                    result = rag.query(
                        user_query=prompt,
                        top_k=Config.TOP_K_RESULTS,
                        brand=None,  # Let the system auto-extract from query
                        denomination=None,  # Let the system auto-extract from query
                        return_raw=False
                    )
                    
                    # Show extracted filters if any (commented out to hide from UI)
                    # if result.get('extracted_filters'):
                    #     filters = result['extracted_filters']
                    #     if filters.get('brand') or filters.get('denomination'):
                    #         filter_parts = []
                    #         if filters.get('brand'):
                    #             filter_parts.append(f"Brand: **{filters['brand']}**")
                    #         if filters.get('denomination'):
                    #             filter_parts.append(f"Denomination: **{filters['denomination']}**")
                    #         st.info(f"🔍 Applied filters: {', '.join(filter_parts)}")
                    
                    # Show query reframing if different (commented out to hide from UI)
                    # if result.get('reframed_query') and result['reframed_query'] != result['original_query']:
                    #     st.info(f"**Optimized search:** {result['reframed_query']}")
                    
                    # Display AI response
                    st.markdown(result['response'])
                    
                    # Show scoring metrics
                    if result.get('results') and len(result['results']) > 0:
                        st.caption(f"Found {result['count']} relevant products")
                        
                        # Show scoring breakdown for top results in expander
                        with st.expander("View Scoring Details", expanded=False):
                            st.markdown(f"""
                            **Scoring Configuration:**
                            - Combined Score = `{int(Config.SEMANTIC_WEIGHT*100)}%` Semantic + `{int(Config.SIMILARITY_WEIGHT*100)}%` Similarity
                            
                            **What each score means:**
                            - **Semantic**: Vector embedding similarity (meaning & context)
                            - **Similarity**: Keyword/lexical matching (exact terms)
                            """)
                            
                            st.markdown("**Scoring Breakdown (Top 5 Results):**")
                            for i, prod in enumerate(result['results'][:5], 1):
                                semantic = prod.get('semantic_score', 0)
                                similarity = prod.get('similarity_score', 0)
                                combined = prod.get('combined_score', semantic)
                                
                                st.markdown(f"""
                                **{i}. {prod.get('product_title', 'N/A')}** (ID: {prod.get('id', 'N/A')})
                                - Semantic Score: `{semantic:.3f}` (vector embedding similarity)
                                - Similarity Score: `{similarity:.3f}` (keyword matching)
                                - Combined Score: `{combined:.3f}` ({int(Config.SEMANTIC_WEIGHT*100)}% semantic + {int(Config.SIMILARITY_WEIGHT*100)}% similarity)
                                """)
                    
                    # Products hidden - only showing AI response
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['response'],
                        "products": result['results'],
                        "reframed_query": result.get('reframed_query', ''),
                        "count": result['count']
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
