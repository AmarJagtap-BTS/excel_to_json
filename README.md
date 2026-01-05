# Excel to JSON Parser with RAG Search

A complete solution for parsing Excel files into JSON format and providing intelligent search capabilities using RAG (Retrieval Augmented Generation).

## Features

- **Excel to JSON Conversion**: Parse Excel files into structured JSON
- **RAG Search**: Semantic search using Azure OpenAI embeddings
- **Configurable Similarity Metrics**: Choose between cosine and semantic similarity
- **Filter Extraction**: Automatic filter extraction from natural language queries
- **Streamlit Web Interface**: User-friendly search interface

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AmarJagtap-BTS/excel_to_json.git
cd excel_to_json
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r ingestion/requirements.txt
pip install -r rag_retrieval/requirements.txt
```

### 3. Configure Azure OpenAI Credentials

Choose one of the following methods:

#### Option A: Environment Variables (Local Development)

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

#### Option B: Streamlit Secrets (Local & Cloud)

Create `.streamlit/secrets.toml` in the `rag_retrieval` folder:

```bash
cd rag_retrieval
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your actual credentials
```

#### Option C: Streamlit Cloud (Deployment)

1. Go to your Streamlit Cloud app settings
2. Click on "Secrets" in the left sidebar
3. Add the following secrets:

```toml
AZURE_OPENAI_ENDPOINT = "your-endpoint-here"
AZURE_OPENAI_API_KEY = "your-api-key-here"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
CHAT_DEPLOYMENT = "gpt-4o-mini"
```

## Usage

### Excel to JSON Conversion

```bash
cd ingestion
python main.py
```

Place your Excel files in the `input/` folder. Converted JSON will be in `output/`.

### RAG Search Application

```bash
cd rag_retrieval
streamlit run app.py
```

The application will:
1. Build the knowledge base from `products.json`
2. Launch the web interface at `http://localhost:8501`

## Configuration

Key configuration options in `rag_retrieval/src/config.py`:

- `SIMILARITY_METRIC`: Choose `"cosine"` or `"semantic"`
- `TOP_K_RESULTS`: Number of search results to return
- `SEMANTIC_WEIGHT`: Weight for semantic similarity (0-1)
- `TEMPERATURE`: LLM response randomness (0 = deterministic)

## Project Structure

```
excel_to_json_parser/
├── ingestion/              # Excel to JSON conversion
│   ├── main.py
│   └── requirements.txt
├── rag_retrieval/          # RAG search application
│   ├── app.py              # Streamlit interface
│   ├── api.py              # REST API
│   ├── prompts.json        # LLM prompts
│   ├── src/
│   │   ├── config.py       # Configuration
│   │   ├── embeddings.py   # Embedding generation
│   │   ├── vector_store.py # FAISS vector database
│   │   └── rag_pipeline.py # RAG pipeline
│   └── .streamlit/
│       └── secrets.toml.example
├── input/                  # Excel input files
└── output/                 # JSON output files
```

## Security

⚠️ **Never commit credentials to Git!**

The following files are excluded via `.gitignore`:
- `.env`
- `.streamlit/secrets.toml`
- `*.pkl` (embeddings cache)
- `*.faiss` (vector store)

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Set main file path: `rag_retrieval/app.py`
5. Add secrets in app settings (see Option C above)
6. Deploy! 🚀

## License

MIT License
