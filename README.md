# Excel to JSON Parser with RAG Search

AI-powered product catalog management system with Excel parsing and intelligent semantic search using Azure OpenAI.

## 🚀 Features

### 1. Excel to JSON Conversion (`ingestion/`)
- Parse Excel files with product catalogs
- Clean and structure data
- Generate searchable JSON format
- Handle multiple product attributes

### 2. RAG-based Search (`rag_retrieval/`)
- **Semantic Search**: Vector embeddings using Azure OpenAI
- **Intelligent Filtering**: Auto-extract brands and denominations
- **Query Optimization**: Automatic query reframing for better results
- **Dual Scoring**: Semantic + keyword similarity
- **Flexible Metrics**: Cosine or semantic (dot product) similarity
- **Web Interface**: Interactive Streamlit chat UI

## 📋 Prerequisites

- Python 3.8+
- Azure OpenAI account with:
  - `text-embedding-3-large` deployment
  - `gpt-4o-mini` (or similar) chat deployment

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd excel_to_json_parser
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# For ingestion (Excel to JSON)
pip install -r ingestion/requirements.txt

# For RAG search
pip install -r rag_retrieval/requirements.txt
```

### 4. Configure Credentials

Create `.streamlit/secrets.toml` in the `rag_retrieval` folder:

```bash
cd rag_retrieval
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your Azure OpenAI credentials:

```toml
AZURE_OPENAI_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key-here"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
CHAT_DEPLOYMENT = "gpt-4o-mini"

SIMILARITY_METRIC = "cosine"
SEMANTIC_WEIGHT = "0.7"
SIMILARITY_WEIGHT = "0.3"
```

## 📖 Usage

### Step 1: Convert Excel to JSON

Place your Excel file in the `input/` folder, then:

```bash
cd ingestion
python main.py --output ../output/products.json
```

This will:
- Parse your Excel file
- Clean and structure the data
- Save as JSON in `output/products.json`

### Step 2: Run RAG Search Application

```bash
cd rag_retrieval
streamlit run app.py
```

The app will:
- Load products from JSON (or convert Excel if needed)
- Generate vector embeddings
- Build FAISS index
- Launch interactive search interface

Access the app at: http://localhost:8501

## 🎯 Project Structure

```
excel_to_json_parser/
├── ingestion/                  # Excel to JSON converter
│   ├── main.py                # Main conversion script
│   └── requirements.txt       # Dependencies
│
├── input/                     # Place Excel files here
│   └── your_products.xlsx
│
├── output/                    # Generated JSON files
│   └── products.json
│
└── rag_retrieval/             # RAG search application
    ├── app.py                 # Streamlit web app
    ├── api.py                 # REST API (optional)
    ├── prompts.json           # LLM prompts configuration
    ├── requirements.txt       # Dependencies
    │
    ├── .streamlit/
    │   ├── secrets.toml       # YOUR CREDENTIALS (not tracked)
    │   └── secrets.toml.example  # Template
    │
    ├── src/
    │   ├── config.py          # Configuration management
    │   ├── embeddings.py      # Embedding generation
    │   ├── vector_store.py    # FAISS vector database
    │   ├── rag_pipeline.py    # Main RAG logic
    │   ├── prompt_manager.py  # Prompt management
    │   └── utils.py           # Utilities
    │
    ├── data/
    │   ├── products.json      # Product catalog
    │   ├── vector_store.faiss # Vector index (generated)
    │   └── metadata.pkl       # Metadata (generated)
    │
    └── docs/
        ├── README.md                    # This file
        ├── CONFIGURATION_GUIDE.md       # Configuration options
        ├── SECURITY.md                  # Security best practices
        ├── SIMILARITY_METRICS.md        # Understanding similarity
        └── GITHUB_SETUP.md              # GitHub deployment guide
```

## 🔧 Configuration

### Similarity Metrics

Choose between two similarity approaches in `secrets.toml`:

```toml
SIMILARITY_METRIC = "cosine"    # Normalized (0-1), recommended
# OR
SIMILARITY_METRIC = "semantic"  # Unnormalized dot product
```

See `rag_retrieval/SIMILARITY_METRICS.md` for details.

### Scoring Weights

Adjust the balance between semantic and keyword matching:

```toml
SEMANTIC_WEIGHT = "0.7"     # Vector similarity (70%)
SIMILARITY_WEIGHT = "0.3"   # Keyword matching (30%)
```

### Search Parameters

```toml
TOP_K_RESULTS = "10"        # Number of results
TEMPERATURE = "0"           # LLM creativity (0-1)
MAX_TOKENS = "1000"        # Response length
```

## 🔐 Security

**Your Azure OpenAI credentials are protected:**
- ✅ Stored in `.streamlit/secrets.toml` (ignored by Git)
- ✅ Never committed to repository
- ✅ Each user uses their own credentials
- ✅ Template provided for easy setup

See `rag_retrieval/SECURITY.md` for details.

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repository
4. Add secrets in App Settings → Secrets
5. Deploy!

See `rag_retrieval/GITHUB_SETUP.md` for step-by-step instructions.

## 📊 Example Queries

```
"Show me credit cards with cashback rewards"
"Find ₹500 food vouchers"
"Best card for fuel expenses"
"Pizza delivery gift cards"
"Cards with airport lounge access"
```

## 🎨 Features in Detail

### Intelligent Query Processing
- **Auto-detection**: Extracts brands and denominations
- **Query Reframing**: Optimizes search queries
- **Context-aware**: Understands product categories

### Dual Scoring System
- **Semantic Score**: Vector similarity (meaning-based)
- **Keyword Score**: Lexical matching (term-based)
- **Combined Score**: Weighted combination

### Flexible Search
- Brand filtering
- Denomination filtering
- Configurable similarity metrics
- Adjustable result counts

## 💰 Cost Estimation

Azure OpenAI usage (approximate):
- **One-time**: $0.50-2.00 (building vector store)
- **Per query**: $0.01-0.10
- Embeddings: ~$0.13 per 1M tokens
- Chat: ~$0.15 per 1M tokens

## 🛠️ Development

### Run Tests

```bash
cd rag_retrieval
python test_similarity_metrics.py
python test_filter_extraction.py
python test_api_connection.py
```

### Update Configuration

No redeployment needed! Just update `secrets.toml` and restart the app.

## 📚 Documentation

- [`CONFIGURATION_GUIDE.md`](rag_retrieval/CONFIGURATION_GUIDE.md) - All configuration options
- [`SECURITY.md`](rag_retrieval/SECURITY.md) - Credential management
- [`SIMILARITY_METRICS.md`](rag_retrieval/SIMILARITY_METRICS.md) - Understanding metrics
- [`GITHUB_SETUP.md`](rag_retrieval/GITHUB_SETUP.md) - Deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- Azure OpenAI for LLM capabilities
- FAISS for vector search
- Streamlit for UI framework

## 📧 Support

For issues or questions:
- Check documentation files
- Create an issue on GitHub
- Review example queries

---

**Made with ❤️ using Azure OpenAI and Streamlit**
