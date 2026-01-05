# Configuration Management Guide

## Overview

The application now supports **flexible configuration** that can be changed **without redeploying**:

- ✅ Environment variables
- ✅ Streamlit secrets
- ✅ Default values (fallback)

## Configuration Priority

The system reads configuration in this order:

1. **Environment Variables** (highest priority)
2. **Streamlit Secrets** (`.streamlit/secrets.toml` or Streamlit Cloud secrets)
3. **Default Values** (hardcoded in `config.py`)

## Local Development

### Method 1: Create `.streamlit/secrets.toml`

```bash
cd rag_retrieval
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your values
```

### Method 2: Use Environment Variables

```bash
export SIMILARITY_METRIC="semantic"
export SEMANTIC_WEIGHT="0.8"
export SIMILARITY_WEIGHT="0.2"
streamlit run app.py
```

## Streamlit Cloud Deployment

### Setup Secrets (One-time)

1. Deploy your app to Streamlit Cloud
2. Go to **App Settings** → **Secrets**
3. Add your configuration:

```toml
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key-here"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

# Similarity metric: "cosine" or "semantic"
SIMILARITY_METRIC = "cosine"

# Scoring weights
SEMANTIC_WEIGHT = "0.7"
SIMILARITY_WEIGHT = "0.3"

# Other settings...
TOP_K_RESULTS = "10"
TEMPERATURE = "0"
MAX_TOKENS = "1000"
```

4. Click **Save**

### Change Configuration (No Redeployment Needed!)

1. Go to **App Settings** → **Secrets**
2. Update the values (e.g., change `SIMILARITY_METRIC = "semantic"`)
3. Click **Save**
4. **App will automatically restart** with new config ✅

## Available Configuration Options

### Similarity Metric
```toml
SIMILARITY_METRIC = "cosine"    # or "semantic"
```

### Search Settings
```toml
TOP_K_RESULTS = "10"           # Number of results to return
```

### Scoring Weights
```toml
SEMANTIC_WEIGHT = "0.7"        # Vector similarity weight
SIMILARITY_WEIGHT = "0.3"      # Keyword similarity weight
```

### LLM Settings
```toml
TEMPERATURE = "0"              # LLM temperature (0-1)
MAX_TOKENS = "1000"           # Max response tokens
MAX_CONTEXT_LENGTH = "4000"   # Max context length
```

### Azure OpenAI Settings
```toml
AZURE_OPENAI_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-key"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
CHAT_DEPLOYMENT = "gpt-4o-mini"
```

## Testing Configuration Changes

### Test Locally
```bash
# Change similarity metric
export SIMILARITY_METRIC="semantic"
streamlit run app.py
```

### Test on Streamlit Cloud
1. Update secrets in app settings
2. Save changes
3. App restarts automatically
4. Test immediately - no redeployment needed!

## Benefits

✅ **No Redeployment**: Change config without redeploying  
✅ **A/B Testing**: Easily switch between cosine vs semantic  
✅ **Security**: API keys in secrets, not in code  
✅ **Flexibility**: Override defaults per environment  
✅ **Quick Tuning**: Adjust weights and parameters on the fly  

## Security Best Practices

1. **Never commit** `.streamlit/secrets.toml` to git
2. Add to `.gitignore`:
   ```
   .streamlit/secrets.toml
   ```
3. Use Streamlit Cloud secrets for production
4. Rotate API keys regularly

## Examples

### Switch to Semantic Similarity
```toml
SIMILARITY_METRIC = "semantic"
```

### Increase Semantic Weight
```toml
SEMANTIC_WEIGHT = "0.8"
SIMILARITY_WEIGHT = "0.2"
```

### More Results
```toml
TOP_K_RESULTS = "20"
```

### More Creative Responses
```toml
TEMPERATURE = "0.7"
MAX_TOKENS = "1500"
```

## Troubleshooting

### Config not updating?
- Verify secrets syntax (valid TOML)
- Check spelling of variable names
- Save changes in Streamlit Cloud
- App should auto-restart (check logs)

### Default values used?
- Check secrets are properly set
- Verify variable names match exactly
- For numbers/booleans, use strings in secrets.toml

## Summary

You can now **change configuration without redeploying**! 🎉

- Update secrets in Streamlit Cloud settings
- App automatically restarts with new config
- Perfect for tuning similarity metrics, weights, and parameters
