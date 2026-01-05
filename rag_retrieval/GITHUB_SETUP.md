# GitHub Setup and Push Guide

## Step-by-Step Instructions to Push Your Code to GitHub

### 1. Create a GitHub Repository

1. Go to https://github.com
2. Click **"New repository"** or **"+"** → **"New repository"**
3. Fill in:
   - **Repository name**: `product-catalog-rag-search` (or your choice)
   - **Description**: "AI-powered product catalog search using Azure OpenAI and RAG"
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** initialize with README (we already have one)
4. Click **"Create repository"**
5. **Copy the repository URL** (e.g., `https://github.com/yourusername/product-catalog-rag-search.git`)

### 2. Initialize Git and Push (Run these commands)

Open Terminal in the `rag_retrieval` folder and run:

```bash
# Navigate to your project folder
cd "/Users/amarjagtap/Documents/untitled folder 3/excel_to_json_parser/rag_retrieval"

# Initialize Git repository
git init

# Add all files (secrets are already protected by .gitignore)
git add .

# Verify secrets.toml is NOT being added
git status
# You should NOT see .streamlit/secrets.toml in the list

# Create first commit
git commit -m "Initial commit: RAG-based product catalog search with Azure OpenAI

Features:
- Semantic search using vector embeddings
- Cosine and semantic similarity metrics
- Brand and denomination filtering
- Query reframing and optimization
- Streamlit web interface
- Secure credential management via secrets
- Configurable without redeployment"

# Add your GitHub repository as remote (REPLACE WITH YOUR URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Verify Upload

After pushing, go to your GitHub repository URL and verify:
- ✅ All code files are present
- ✅ `.streamlit/secrets.toml` is **NOT** visible (protected by .gitignore)
- ✅ README.md shows up on the main page
- ✅ Documentation files are included

### 4. Add Repository Description (Optional)

On GitHub:
1. Go to your repository
2. Click **"About"** (gear icon)
3. Add:
   - **Description**: "AI-powered product catalog search using Azure OpenAI RAG pipeline"
   - **Topics**: `azure-openai`, `rag`, `streamlit`, `vector-search`, `semantic-search`, `python`
   - **Website**: Your deployed Streamlit app URL (if deployed)

### 5. Security Check

Before making repository public:

```bash
# Check what's being tracked
git ls-files | grep -i secret
# Should return: .streamlit/secrets.toml.example (template only)
# Should NOT return: .streamlit/secrets.toml (your real credentials)

# Verify .gitignore is working
cat .gitignore | grep secrets
# Should show: .streamlit/secrets.toml
```

### Quick Command Reference

```bash
# Check status
git status

# See what would be committed
git diff --staged

# Add more files later
git add <filename>
git commit -m "Description of changes"
git push

# Pull latest changes
git pull

# View commit history
git log --oneline
```

### Common Issues

**Issue**: "failed to push some refs"
**Solution**: 
```bash
git pull origin main --rebase
git push -u origin main
```

**Issue**: "Repository not found"
**Solution**: Check the remote URL:
```bash
git remote -v
# If wrong, update it:
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

**Issue**: Accidentally committed secrets
**Solution**: 
1. Rotate your Azure OpenAI API key immediately
2. Remove from history (see SECURITY.md)
3. Push updated .gitignore

### Next Steps After Pushing

1. ✅ **Add Collaborators** (if team project)
   - Settings → Collaborators → Add people

2. ✅ **Enable GitHub Actions** (optional)
   - For automated testing or deployment

3. ✅ **Deploy to Streamlit Cloud**
   - Connect your GitHub repo
   - Add secrets in Streamlit Cloud settings

4. ✅ **Add Branch Protection** (recommended)
   - Settings → Branches → Add rule
   - Require pull requests for main branch

## Complete One-Command Setup

Save this as `push_to_github.sh` and run it:

```bash
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}GitHub Repository Setup${NC}"
echo ""

# Get GitHub repository URL
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo -e "${RED}Error: Repository URL is required${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 1: Initializing Git repository...${NC}"
git init

echo -e "${GREEN}Step 2: Adding files...${NC}"
git add .

echo -e "${YELLOW}Files to be committed:${NC}"
git status --short

echo ""
echo -e "${YELLOW}⚠️  Checking for secrets.toml (should NOT be listed above)...${NC}"
if git ls-files | grep -q "secrets.toml$"; then
    echo -e "${RED}ERROR: secrets.toml is being tracked! Aborting.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ secrets.toml is properly ignored${NC}"
fi

echo ""
read -p "Continue with commit? (y/n): " CONTINUE

if [ "$CONTINUE" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo -e "${GREEN}Step 3: Creating initial commit...${NC}"
git commit -m "Initial commit: RAG-based product catalog search

Features:
- Semantic search using vector embeddings
- Cosine and semantic similarity metrics  
- Brand and denomination filtering
- Query reframing and optimization
- Streamlit web interface
- Secure credential management
- Configurable without redeployment"

echo -e "${GREEN}Step 4: Adding remote repository...${NC}"
git remote add origin "$REPO_URL"

echo -e "${GREEN}Step 5: Pushing to GitHub...${NC}"
git branch -M main
git push -u origin main

echo ""
echo -e "${GREEN}✓ Successfully pushed to GitHub!${NC}"
echo ""
echo "Next steps:"
echo "1. Visit: $REPO_URL"
echo "2. Verify files are uploaded"
echo "3. Check .streamlit/secrets.toml is NOT visible"
echo "4. Deploy to Streamlit Cloud (optional)"
```

Make it executable and run:
```bash
chmod +x push_to_github.sh
./push_to_github.sh
```

## Summary

Your code is now ready to be pushed to GitHub with:
- ✅ Secrets protected by .gitignore
- ✅ Complete documentation
- ✅ Professional README
- ✅ Security guidelines
- ✅ Configuration management

**Your Azure OpenAI credentials will remain private!** 🔒
