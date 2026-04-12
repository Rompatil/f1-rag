# Deployment Guide

## The Problem
sentence-transformers + torch = ~5GB. Serverless platforms (Vercel, Railway free tier) cap at 500MB.

## The Solution
Build the FAISS index locally (where torch is fine), then deploy only the lightweight runtime.

## Step-by-Step

### 1. Build locally (one-time)
```bash
# Install full deps (includes torch)
pip install -r requirements-local.txt

# Fetch real F1 data (optional — sample data works too)
python fetch_data.py --seasons 1995-2025

# Build the FAISS index
python build_index.py

# This creates data/vector_store/ with:
#   index.faiss  (~250KB)
#   chunks.json  (~500KB)
```

### 2. Commit the index
```bash
git add data/vector_store/
git commit -m "Add pre-built FAISS index"
git push
```

### 3. Deploy

#### Railway (recommended)
```bash
npm install -g @railway/cli
railway login
railway init
railway variables set ANTHROPIC_API_KEY=sk-ant-...
railway variables set VOYAGE_API_KEY=pa-...
railway up
```

#### Render
- Connect GitHub repo at render.com
- Add env vars: ANTHROPIC_API_KEY, VOYAGE_API_KEY
- It auto-detects render.yaml

#### Fly.io
```bash
fly launch
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly secrets set VOYAGE_API_KEY=pa-...
fly deploy
```

#### Docker (anywhere)
```bash
docker build -t f1-rag .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e VOYAGE_API_KEY=pa-... \
  f1-rag
```

### How it works in production
- Server starts, loads pre-built FAISS index from data/vector_store/
- User sends query → Voyage AI API embeds the query (lightweight HTTP call, no torch)
- FAISS searches for similar chunks
- Claude generates grounded answer
- Total runtime deps: ~50MB (fastapi + httpx + faiss-cpu + numpy + pandas)

### Env vars needed
| Variable | Required | Purpose |
|----------|----------|---------|
| ANTHROPIC_API_KEY | Yes | Claude API for answer generation |
| VOYAGE_API_KEY | Yes (production) | Query embedding via API (no torch needed) |
| PORT | Auto-set | Platform sets this automatically |

### Getting a Voyage API key
1. Go to https://dash.voyageai.com/
2. Sign up (free tier: 200M tokens/month)
3. Copy API key → set as VOYAGE_API_KEY
