# F1 RAG Backend — Retrieval-Augmented Generation for Formula 1 Data

A production-ready RAG system that answers Formula 1 questions by retrieving relevant data from a vector store and generating grounded, hallucination-free responses using Claude (Anthropic API).

## Architecture

```
                           ┌─────────────────────────────────┐
                           │         FastAPI Server           │
                           │   POST /query  GET /health       │
                           └──────────┬──────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
               ┌────▼────┐    ┌──────▼──────┐   ┌─────▼─────┐
               │  Cache   │    │  Retriever  │   │  Logger   │
               │ (LRU+TTL)│    │  (Top-K)    │   │ (JSON)    │
               └──────────┘    └──────┬──────┘   └───────────┘
                                      │
                          ┌───────────┼───────────┐
                          │                       │
                   ┌──────▼──────┐        ┌───────▼───────┐
                   │  Embedder   │        │  FAISS Index  │
                   │ (MiniLM)    │        │ (Vector Store)│
                   └─────────────┘        └───────────────┘
                                                  │
                                          ┌───────▼───────┐
                                          │    Claude      │
                                          │  (Generation)  │
                                          └───────────────┘
```

### Pipeline

1. **Ingestion** — Load Ergast-schema CSVs → join relationally → generate 6 types of semantic chunks (race results, driver seasons, constructor seasons, circuits, championships, driver careers)
2. **Embedding** — Encode chunks with `all-MiniLM-L6-v2` → store in FAISS (cosine similarity)
3. **Retrieval** — Embed user query → top-k vector search → assemble context with token budgeting
4. **Generation** — Claude generates grounded answer using only retrieved context → returns structured response with sources + confidence

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your Anthropic API key
cp .env.example .env
# Edit .env with your key

# Fetch real F1 data (1995-2025, ~2-3 min)
python fetch_data.py

# Build index + start server with interactive frontend
python run.py --ingest --serve
# Open http://localhost:8000

# Or use CLI chat mode
python run.py --interactive
```

## API

### POST /query
```json
// Request
{ "query": "Who won the 2023 British Grand Prix?" }

// Response
{
  "answer": "Max Verstappen won the 2023 British Grand Prix, driving for Red Bull Racing...",
  "sources": [
    { "chunk_id": "race_1124", "content": "Race: British Grand Prix (2023)...", "score": 0.89, "category": "race_result" }
  ],
  "confidence": 0.95,
  "cached": false
}
```

### GET /health
```json
{ "status": "ready", "chunks_indexed": 152, "cache_stats": { "hits": 5, "misses": 12, "hit_rate": 0.294, "size": 12 } }
```

### POST /ingest
Re-ingests all data from `data/raw/`.

## Supported Query Types

- **Race results** — "Who won the 2021 Abu Dhabi GP?"
- **Driver comparisons** — "Compare Hamilton and Verstappen careers"
- **Team performance** — "How did Red Bull perform in 2023?"
- **Historical dominance** — "Which team dominated 2010–2013?"
- **Circuit info** — "Tell me about Silverstone"
- **Season summaries** — "Who won the 2022 championship?"

## Data

Ships with Ergast-schema sample data. To load the **full real dataset** (1995–2025, every race, every result):

```bash
python fetch_data.py                      # ~2-3 min, fetches everything
python fetch_data.py --seasons 2000-2025  # custom range
python fetch_data.py --workers 8          # more parallelism
python run.py --ingest                    # rebuild index
```

The fetcher handles **pagination properly** (the API caps at 100 items/page — a 24-race season has ~480 results) and uses **concurrent requests** (4 workers by default) for speed. Data comes from the [Jolpica API](https://github.com/jolpica/jolpica-f1) (free Ergast successor).

### Schema

| File | Rows (sample) | Content |
|------|--------|---------|
| `drivers.csv` | 28 | Driver bios, nationality, DOB |
| `constructors.csv` | 16 | Team names, nationality |
| `circuits.csv` | 25 | Circuit location, coordinates |
| `races.csv` | 50 | Race calendar 2010–2024 |
| `results.csv` | 81 | Race results with positions, times, fastest laps |
| `driver_standings.csv` | 64 | End-of-season driver standings |
| `constructor_standings.csv` | 37 | End-of-season constructor standings |

## Project Structure

```
f1-rag/
├── api/
│   ├── server.py          # FastAPI endpoints
│   └── models.py          # Pydantic request/response schemas
├── ingestion/
│   ├── loader.py          # CSV loading + relational joins
│   └── chunker.py         # Semantic chunk generation (6 types)
├── embeddings/
│   ├── embedder.py        # sentence-transformers wrapper
│   └── vector_store.py    # FAISS index with save/load
├── retrieval/
│   └── retriever.py       # Top-k search + context assembly
├── generation/
│   └── generator.py       # Claude API integration + grounding
├── utils/
│   ├── config.py          # Centralized settings
│   ├── cache.py           # LRU cache with TTL
│   └── logger.py          # Structured JSON logging
├── tests/
│   ├── test_ingestion.py  # 13 tests for loader + chunker
│   └── test_components.py # 16 tests for vector store, cache, models
├── data/raw/              # Ergast-schema CSVs
├── pipeline.py            # Orchestrates full RAG flow
├── run.py                 # CLI entry point
├── fetch_data.py          # Downloads real F1 data from Jolpica API
└── requirements.txt
```

## Key Design Decisions

- **Semantic chunking over raw rows** — instead of embedding individual CSV rows, the system creates meaningful text (race summaries, driver profiles, season stats) that embeds well and retrieves accurately
- **Grounded generation** — Claude's system prompt enforces that answers use only retrieved context, with a self-assessed confidence score
- **Pre-normalized embeddings + FAISS IndexFlatIP** — cosine similarity via inner product on normalized vectors, no approximate search overhead at this scale
- **Save/load persistence** — FAISS index + chunk metadata are saved to disk, so re-embedding isn't needed on restart
- **LRU cache with TTL** — repeated queries skip embedding + Claude API calls entirely

## Testing

```bash
pytest tests/ -v
# 29 tests, all passing
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI |
| LLM | Claude (Anthropic API) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| Data | Ergast/Jolpica F1 dataset |

## License

MIT
