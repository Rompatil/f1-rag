"""
FastAPI server — exposes the RAG pipeline as a REST API.

Endpoints:
    POST /query         — ask an F1 question (full response)
    POST /query/stream  — ask with Server-Sent Events (text appears live)
    GET  /health        — system status
    POST /ingest        — trigger re-ingestion
    GET  /examples      — sample queries
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

from api.models import QueryRequest, QueryResponse, HealthResponse, SourceSnippet
from utils.cache import query_cache
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    from pipeline import pipeline
    app.state.pipeline = pipeline

    if not pipeline.load_index():
        # In production, we expect a pre-built index
        # Only auto-ingest if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("No saved index found, running ingestion...")
            count = pipeline.ingest()
            logger.info(f"Ingested {count} chunks on startup")
        except ImportError:
            logger.error(
                "No pre-built index found and sentence-transformers not installed. "
                "Run 'python build_index.py' locally first, then commit data/vector_store/"
            )
    else:
        logger.info(f"Loaded existing index with {pipeline.vector_store.size} chunks")

    yield
    pipeline.generator.close()


app = FastAPI(
    title="F1 RAG API",
    description="Retrieval-Augmented Generation for Formula 1 data",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/query", response_model=QueryResponse)
async def query_f1(request: QueryRequest):
    """Full (non-streaming) query."""
    try:
        pipeline = app.state.pipeline
        t0 = time.time()
        response = pipeline.query(request.query)
        response.query = request.query
        response.latency_ms = round((time.time() - t0) * 1000, 1)
        response.retrieval_count = len(response.sources)
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_f1_stream(request: QueryRequest):
    """
    Streaming query — returns Server-Sent Events.
    
    Events:
      event: sources   — JSON array of retrieved sources
      event: token     — a text chunk from Claude
      event: done      — final metadata (confidence, latency)
    """
    pipeline = app.state.pipeline

    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="System not ready")

    # Check cache first
    cached = query_cache.get(request.query)
    if cached is not None:
        async def cached_stream():
            yield f"event: sources\ndata: {json.dumps([s.model_dump() for s in cached.sources])}\n\n"
            yield f"event: token\ndata: {json.dumps(cached.answer)}\n\n"
            yield f"event: done\ndata: {json.dumps({'confidence': cached.confidence, 'cached': True, 'latency_ms': 0})}\n\n"
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    t0 = time.time()

    # Retrieve
    results = pipeline.retriever.retrieve(request.query)
    context = pipeline.retriever.build_context(results)

    sources = [
        SourceSnippet(
            chunk_id=chunk.chunk_id,
            content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            score=round(score, 4),
            category=chunk.category,
        )
        for chunk, score in results
    ]

    def event_stream():
        # Send sources immediately (user sees them while Claude thinks)
        yield f"event: sources\ndata: {json.dumps([s.model_dump() for s in sources])}\n\n"

        # Stream Claude's response
        full_answer = ""
        confidence = 0.5

        for chunk in pipeline.generator.generate_stream(request.query, context):
            if chunk.startswith("\n__META__:"):
                confidence = float(chunk.split(":")[1])
            else:
                full_answer += chunk
                yield f"event: token\ndata: {json.dumps(chunk)}\n\n"

        # Strip confidence line from displayed text
        import re
        answer_clean = re.sub(r"\n*CONFIDENCE:\s*\[?(high|medium|low)\]?\s*$", "", full_answer, flags=re.IGNORECASE).strip()

        elapsed = round((time.time() - t0) * 1000, 1)

        # Cache the full response
        response = QueryResponse(
            answer=answer_clean, sources=sources,
            confidence=confidence, cached=False,
            query=request.query, latency_ms=elapsed,
            retrieval_count=len(sources),
        )
        query_cache.put(request.query, response)

        # Final event
        yield f"event: done\ndata: {json.dumps({'confidence': confidence, 'cached': False, 'latency_ms': elapsed})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health", response_model=HealthResponse)
async def health():
    pipeline = app.state.pipeline
    categories = {}
    if pipeline.vector_store.chunks:
        categories = dict(Counter(c.category for c in pipeline.vector_store.chunks))
    return HealthResponse(
        status="ready" if pipeline.is_ready else "not_ready",
        chunks_indexed=pipeline.vector_store.size,
        cache_stats=query_cache.stats,
        chunk_categories=categories,
    )


@app.post("/ingest")
async def ingest():
    try:
        pipeline = app.state.pipeline
        count = pipeline.ingest()
        return {"status": "ok", "chunks_indexed": count}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples")
async def example_queries():
    return {
        "categories": [
            {"name": "Race Results", "icon": "🏁", "queries": [
                "Who won the 2021 Abu Dhabi Grand Prix?",
                "What was the podium at the 2023 British Grand Prix?",
                "Who won the 2024 Italian Grand Prix at Monza?",
            ]},
            {"name": "Driver Analysis", "icon": "👤", "queries": [
                "Compare Lewis Hamilton and Max Verstappen careers",
                "How did Charles Leclerc perform in 2022?",
                "What are Fernando Alonso's career stats?",
            ]},
            {"name": "Team Performance", "icon": "🏎️", "queries": [
                "How dominant was Red Bull in 2023?",
                "Compare Ferrari and McLaren in 2024",
                "Which teams won constructors championships from 2010-2023?",
            ]},
            {"name": "Championships", "icon": "🏆", "queries": [
                "Who won the 2021 drivers championship?",
                "List all world champions from 2010 to 2023",
                "How close was the 2021 title fight?",
            ]},
            {"name": "Circuits", "icon": "🗺️", "queries": [
                "Tell me about the Silverstone circuit",
                "Which circuits are street circuits?",
                "What is the Yas Marina circuit in Abu Dhabi?",
            ]},
        ]
    }
