"""
RAG Pipeline — orchestrates the full ingestion → retrieval → generation flow.
Single entry point for both CLI and API usage.
"""

import time
from pathlib import Path

from ingestion.loader import F1DataLoader
from ingestion.chunker import F1Chunker, Chunk
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from retrieval.retriever import Retriever
from generation.generator import Generator
from api.models import QueryResponse, SourceSnippet
from utils.config import settings
from utils.cache import query_cache
from utils.logger import logger, log_query


class RAGPipeline:
    """Full RAG pipeline: ingest, embed, retrieve, generate."""

    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.generator = Generator()
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self.vector_store.size > 0

    def ingest(self, data_dir: Path | None = None) -> int:
        t0 = time.time()
        loader = F1DataLoader(data_dir)
        loader.load_all()

        chunker = F1Chunker(loader)
        chunks = chunker.generate_all_chunks()
        if not chunks:
            logger.error("No chunks generated — check data files")
            return 0

        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.build(chunks, embeddings)
        self.vector_store.save()

        elapsed = time.time() - t0
        logger.info(f"Ingestion complete: {len(chunks)} chunks in {elapsed:.1f}s")
        self._ready = True
        return len(chunks)

    def load_index(self) -> bool:
        loaded = self.vector_store.load()
        if loaded:
            self._ready = True
            logger.info(f"Index loaded: {self.vector_store.size} chunks ready")
        return loaded

    def query(self, query: str) -> QueryResponse:
        if not self.is_ready:
            return QueryResponse(
                answer="System not ready. Run ingestion first.",
                sources=[], confidence=0.0,
            )

        cached = query_cache.get(query)
        if cached is not None:
            logger.info(f"Cache hit for: '{query[:50]}'")
            cached.cached = True
            return cached

        t0 = time.time()

        results = self.retriever.retrieve(query)
        context = self.retriever.build_context(results)
        gen_result = self.generator.generate(query, context)

        sources = [
            SourceSnippet(
                chunk_id=chunk.chunk_id,
                content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                score=round(score, 4),
                category=chunk.category,
            )
            for chunk, score in results
        ]

        response = QueryResponse(
            answer=gen_result["answer"],
            sources=sources,
            confidence=gen_result["confidence"],
            cached=False,
        )

        query_cache.put(query, response)

        elapsed_ms = (time.time() - t0) * 1000
        log_query(
            logger, query=query,
            retrieved_chunks=len(results),
            top_score=results[0][1] if results else 0.0,
            answer_len=len(gen_result["answer"]),
            confidence=gen_result["confidence"],
            latency_ms=elapsed_ms,
        )
        return response


pipeline = RAGPipeline()
