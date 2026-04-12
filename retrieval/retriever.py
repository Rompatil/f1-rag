"""
Retrieval Layer — takes a user query, embeds it, and retrieves
the most relevant chunks from the vector store.
"""

from ingestion.chunker import Chunk
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from utils.config import settings
from utils.logger import logger


class Retriever:
    """Retrieves relevant F1 data chunks for a given query."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        categories: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            categories=categories,
        )
        if results:
            logger.debug(f"Retrieved {len(results)} chunks for: '{query[:60]}' (top={results[0][1]:.4f})")
        else:
            logger.debug(f"No results for: '{query[:60]}'")
        return results

    def build_context(
        self,
        results: list[tuple[Chunk, float]],
        max_tokens: int | None = None,
    ) -> str:
        budget = max_tokens or settings.retrieval.max_context_tokens
        context_parts: list[str] = []
        current_tokens = 0

        for chunk, score in results:
            chunk_tokens = len(chunk.content) // 4
            if current_tokens + chunk_tokens > budget:
                break
            context_parts.append(
                f"[Source: {chunk.chunk_id} | Category: {chunk.category} | "
                f"Relevance: {score:.3f}]\n{chunk.content}"
            )
            current_tokens += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Built context: {len(context_parts)} chunks, ~{current_tokens} tokens")
        return context
