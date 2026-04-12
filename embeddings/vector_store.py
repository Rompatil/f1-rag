"""
Vector Store — FAISS-based index for chunk embeddings.

Supports:
- Building index from chunks + embeddings
- Saving/loading to disk (avoids re-embedding on restart)
- Top-k similarity search with score thresholding
- Metadata filtering by category
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from ingestion.chunker import Chunk
from utils.config import settings
from utils.logger import logger


class VectorStore:
    """FAISS-backed vector store with chunk metadata."""

    def __init__(self, dimension: int | None = None):
        self.dimension = dimension or settings.embedding.dimension
        self.index: faiss.IndexFlatIP | None = None  # Inner product (cosine on normalized vecs)
        self.chunks: list[Chunk] = []
        self._id_to_idx: dict[str, int] = {}

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Build the FAISS index from chunks and their embeddings.

        Args:
            chunks: list of Chunk objects
            embeddings: numpy array of shape (len(chunks), dimension)
        """
        assert len(chunks) == embeddings.shape[0], \
            f"Chunk count {len(chunks)} != embedding count {embeddings.shape[0]}"

        self.dimension = embeddings.shape[1]
        self.chunks = chunks
        self._id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}

        # Use IndexFlatIP for cosine similarity (vectors are pre-normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info(f"Built FAISS index: {self.size} vectors, dim={self.dimension}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        score_threshold: float | None = None,
        categories: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for the most similar chunks to the query embedding.

        Args:
            query_embedding: shape (1, dimension)
            top_k: number of results to return
            score_threshold: minimum cosine similarity
            categories: optional list of categories to filter by

        Returns:
            List of (Chunk, score) tuples, sorted by score descending.
        """
        if self.index is None or self.size == 0:
            logger.warning("Search called on empty index")
            return []

        k = top_k or settings.retrieval.top_k
        threshold = score_threshold or settings.retrieval.score_threshold

        # If filtering by category, we need to search more and filter after
        search_k = min(k * 3, self.size) if categories else min(k, self.size)

        scores, indices = self.index.search(query_embedding, search_k)
        scores = scores[0]  # Flatten from (1, k) to (k,)
        indices = indices[0]

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            if score < threshold:
                continue

            chunk = self.chunks[idx]

            if categories and chunk.category not in categories:
                continue

            results.append((chunk, float(score)))

            if len(results) >= k:
                break

        return results

    def save(self, directory: Path | None = None) -> None:
        """Save index + metadata to disk."""
        save_dir = directory or settings.paths.vector_store
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "index.faiss"))

        # Save chunk metadata — custom encoder handles numpy/pandas types
        import pandas as pd

        class _Encoder(json.JSONEncoder):
            def default(self, o):
                if o is pd.NA or (isinstance(o, float) and pd.isna(o)):
                    return None
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        chunk_data = [c.to_dict() for c in self.chunks]
        with open(save_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2, cls=_Encoder)

        logger.info(f"Saved vector store to {save_dir} ({self.size} vectors)")

    def load(self, directory: Path | None = None) -> bool:
        """Load index + metadata from disk. Returns True if successful."""
        load_dir = directory or settings.paths.vector_store
        index_path = load_dir / "index.faiss"
        chunks_path = load_dir / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            logger.warning(f"No saved index found at {load_dir}")
            return False

        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        self.chunks = [Chunk(**cd) for cd in chunk_data]
        self._id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}

        logger.info(f"Loaded vector store from {load_dir} ({self.size} vectors)")
        return True

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        idx = self._id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None
