"""
Embedding Layer — two modes:

1. LOCAL mode (for ingestion/build): Uses sentence-transformers (requires torch).
   Run `python run.py --ingest` locally to build the FAISS index.

2. API mode (for production/deploy): Uses Voyage AI API (tiny dependency, no torch).
   Set VOYAGE_API_KEY env var. Only used for encoding queries at runtime.

The system auto-detects which is available.
"""

import os
import numpy as np
from utils.config import settings
from utils.logger import logger


class Embedder:
    """Wraps embedding — auto-selects local model or Voyage API."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding.model_name
        self.dimension = settings.embedding.dimension
        self._local_model = None
        self._mode = None  # "local" or "api"

    @property
    def mode(self) -> str:
        if self._mode:
            return self._mode
        # Try local first (for ingestion)
        try:
            from sentence_transformers import SentenceTransformer
            self._mode = "local"
        except ImportError:
            # Fall back to API
            if os.environ.get("VOYAGE_API_KEY"):
                self._mode = "api"
            else:
                # Last resort: try local anyway (will error clearly)
                self._mode = "local"
        logger.info(f"Embedder mode: {self._mode}")
        return self._mode

    @property
    def local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.model_name}")
            self._local_model = SentenceTransformer(self.model_name)
            test = self._local_model.encode(["test"])
            actual_dim = test.shape[1]
            if actual_dim != self.dimension:
                logger.warning(f"Model dim {actual_dim} != config {self.dimension}, updating.")
                self.dimension = actual_dim
        return self._local_model

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed a batch of texts. Used during ingestion (local mode)."""
        bs = batch_size or settings.embedding.batch_size
        logger.info(f"Embedding {len(texts)} texts (mode={self.mode}, batch={bs})")

        if self.mode == "local":
            embeddings = self.local_model.encode(
                texts, batch_size=bs,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            )
            return np.array(embeddings, dtype=np.float32)
        else:
            return self._voyage_embed(texts, input_type="document")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Used at runtime."""
        if self.mode == "local":
            embedding = self.local_model.encode([query], normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)
        else:
            return self._voyage_embed([query], input_type="query")

    def _voyage_embed(self, texts: list[str], input_type: str = "document") -> np.ndarray:
        """Call Voyage AI API for embeddings (lightweight, no torch needed)."""
        import httpx

        api_key = os.environ.get("VOYAGE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "No embedding backend available. Either:\n"
                "  1. Install sentence-transformers (pip install sentence-transformers) for local mode\n"
                "  2. Set VOYAGE_API_KEY env var for API mode"
            )

        # Voyage API — batch up to 128 texts
        all_embeddings = []
        for i in range(0, len(texts), 128):
            batch = texts[i:i+128]
            response = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "voyage-3-lite",
                    "input": batch,
                    "input_type": input_type,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            for item in data["data"]:
                all_embeddings.append(item["embedding"])

        result = np.array(all_embeddings, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        result = result / norms

        # Update dimension on first call
        if result.shape[1] != self.dimension:
            self.dimension = result.shape[1]

        return result
