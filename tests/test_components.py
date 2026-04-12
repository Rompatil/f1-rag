"""Tests for embeddings, vector store, cache, and API models."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from ingestion.chunker import Chunk
from embeddings.vector_store import VectorStore
from utils.cache import QueryCache
from api.models import QueryRequest, QueryResponse, SourceSnippet


# ---- Vector Store (tested with synthetic embeddings) ----

@pytest.fixture
def sample_chunks():
    return [
        Chunk("c1", "race_result", "Max Verstappen won the 2023 British GP", {"year": 2023}),
        Chunk("c2", "driver_career", "Lewis Hamilton 7 time world champion", {"year": 2020}),
        Chunk("c3", "circuit", "Silverstone Circuit UK 18 turns", {"country": "UK"}),
        Chunk("c4", "championship", "2023 Drivers Championship Verstappen", {"year": 2023}),
        Chunk("c5", "constructor_season", "Red Bull 2023 season 21 wins", {"year": 2023}),
    ]


@pytest.fixture
def vector_store(sample_chunks):
    dim = 8
    vs = VectorStore(dimension=dim)
    # Create synthetic normalized embeddings
    rng = np.random.RandomState(42)
    embeddings = rng.randn(len(sample_chunks), dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    vs.build(sample_chunks, embeddings)
    return vs


class TestVectorStore:
    def test_build(self, vector_store):
        assert vector_store.size == 5

    def test_search_returns_results(self, vector_store):
        query = np.random.randn(1, 8).astype(np.float32)
        query /= np.linalg.norm(query)
        results = vector_store.search(query, top_k=3, score_threshold=-1.0)
        assert len(results) == 3

    def test_search_respects_top_k(self, vector_store):
        query = np.random.randn(1, 8).astype(np.float32)
        query /= np.linalg.norm(query)
        results = vector_store.search(query, top_k=2, score_threshold=-1.0)
        assert len(results) == 2

    def test_search_category_filter(self, vector_store):
        query = np.random.randn(1, 8).astype(np.float32)
        query /= np.linalg.norm(query)
        results = vector_store.search(query, top_k=5, score_threshold=-1.0,
                                      categories=["race_result"])
        assert all(c.category == "race_result" for c, _ in results)

    def test_save_and_load(self, vector_store, tmp_path):
        vector_store.save(tmp_path)
        vs2 = VectorStore()
        loaded = vs2.load(tmp_path)
        assert loaded is True
        assert vs2.size == vector_store.size
        assert vs2.chunks[0].chunk_id == "c1"

    def test_get_chunk_by_id(self, vector_store):
        chunk = vector_store.get_chunk_by_id("c2")
        assert chunk is not None
        assert "Hamilton" in chunk.content

    def test_empty_index_search(self):
        vs = VectorStore()
        query = np.random.randn(1, 8).astype(np.float32)
        results = vs.search(query)
        assert results == []


# ---- Cache ----

class TestCache:
    def test_put_and_get(self):
        cache = QueryCache(max_size=10, ttl=60)
        cache.put("test query", {"answer": "test"})
        assert cache.get("test query") == {"answer": "test"}

    def test_miss(self):
        cache = QueryCache(max_size=10, ttl=60)
        assert cache.get("nonexistent") is None

    def test_case_insensitive(self):
        cache = QueryCache(max_size=10, ttl=60)
        cache.put("Hello World", "result")
        assert cache.get("hello world") == "result"

    def test_eviction(self):
        cache = QueryCache(max_size=2, ttl=60)
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.put("q3", "r3")  # Should evict q1
        assert cache.get("q1") is None
        assert cache.get("q2") == "r2"
        assert cache.get("q3") == "r3"

    def test_stats(self):
        cache = QueryCache(max_size=10, ttl=60)
        cache.put("q1", "r1")
        cache.get("q1")  # hit
        cache.get("q2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


# ---- API Models ----

class TestAPIModels:
    def test_query_request_valid(self):
        req = QueryRequest(query="Who won the 2023 championship?")
        assert req.query == "Who won the 2023 championship?"

    def test_query_request_too_short(self):
        with pytest.raises(Exception):
            QueryRequest(query="ab")

    def test_query_response(self):
        resp = QueryResponse(
            answer="Verstappen won",
            sources=[SourceSnippet(chunk_id="c1", content="...", score=0.95, category="race")],
            confidence=0.9,
        )
        data = resp.model_dump()
        assert data["answer"] == "Verstappen won"
        assert data["confidence"] == 0.9
        assert len(data["sources"]) == 1

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            QueryResponse(answer="x", sources=[], confidence=1.5)
