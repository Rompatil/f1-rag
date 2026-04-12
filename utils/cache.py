"""
Query-level caching with TTL eviction.
Avoids redundant embeddings + Claude calls for repeated queries.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from utils.config import settings


@dataclass
class CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl: float) -> bool:
        return (time.time() - self.created_at) > ttl


class QueryCache:
    """Thread-safe LRU cache with TTL for RAG query results."""

    def __init__(
        self,
        max_size: int = settings.cache.max_size,
        ttl: float = settings.cache.ttl_seconds,
    ):
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> Any | None:
        key = self._key(query)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired(self._ttl):
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, query: str, value: Any) -> None:
        key = self._key(query)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = CacheEntry(value=value)
            else:
                if len(self._store) >= self._max_size:
                    self._store.popitem(last=False)
                self._store[key] = CacheEntry(value=value)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "size": len(self._store),
        }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# Singleton
query_cache = QueryCache()
