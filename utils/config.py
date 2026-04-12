"""
Central configuration for the F1 RAG system.
All tunables in one place — no magic constants scattered through code.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env — try python-dotenv first, fall back to manual parsing
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent.parent
    data_raw: Path = field(default=None)
    data_processed: Path = field(default=None)
    vector_store: Path = field(default=None)
    logs: Path = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "data_raw", self.root / "data" / "raw")
        object.__setattr__(self, "data_processed", self.root / "data" / "processed")
        object.__setattr__(self, "vector_store", self.root / "data" / "vector_store")
        object.__setattr__(self, "logs", self.root / "logs")

    def ensure_dirs(self):
        for p in [self.data_raw, self.data_processed, self.vector_store, self.logs]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # Matches all-MiniLM-L6-v2 output
    batch_size: int = 64


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 15
    score_threshold: float = 0.15
    max_context_tokens: int = 12000


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.2
    anthropic_api_key: str = field(default=None)

    def __post_init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        object.__setattr__(self, "anthropic_api_key", key)


@dataclass(frozen=True)
class CacheConfig:
    max_size: int = 512
    ttl_seconds: int = 3600  # 1 hour


@dataclass(frozen=True)
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Singleton
settings = Settings()
settings.paths.ensure_dirs()
