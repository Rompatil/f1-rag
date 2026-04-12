"""
Structured logging for the F1 RAG system.
Logs queries, retrieval results, and generation output for debugging + evaluation.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import settings


def setup_logger(name: str = "f1_rag") -> logging.Logger:
    """Create a logger that writes structured JSON to file + readable text to console."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console: human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    # File: structured JSON lines
    log_file = settings.paths.logs / "rag.jsonl"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, default=str)


def log_query(
    logger: logging.Logger,
    query: str,
    retrieved_chunks: int,
    top_score: float,
    answer_len: int,
    confidence: float,
    latency_ms: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a complete query cycle for evaluation."""
    record = logger.makeRecord(
        logger.name, logging.INFO,
        "(query)", 0, f"Query processed: {query[:80]}...", (), None,
    )
    record.data = {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "top_score": round(top_score, 4),
        "answer_length": answer_len,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
        **(extra or {}),
    }
    logger.handle(record)


logger = setup_logger()
