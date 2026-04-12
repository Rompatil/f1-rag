#!/usr/bin/env python3
"""
Pre-build script — run this locally before deploying.

This builds the FAISS index using sentence-transformers (requires torch).
The built index is saved to data/vector_store/ and committed to your repo.
In production, the server loads this pre-built index — no torch needed.

Usage:
    pip install -r requirements-local.txt    # One-time: install torch + sentence-transformers
    python build_index.py                     # Build the index
    git add data/vector_store/                # Commit the index
    # Then deploy
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import pipeline
from utils.logger import logger


def main():
    print("=" * 60)
    print("  F1 RAG — Building FAISS Index")
    print("  This runs sentence-transformers locally (requires torch)")
    print("=" * 60)
    print()

    count = pipeline.ingest()

    if count > 0:
        print()
        print(f"✅ Built index with {count} chunks")
        print(f"   Saved to: data/vector_store/")
        print()
        print("Next steps:")
        print("  1. git add data/vector_store/")
        print("  2. git commit -m 'Add pre-built FAISS index'")
        print("  3. Deploy (Railway/Render/Fly.io)")
        print()
        print("In production, set these env vars:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  VOYAGE_API_KEY=pa-...  (for query embeddings)")
    else:
        print("❌ Failed to build index. Check your data/ directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
