#!/usr/bin/env python3
"""
F1 RAG System — Entry Point

Usage:
    python run.py --ingest          # Ingest data + build index
    python run.py --serve           # Start API server
    python run.py --interactive     # Interactive query mode (no API)
    python run.py --ingest --serve  # Ingest then serve
"""

import argparse
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger


def run_ingest():
    from pipeline import pipeline
    logger.info("Starting ingestion pipeline...")
    count = pipeline.ingest()
    if count > 0:
        logger.info(f"Ingestion successful: {count} chunks indexed")
    else:
        logger.error("Ingestion produced 0 chunks. Check your data/ directory.")
        sys.exit(1)


def run_serve(host: str = "0.0.0.0", port: int = None):
    import uvicorn
    port = port or int(os.environ.get("PORT", 8000))
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


def run_interactive():
    from pipeline import pipeline

    # Load or ingest
    if not pipeline.load_index():
        logger.info("No index found. Running ingestion first...")
        pipeline.ingest()

    print("\n" + "=" * 60)
    print("  F1 RAG — Interactive Mode")
    print("  Type your F1 questions. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("🏎️  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = pipeline.query(query)

        print(f"\n📋 Answer (confidence: {response.confidence:.0%}):\n")
        print(response.answer)
        print(f"\n📎 Sources ({len(response.sources)}):")
        for s in response.sources[:5]:
            print(f"   [{s.category}] {s.chunk_id} (score: {s.score:.3f})")
        print(f"   Cached: {response.cached}")
        print()


def main():
    parser = argparse.ArgumentParser(description="F1 RAG System")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion pipeline")
    parser.add_argument("--serve", action="store_true", help="Start the API server")
    parser.add_argument("--interactive", action="store_true", help="Interactive query mode")
    parser.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    args = parser.parse_args()

    if not any([args.ingest, args.serve, args.interactive]):
        parser.print_help()
        print("\nExample:")
        print("  python run.py --ingest          # First time: build index")
        print("  python run.py --serve           # Start API")
        print("  python run.py --interactive     # Chat mode")
        sys.exit(0)

    if args.ingest:
        run_ingest()

    if args.serve:
        run_serve(args.host, args.port)
    elif args.interactive:
        run_interactive()


if __name__ == "__main__":
    main()
