#!/usr/bin/env python
"""
Entrypoint shortcut to seed PostgreSQL and Neo4j for CKAN Movie Recommender System.
Usage:
    uv run python seed.py
    uv run python seed.py --postgres-only
    uv run python seed.py --neo4j-only
    uv run python seed.py --clean
"""
from app.scripts.seed import main

if __name__ == "__main__":
    main()

