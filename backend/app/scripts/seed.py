import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.database import SessionLocal, engine, Base
from app.models.sql_models import User, Movie, Rating, Interaction
from app.core.neo4j_client import neo4j_client

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("seed")

# Bcrypt hash for default password: "password123"
DEFAULT_PASSWORD_HASH = "$2b$10$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"


def seed_postgres(clean: bool = False, max_users: int = 2500):
    logger.info("=" * 60)
    logger.info("  1. SEEDING POSTGRESQL DATABASE")
    logger.info(f"  Target: {settings.DATABASE_URL}")
    logger.info(f"  Max Users Target: {max_users}")
    logger.info("=" * 60)

    # Ensure tables are created
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    try:
        if clean:
            logger.warning("Cleaning existing tables (TRUNCATE CASCADE)...")
            session.execute(text("TRUNCATE TABLE interactions, recommendations, ratings, movies, users RESTART IDENTITY CASCADE;"))
            session.commit()

        # Check existing movies count
        existing_movie_count = session.query(Movie).count()
        if existing_movie_count > 0 and not clean:
            logger.info(f"PostgreSQL already contains {existing_movie_count} movies. Skipping movie seed.")
        else:
            metadata_file = Path(settings.METADATA_PATH)
            if not metadata_file.exists():
                logger.error(f"Metadata file not found at {metadata_file}!")
                return

            logger.info(f"Loading movie metadata from {metadata_file}...")
            with open(metadata_file, "r", encoding="utf-8") as f:
                raw_movies = json.load(f)

            logger.info(f"Inserting {len(raw_movies)} movies in batches...")
            batch_size = 2000
            movie_records = []
            
            # Check if movies_and_posters.csv is available
            posters_file = metadata_file.parent / "movies_and_posters.csv"
            poster_map = {}
            if posters_file.exists():
                logger.info(f"Loading posters from {posters_file}...")
                import csv
                with open(posters_file, mode="r", encoding="utf-8") as pf:
                    rdr = csv.DictReader(pf)
                    for r in rdr:
                        p_mid = r.get("movieId")
                        p_url = r.get("poster_url", "").strip()
                        if p_mid and p_url and p_url.startswith("http"):
                            try:
                                poster_map[int(p_mid)] = p_url.replace("/original/", "/w500/")
                            except ValueError:
                                pass
                logger.info(f"Loaded {len(poster_map)} posters for seeding.")

            for m in raw_movies:
                mid_lens = int(m["movie_id"])
                p_url = poster_map.get(mid_lens, f"https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60")
                movie_records.append({
                    "id": int(m["ckan_id"]),
                    "movieLensId": mid_lens,
                    "title": str(m["title"]),
                    "fullTitle": str(m.get("full_title", "")),
                    "releaseYear": m.get("release_year"),
                    "genres": m.get("genres", []),
                    "posterUrl": p_url,
                })

            for i in tqdm(range(0, len(movie_records), batch_size), desc="PostgreSQL Movies"):
                chunk = movie_records[i : i + batch_size]
                stmt = pg_insert(Movie).values(chunk).on_conflict_do_nothing(index_elements=["id"])
                session.execute(stmt)
                session.commit()

            logger.info(f"Successfully seeded {len(movie_records)} movies into PostgreSQL.")

        # Seed Benchmark Users (IDs 0..max_users-1)
        existing_user_count = session.query(User).count()
        if existing_user_count > 0 and not clean:
            logger.info(f"PostgreSQL already contains {existing_user_count} users. Skipping user seed.")
        else:
            logger.info(f"Seeding {max_users} users (IDs 0..{max_users-1}) with default password 'password123'...")
            user_records = []
            for uid in range(max_users):
                user_records.append({
                    "id": uid,
                    "email": f"user{uid}@moviekg.ai",
                    "name": f"Movie Fan #{uid}",
                    "passwordHash": DEFAULT_PASSWORD_HASH,
                })
            # Add demo user
            user_records.append({
                "id": max_users,
                "email": "demo@moviekg.ai",
                "name": "Demo Cinephile",
                "passwordHash": DEFAULT_PASSWORD_HASH,
            })

            batch_size = 500
            for i in tqdm(range(0, len(user_records), batch_size), desc="PostgreSQL Users"):
                chunk = user_records[i : i + batch_size]
                stmt = pg_insert(User).values(chunk).on_conflict_do_nothing(index_elements=["id"])
                session.execute(stmt)
                session.commit()

            logger.info(f"Successfully seeded {len(user_records)} users into PostgreSQL.")

        # Seed full ratings & interactions
        ratings_path = Path(settings.DATA_DIR) / "ratings_final.npy"
        if ratings_path.exists():
            existing_ratings_count = session.query(Rating).count()
            if existing_ratings_count > 0 and not clean:
                logger.info(f"PostgreSQL already contains {existing_ratings_count} ratings. Skipping ratings seed.")
            else:
                logger.info(f"Loading full ratings dataset from {ratings_path}...")
                ratings_arr = np.load(ratings_path)
                
                rating_records = []
                interaction_records = []
                
                for u, m, r in ratings_arr:
                    u_int, m_int, r_int = int(u), int(m), int(r)
                    if u_int < max_users:
                        is_like = (r_int == 1)
                        rating_records.append({
                            "userId": u_int,
                            "movieId": m_int,
                            "rating": 4.5 if is_like else 2.0,
                        })
                        interaction_records.append({
                            "userId": u_int,
                            "movieId": m_int,
                            "type": "LIKE" if is_like else "DISLIKE",
                        })

                batch_size = 5000
                logger.info(f"Inserting {len(rating_records)} ratings (LIKE & DISLIKE)...")
                for i in tqdm(range(0, len(rating_records), batch_size), desc="PostgreSQL Ratings"):
                    chunk = rating_records[i : i + batch_size]
                    stmt = pg_insert(Rating).values(chunk).on_conflict_do_nothing(constraint="uq_user_movie_rating")
                    session.execute(stmt)
                    session.commit()

                logger.info(f"Inserting {len(interaction_records)} user interactions...")
                for i in tqdm(range(0, len(interaction_records), batch_size), desc="PostgreSQL Interactions"):
                    chunk = interaction_records[i : i + batch_size]
                    session.bulk_insert_mappings(Interaction, chunk)
                    session.commit()

                logger.info(f"Successfully seeded {len(rating_records)} ratings and interactions.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error seeding PostgreSQL: {e}")
        raise e
    finally:
        session.close()


def seed_neo4j(clean: bool = False, max_users: int = 2500):
    logger.info("=" * 60)
    logger.info("  2. SEEDING NEO4J KNOWLEDGE GRAPH")
    logger.info(f"  Target: {settings.NEO4J_URI} as {settings.NEO4J_USER}")
    logger.info(f"  Max Users Target: {max_users}")
    logger.info("=" * 60)

    try:
        neo4j_client.connect()
    except Exception as e:
        logger.error(f"Could not connect to Neo4j at {settings.NEO4J_URI}: {e}")
        return

    session = neo4j_client.get_session()
    batch_size = 2500

    try:
        if clean:
            logger.warning("Cleaning existing Neo4j graph nodes & relationships...")
            session.run("MATCH (n) DETACH DELETE n")

        # 1. Unique Constraints & Indexes
        logger.info("Creating Neo4j unique constraints & indexes...")
        constraints = [
            "CREATE CONSTRAINT movie_id_unique IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT genre_name_unique IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT director_id_unique IF NOT EXISTS FOR (d:Director) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT actor_id_unique IF NOT EXISTS FOR (a:Actor) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT writer_id_unique IF NOT EXISTS FOR (w:Writer) REQUIRE w.id IS UNIQUE",
            "CREATE CONSTRAINT producer_id_unique IF NOT EXISTS FOR (p:Producer) REQUIRE p.id IS UNIQUE",
            "CREATE INDEX movie_lens_id_idx IF NOT EXISTS FOR (m:Movie) ON (m.movieLensId)",
        ]
        for c in constraints:
            session.run(c)
        logger.info("Neo4j constraints and indexes verified.")

        # 2. Ingest Movie Nodes & Genre Edges
        metadata_file = Path(settings.METADATA_PATH)
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                movies = json.load(f)

            # Check if movies_and_posters.csv is available
            posters_file = metadata_file.parent / "movies_and_posters.csv"
            poster_map = {}
            if posters_file.exists():
                import csv
                with open(posters_file, mode="r", encoding="utf-8") as pf:
                    rdr = csv.DictReader(pf)
                    for r in rdr:
                        p_mid = r.get("movieId")
                        p_url = r.get("poster_url", "").strip()
                        if p_mid and p_url and p_url.startswith("http"):
                            try:
                                poster_map[int(p_mid)] = p_url.replace("/original/", "/w500/")
                            except ValueError:
                                pass

            logger.info(f"Ingesting {len(movies)} Movie nodes into Neo4j...")
            movie_query = """
            UNWIND $batch AS m
            MERGE (node:Movie {id: m.id})
            SET node.movieLensId = m.movieLensId,
                node.title = m.title,
                node.fullTitle = m.fullTitle,
                node.releaseYear = m.releaseYear,
                node.genres = m.genres,
                node.posterUrl = m.posterUrl
            """
            for i in tqdm(range(0, len(movies), batch_size), desc="Neo4j Movies"):
                batch = [{
                    "id": int(m["ckan_id"]),
                    "movieLensId": int(m["movie_id"]),
                    "title": str(m["title"]),
                    "fullTitle": str(m.get("full_title", "")),
                    "releaseYear": m.get("release_year"),
                    "genres": m.get("genres", []),
                    "posterUrl": poster_map.get(int(m["movie_id"])),
                } for m in movies[i : i + batch_size]]
                session.run(movie_query, batch=batch)

            # Ingest Genre relationships
            genre_pairs = []
            for m in movies:
                mid = int(m["ckan_id"])
                for g in m.get("genres", []):
                    if g:
                        genre_pairs.append({"movieId": mid, "genre": g})

            logger.info(f"Ingesting {len(genre_pairs)} HAS_GENRE relationships...")
            genre_query = """
            UNWIND $batch AS g
            MATCH (m:Movie {id: g.movieId})
            MERGE (genre:Genre {name: g.genre})
            MERGE (m)-[:HAS_GENRE]->(genre)
            """
            for i in tqdm(range(0, len(genre_pairs), batch_size), desc="Neo4j HAS_GENRE"):
                batch = genre_pairs[i : i + batch_size]
                session.run(genre_query, batch=batch)

        # 3. Ingest Benchmark Users & [:LIKED] edges
        ratings_path = Path(settings.DATA_DIR) / "ratings_final.npy"
        if ratings_path.exists():
            ratings_arr = np.load(ratings_path)
            users_set = set()
            user_likes = []

            for u, m, r in ratings_arr:
                u_int, m_int, r_int = int(u), int(m), int(r)
                if u_int < max_users and r_int == 1:
                    users_set.add(u_int)
                    user_likes.append({"userId": u_int, "movieId": m_int})

            user_batch = [{"id": uid, "name": f"User #{uid}"} for uid in users_set]
            logger.info(f"Ingesting {len(user_batch)} User nodes into Neo4j...")
            for i in tqdm(range(0, len(user_batch), batch_size), desc="Neo4j Users"):
                session.run("""
                UNWIND $batch AS u
                MERGE (node:User {id: u.id})
                SET node.name = u.name
                """, batch=user_batch[i : i + batch_size])

            logger.info(f"Ingesting {len(user_likes)} [:LIKED] relationships...")
            like_query = """
            UNWIND $batch AS row
            MATCH (u:User {id: row.userId})
            MATCH (m:Movie {id: row.movieId})
            MERGE (u)-[:LIKED]->(m)
            """
            for i in tqdm(range(0, len(user_likes), batch_size), desc="Neo4j [:LIKED]"):
                batch = user_likes[i : i + batch_size]
                session.run(like_query, batch=batch)

        # 4. Ingest KG Semantic Triples (Directors, Writers, Producers, Actors)
        kg_path = Path(settings.KG_PATH)
        rel_path = Path(settings.DATA_DIR) / "kg_relations.json"

        if kg_path.exists() and rel_path.exists():
            logger.info(f"Loading KG Triples from {kg_path}...")
            kg = np.load(kg_path)
            with open(rel_path, "r", encoding="utf-8") as f:
                rel_info = json.load(f)

            index2rel = {int(k): v for k, v in rel_info["index2rel"].items()}

            directors = []
            actors = []
            writers = []
            producers = []

            for h, r, t in kg:
                h_int, r_int, t_int = int(h), int(r), int(t)
                rel_name = index2rel.get(r_int, "")

                if rel_name == "film.film.director":
                    directors.append({"head": h_int, "tail": t_int})
                elif rel_name in ("film.film.actor", "film.film.star"):
                    actors.append({"head": h_int, "tail": t_int})
                elif rel_name == "film.film.writer":
                    writers.append({"head": h_int, "tail": t_int})
                elif rel_name == "film.film.producer":
                    producers.append({"head": h_int, "tail": t_int})

            # Ingest Directors
            if directors:
                logger.info(f"Ingesting {len(directors)} DIRECTED_BY edges...")
                q = """
                UNWIND $batch AS row
                MATCH (m:Movie {id: row.head})
                MERGE (d:Director {id: row.tail})
                MERGE (m)-[:DIRECTED_BY]->(d)
                """
                for i in tqdm(range(0, len(directors), batch_size), desc="Neo4j DIRECTED_BY"):
                    session.run(q, batch=directors[i : i + batch_size])

            # Ingest Writers
            if writers:
                logger.info(f"Ingesting {len(writers)} WRITTEN_BY edges...")
                q = """
                UNWIND $batch AS row
                MATCH (m:Movie {id: row.head})
                MERGE (w:Writer {id: row.tail})
                MERGE (m)-[:WRITTEN_BY]->(w)
                """
                for i in tqdm(range(0, len(writers), batch_size), desc="Neo4j WRITTEN_BY"):
                    session.run(q, batch=writers[i : i + batch_size])

            # Ingest Producers
            if producers:
                logger.info(f"Ingesting {len(producers)} PRODUCED_BY edges...")
                q = """
                UNWIND $batch AS row
                MATCH (m:Movie {id: row.head})
                MERGE (p:Producer {id: row.tail})
                MERGE (m)-[:PRODUCED_BY]->(p)
                """
                for i in tqdm(range(0, len(producers), batch_size), desc="Neo4j PRODUCED_BY"):
                    session.run(q, batch=producers[i : i + batch_size])

            # Ingest Actors (sample top 30k to keep ingestion fast & responsive)
            if actors:
                sampled_actors = actors[:30000]
                logger.info(f"Ingesting {len(sampled_actors)} ACTED_BY edges...")
                q = """
                UNWIND $batch AS row
                MATCH (m:Movie {id: row.head})
                MERGE (a:Actor {id: row.tail})
                MERGE (m)-[:ACTED_BY]->(a)
                """
                for i in tqdm(range(0, len(sampled_actors), batch_size), desc="Neo4j ACTED_BY"):
                    session.run(q, batch=sampled_actors[i : i + batch_size])

        # 5. Sync Curated Entity Names (Directors, Actors, Studios)
        names_path = Path(settings.DATA_DIR) / "entity_names.json"
        if names_path.exists():
            with open(names_path, "r", encoding="utf-8") as f:
                names = json.load(f)
            batch = [{"id": int(k), "name": v} for k, v in names.items()]
            session.run("""
            UNWIND $batch AS item
            MATCH (n) WHERE n.id = item.id
            SET n.name = item.name
            """, batch=batch)
            logger.info(f"Synced {len(names)} human-readable entity names into Neo4j nodes.")

        logger.info("Neo4j Knowledge Graph seeding completed successfully!")

    except Exception as e:
        logger.error(f"Error seeding Neo4j: {e}")
        raise e
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Seed PostgreSQL and Neo4j for CKAN Movie Recommender System")
    parser.add_argument("--clean", action="store_true", help="Clean/truncate existing database tables and graph before seeding")
    parser.add_argument("--postgres-only", action="store_true", help="Only seed PostgreSQL")
    parser.add_argument("--neo4j-only", action="store_true", help="Only seed Neo4j")
    parser.add_argument("--max-users", type=int, default=2500, help="Maximum number of benchmark users to seed (default: 2500 - entire benchmark dataset)")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("Starting CKAN Unified Database Seeder...")

    if not args.neo4j_only:
        try:
            seed_postgres(clean=args.clean, max_users=args.max_users)
        except Exception as e:
            logger.error(f"PostgreSQL seeding failed: {e}")

    if not args.postgres_only:
        try:
            seed_neo4j(clean=args.clean, max_users=args.max_users)
        except Exception as e:
            logger.error(f"Neo4j seeding failed: {e}")

    logger.info(f"All seeding operations finished in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
