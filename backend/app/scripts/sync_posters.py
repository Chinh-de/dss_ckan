import csv
import logging
from pathlib import Path
from tqdm import tqdm
from sqlalchemy import text
from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.core.neo4j_client import neo4j_client

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("sync_posters")

def sync_movie_posters(csv_path: str = None, optimize_w500: bool = True):
    if csv_path is None:
        csv_path = Path(settings.DATA_DIR) / "movies_and_posters.csv"
        if not csv_path.exists():
            csv_path = Path(settings.DATA_DIR) / "movie" / "movies_and_posters.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.error(f"Posters file not found at {csv_path}")
        return

    logger.info(f"Loading poster mappings from {csv_path}...")
    poster_map = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid_str = row.get("movieId")
            url = row.get("poster_url", "").strip()
            if mid_str and url and url.startswith("http"):
                try:
                    mid = int(mid_str)
                    if optimize_w500:
                        url = url.replace("/original/", "/w500/")
                    poster_map[mid] = url
                except ValueError:
                    continue

    logger.info(f"Loaded {len(poster_map)} valid poster URLs from CSV.")

    # 1. Update PostgreSQL
    logger.info("Updating PostgreSQL Movie.posterUrl...")
    db = SessionLocal()
    try:
        # Prepare parameters list for fast batch execution
        params = [{"movie_lens_id": mid, "poster_url": url} for mid, url in poster_map.items()]
        batch_size = 2000
        
        update_query = text("""
            UPDATE movies 
            SET "posterUrl" = :poster_url 
            WHERE "movieLensId" = :movie_lens_id;
        """)

        for i in tqdm(range(0, len(params), batch_size), desc="PostgreSQL Posters"):
            chunk = params[i : i + batch_size]
            db.execute(update_query, chunk)
            db.commit()

        # Check updated count
        result = db.execute(text("SELECT COUNT(*) FROM movies WHERE \"posterUrl\" LIKE 'https://image.tmdb.org%';")).scalar()
        logger.info(f"PostgreSQL update complete! Total movies with TMDb posters: {result}")
    except Exception as e:
        logger.error(f"Error updating PostgreSQL: {e}")
        db.rollback()
    finally:
        db.close()

    # 2. Update Neo4j Movie nodes
    logger.info("Updating Neo4j (m:Movie) posterUrl...")
    try:
        neo4j_client.connect()
        neo4j_session = neo4j_client.get_session()
        
        cypher_batch_size = 2500
        cypher_query = """
        UNWIND $batch AS item
        MATCH (m:Movie {movieLensId: item.movieLensId})
        SET m.posterUrl = item.posterUrl
        """

        items = [{"movieLensId": mid, "posterUrl": url} for mid, url in poster_map.items()]
        for i in tqdm(range(0, len(items), cypher_batch_size), desc="Neo4j Posters"):
            chunk = items[i : i + cypher_batch_size]
            neo4j_session.run(cypher_query, batch=chunk)

        # Count updated nodes in Neo4j
        count_res = neo4j_session.run("MATCH (m:Movie) WHERE m.posterUrl IS NOT NULL RETURN count(m) AS cnt;").single()
        cnt = count_res["cnt"] if count_res else 0
        logger.info(f"Neo4j update complete! Total Movie nodes with posterUrl: {cnt}")
        neo4j_session.close()
    except Exception as e:
        logger.error(f"Error updating Neo4j: {e}")

    logger.info("Poster synchronization successfully finished!")

if __name__ == "__main__":
    sync_movie_posters()
