import json
import logging
from pathlib import Path
from app.core.config import settings
from app.core.neo4j_client import neo4j_client

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("sync_names")

def sync_names_to_neo4j():
    names_path = Path(settings.DATA_DIR) / "entity_names.json"
    if not names_path.exists():
        logger.warning(f"File {names_path} not found.")
        return

    with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)

    logger.info(f"Loaded {len(names)} entity names. Connecting to Neo4j...")
    neo4j_client.connect()
    session = neo4j_client.get_session()

    batch = [{"id": int(k), "name": v} for k, v in names.items()]
    cypher = """
    UNWIND $batch AS item
    MATCH (n) WHERE n.id = item.id
    SET n.name = item.name
    """
    session.run(cypher, batch=batch)
    session.close()
    logger.info(f"Successfully synced {len(names)} human-readable names to Neo4j nodes!")

if __name__ == "__main__":
    sync_names_to_neo4j()

