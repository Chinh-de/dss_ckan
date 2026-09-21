import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from app.core.config import settings

logger = logging.getLogger("neo4j_client")

class Neo4jClient:
    _instance = None
    _driver: Optional[Driver] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
        return cls._instance

    def connect(self):
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=10,
                )
                self.verify_connectivity()
                logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
            except Exception as e:
                logger.warning(f"Could not connect to Neo4j at {settings.NEO4J_URI}: {e}")
                self._driver = None

    def verify_connectivity(self) -> bool:
        if self._driver:
            try:
                self._driver.verify_connectivity()
                return True
            except Exception as e:
                logger.error(f"Neo4j connectivity check failed: {e}")
                return False
        return False

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    def get_session(self):
        if self._driver is None:
            self.connect()
        if self._driver:
            return self._driver.session()
        raise RuntimeError("Neo4j driver is not connected.")

    def read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self._driver is None:
            self.connect()
        if self._driver is None:
            logger.warning("Neo4j is not connected. Returning empty list.")
            return []

        parameters = parameters or {}
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j read query: {e}")
            return []

    def write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self._driver is None:
            self.connect()
        if self._driver is None:
            logger.warning("Neo4j is not connected. Skipping write query.")
            return []

        parameters = parameters or {}
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j write query: {e}")
            return []

neo4j_client = Neo4jClient()

