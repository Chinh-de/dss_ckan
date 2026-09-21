import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.neo4j_client import neo4j_client
from app.recommendation.engine import recommendation_engine
from app.api.v1.auth import router as auth_router
from app.api.v1.movies import router as movies_router
from app.api.v1.ratings import router as ratings_router
from app.api.v1.recommendations import router as recs_router
from app.api.v1.explainability import router as explain_router
from app.api.v1.graph import router as graph_router
from app.api.v1.users import router as users_router

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up CKAN Unified FastAPI Backend...")
    # Initialize SQL database schema if reachable
    try:
        from app.core.database import engine, Base
        import app.models.sql_models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL tables verified / created successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL to create tables: {e}")

    try:
        neo4j_client.connect()
    except Exception as e:
        logger.warning(f"Could not connect to Neo4j on startup: {e}")

    try:
        recommendation_engine.initialize()
    except Exception as e:
        logger.error(f"Error during recommendation engine initialization: {e}")
    yield
    logger.info("Shutting down CKAN Unified FastAPI Backend...")
    neo4j_client.close()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Unified Backend combining REST API, PostgreSQL, Neo4j, and CKAN Recommendation Engine",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 Routers
api_v1_prefix = settings.API_V1_STR
app.include_router(auth_router, prefix=api_v1_prefix)
app.include_router(movies_router, prefix=api_v1_prefix)
app.include_router(ratings_router, prefix=api_v1_prefix)
app.include_router(recs_router, prefix=api_v1_prefix)
app.include_router(explain_router, prefix=api_v1_prefix)
app.include_router(graph_router, prefix=api_v1_prefix)
app.include_router(users_router, prefix=api_v1_prefix)

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "neo4j_connected": neo4j_client.verify_connectivity(),
        "model_loaded": recommendation_engine.is_loaded,
        "entities_count": recommendation_engine.n_entity,
        "relations_count": recommendation_engine.n_relation,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

