import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BACKEND_DIR.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "CKAN Knowledge-aware Movie Recommendation Platform"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres_password@localhost:5435/ckan_recommendation"
    )
    
    # Neo4j Graph DB
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "neo4j_password")
    
    # JWT Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "ckan_secret_jwt_key_2026_super_secure")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Data & ML Model Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BACKEND_DIR / "data" / "movie")))
    MODEL_PATH: str = os.getenv("CKAN_MODEL_PATH", str(BACKEND_DIR / "models" / "ckan_movie.pt"))
    CACHE_PATH: str = os.getenv("CKAN_CACHE_PATH", str(BACKEND_DIR / "models" / "cache" / "propagation_cache.pkl"))
    METADATA_PATH: str = os.getenv("CKAN_METADATA_PATH", str(BACKEND_DIR / "data" / "movie" / "movies_metadata.json"))
    KG_PATH: str = os.getenv("CKAN_KG_PATH", str(BACKEND_DIR / "data" / "movie" / "kg_final.npy"))
    
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "2048"))
    USE_CUDA: bool = os.getenv("USE_CUDA", "false").lower() in ("true", "1", "yes")

    class Config:
        case_sensitive = True
        env_file = (".env", "../.env")
        extra = "ignore"

settings = Settings()

