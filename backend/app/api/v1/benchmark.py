from fastapi import APIRouter, Query
from app.models.schemas import ColdStartSimulationResponseDto, DomainName
from app.recommendation.engine import recommendation_engine

router = APIRouter(prefix="/benchmark", tags=["Benchmark & Cold Start"])

@router.get("/cold-start", response_model=ColdStartSimulationResponseDto)
def simulate_cold_start(
    domain: DomainName = Query("movie", description="Domain dataset: movie, book, or music"),
    interactions: int = Query(3, ge=1, le=20, description="Available interaction history count (1-20)")
):
    result = recommendation_engine.simulate_cold_start(domain_name=domain, interactions=interactions)
    return result
