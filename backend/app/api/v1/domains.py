from fastapi import APIRouter
from app.models.schemas import DomainListResponseDto
from app.recommendation.engine import recommendation_engine

router = APIRouter(prefix="/domains", tags=["Domains"])

@router.get("", response_model=DomainListResponseDto)
def get_domains():
    domains = recommendation_engine.get_domains_info()
    return DomainListResponseDto(domains=domains)
