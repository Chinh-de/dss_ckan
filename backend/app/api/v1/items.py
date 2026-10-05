from typing import Optional
from fastapi import APIRouter, Query
from app.models.schemas import ItemListResponseDto
from app.recommendation.engine import recommendation_engine

router = APIRouter(prefix="/items", tags=["Catalog & Explore"])

@router.get("", response_model=ItemListResponseDto)
def list_items(
    domain: str = Query("movie", description="Domain dataset: movie, book, or music"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search keyword in title or author/director"),
    category: Optional[str] = Query(None, description="Genre or category filter")
):
    result = recommendation_engine.get_items(
        domain_name=domain,
        page=page,
        limit=limit,
        search=search,
        category=category
    )
    return result
