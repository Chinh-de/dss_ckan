from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.schemas import ExplanationResponseDto, DomainName
from app.recommendation.engine import recommendation_engine, ItemNotFoundError

router = APIRouter(prefix="/explainability", tags=["Explainability"])

@router.get("/{itemId}", response_model=ExplanationResponseDto)
def explain_recommendation(
    itemId: int,
    domain: DomainName = Query("movie", description="Domain dataset: movie, book, or music"),
    userId: int = Query(1, description="User ID"),
    db: Session = Depends(get_db)
):
    try:
        explanation = recommendation_engine.explain(
            domain_name=domain,
            user_id=userId,
            item_id=itemId,
            db=db if domain == "movie" else None
        )
        return explanation
    except ItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể tạo giải thích cho mục {itemId}: {str(e)}")
