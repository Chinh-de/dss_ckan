from fastapi import APIRouter, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.sql_models import Movie, Recommendation
from app.models.schemas import (
    RecommendationResponseDto,
    RecommendationItemDto,
    RecommendationFeedbackDto,
    DomainName,
)
from app.recommendation.engine import recommendation_engine

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("", response_model=RecommendationResponseDto)
def get_recommendations(
    domain: DomainName = Query("movie", description="Domain dataset: movie, book, or music"),
    userId: int = Query(1, description="User ID"),
    topK: int = Query(12, ge=1, le=50, description="Number of items to recommend"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    recs = recommendation_engine.recommend(
        domain_name=domain,
        user_id=userId,
        top_k=topK,
        db=db if domain == "movie" else None
    )

    # For movie domain, enrich with PostgreSQL DB if records exist
    if domain == "movie" and recs and db:
        try:
            movie_ids = [r.id for r in recs]
            db_movies = {m.id: m for m in db.query(Movie).filter(Movie.id.in_(movie_ids)).all()}
            for r in recs:
                db_m = db_movies.get(r.id)
                if db_m:
                    r.title = db_m.title or r.title
                    r.movieLensId = db_m.movieLensId or r.movieLensId
                    r.genres = db_m.genres or r.genres
                    r.releaseYear = db_m.releaseYear or r.releaseYear
                    r.posterUrl = db_m.posterUrl or r.posterUrl
                    r.totalRatings = len(db_m.ratings) if db_m.ratings else 0
        except Exception as e:
            pass

    return RecommendationResponseDto(
        userId=userId,
        domain=domain,
        total=len(recs),
        recommendations=recs
    )

@router.post("/feedback")
def submit_feedback(payload: RecommendationFeedbackDto):
    recommendation_engine.record_feedback(
        domain_name=payload.domain,
        user_id=payload.userId,
        item_id=payload.itemId,
        action=payload.action
    )
    return {
        "status": "success",
        "domain": payload.domain,
        "userId": payload.userId,
        "itemId": payload.itemId,
        "action": payload.action,
        "message": f"Tương tác {payload.action} đã được ghi nhận vào phiên làm việc."
    }
