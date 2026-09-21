from fastapi import APIRouter, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.sql_models import Movie, Recommendation
from app.models.schemas import RecommendationResponseDto
from app.recommendation.engine import recommendation_engine

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

def save_recs_to_db(user_id: int, recs: list, db: Session):
    try:
        for r in recs:
            db.add(Recommendation(userId=user_id, movieId=r.movieId, score=r.score))
        db.commit()
    except Exception as e:
        print(f"Failed to record recommendations to DB: {e}")

@router.get("", response_model=RecommendationResponseDto)
def get_recommendations(
    userId: int = Query(1, description="User ID"),
    topK: int = Query(10, ge=1, le=50, description="Number of items to recommend"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    recs = recommendation_engine.recommend(user_id=userId, top_k=topK, db=db)

    # Enrich movie details with Postgres DB
    if recs:
        movie_ids = [r.movieId for r in recs]
        db_movies = {m.id: m for m in db.query(Movie).filter(Movie.id.in_(movie_ids)).all()}

        for r in recs:
            db_m = db_movies.get(r.movieId)
            if db_m:
                r.title = db_m.title or r.title
                r.movieLensId = db_m.movieLensId or r.movieLensId
                r.genres = db_m.genres or r.genres
                r.releaseYear = db_m.releaseYear or r.releaseYear
                r.posterUrl = db_m.posterUrl or r.posterUrl
                r.totalRatings = len(db_m.ratings) if db_m.ratings else 0

    return RecommendationResponseDto(
        userId=userId,
        total=len(recs),
        recommendations=recs
    )

