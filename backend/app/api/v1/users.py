from typing import Optional, List, Dict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from app.core.database import get_db
from app.core.neo4j_client import neo4j_client
from app.models.sql_models import User, Movie, Rating
from app.models.schemas import (
    UserProfileDto,
    UserListResponse,
    UserHistoryItemDto,
    UserHistoryResponse,
)

router = APIRouter(prefix="/users", tags=["Users"])

class CreateUserDto(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None

def _get_user_stats(user: User, db: Session) -> UserProfileDto:
    """Helper to compute user ratings, likes, dislikes, and top genres."""
    ratings_data = (
        db.query(Rating.rating, Movie.genres)
        .join(Movie, Movie.id == Rating.movieId)
        .filter(Rating.userId == user.id)
        .all()
    )

    total_ratings = len(ratings_data)
    total_likes = 0
    total_dislikes = 0
    genre_freq: Dict[str, int] = {}

    for r, genres in ratings_data:
        is_like = r >= 3.5
        if is_like:
            total_likes += 1
        else:
            total_dislikes += 1

        if genres and isinstance(genres, list):
            for g in genres:
                if g:
                    genre_freq[g] = genre_freq.get(g, 0) + (2 if is_like else 1)

    top_genres = sorted(genre_freq.keys(), key=lambda k: genre_freq[k], reverse=True)[:4]

    return UserProfileDto(
        id=user.id,
        name=user.name or f"User #{user.id}",
        email=user.email or f"user{user.id}@moviekg.ai",
        totalRatings=total_ratings,
        totalLikes=total_likes,
        totalDislikes=total_dislikes,
        topGenres=top_genres,
    )

@router.get("", response_model=UserListResponse)
def list_users(
    search: Optional[str] = Query(None, description="Search by ID, name or email"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(User)

    if search:
        s = search.strip()
        if s.isdigit():
            query = query.filter(or_(User.id == int(s), User.name.ilike(f"%{s}%"), User.email.ilike(f"%{s}%")))
        else:
            query = query.filter(or_(User.name.ilike(f"%{s}%"), User.email.ilike(f"%{s}%")))

    total = query.count()
    users = query.order_by(User.id.asc()).offset((page - 1) * limit).limit(limit).all()

    items = [_get_user_stats(u, db) for u in users]

    return UserListResponse(
        total=total,
        page=page,
        limit=limit,
        users=items
    )

@router.post("", response_model=UserProfileDto)
def create_user(dto: Optional[CreateUserDto] = None, db: Session = Depends(get_db)):
    max_id = db.query(func.max(User.id)).scalar() or 0
    new_id = max_id + 1

    req_name = dto.name.strip() if dto and dto.name and dto.name.strip() else f"Movie Fan #{new_id}"
    req_email = dto.email.strip() if dto and dto.email and dto.email.strip() else f"user{new_id}@moviekg.ai"

    existing = db.query(User).filter(User.email == req_email).first()
    if existing:
        req_email = f"user{new_id}_{int(datetime.utcnow().timestamp())}@moviekg.ai"

    user = User(
        id=new_id,
        name=req_name,
        email=req_email,
        passwordHash=None
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    try:
        neo4j_client.write(
            "MERGE (u:User {id: $id}) SET u.name = $name",
            {"id": user.id, "name": user.name}
        )
    except Exception as e:
        print(f"Warning: Could not sync new user to Neo4j: {e}")

    return _get_user_stats(user, db)

@router.get("/{userId}", response_model=UserProfileDto)
def get_user_profile(userId: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == userId).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {userId} not found")
    return _get_user_stats(user, db)

@router.get("/{userId}/history", response_model=UserHistoryResponse)
def get_user_history(
    userId: int,
    filter_type: Optional[str] = Query("ALL", description="Filter by ALL, LIKE, or DISLIKE"),
    search: Optional[str] = Query(None, description="Search movie title in user history"),
    genre: Optional[str] = Query(None, description="Filter by genre name"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == userId).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {userId} not found")

    user_profile = _get_user_stats(user, db)

    query = (
        db.query(Rating, Movie)
        .join(Movie, Movie.id == Rating.movieId)
        .filter(Rating.userId == userId)
    )

    if filter_type == "LIKE":
        query = query.filter(Rating.rating >= 3.5)
    elif filter_type == "DISLIKE":
        query = query.filter(Rating.rating < 3.5)

    if search:
        query = query.filter(Movie.title.ilike(f"%{search.strip()}%"))

    # Total matching items
    total = query.count()

    # Pagination and order (most recent ratings or highest rating first)
    records = (
        query
        .order_by(Rating.createdAt.desc(), Rating.id.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    items = []
    for r, m in records:
        genres = m.genres if isinstance(m.genres, list) else []
        if genre and genre not in genres:
            continue
        items.append(
            UserHistoryItemDto(
                ratingId=r.id,
                movieId=m.id,
                movieLensId=m.movieLensId,
                title=m.title,
                fullTitle=m.fullTitle,
                releaseYear=m.releaseYear,
                posterUrl=m.posterUrl,
                genres=genres,
                rating=r.rating,
                interactionType="LIKE" if r.rating >= 3.5 else "DISLIKE",
                ratedAt=r.createdAt.isoformat(),
            )
        )

    return UserHistoryResponse(
        user=user_profile,
        total=total,
        page=page,
        limit=limit,
        items=items
    )
