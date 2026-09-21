from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.neo4j_client import neo4j_client
from app.models.sql_models import User, Movie, Rating, Interaction
from app.models.schemas import RatingCreateDto, RatingResponseDto

router = APIRouter(prefix="/ratings", tags=["Ratings"])

@router.post("", response_model=RatingResponseDto)
def create_or_update_rating(dto: RatingCreateDto, db: Session = Depends(get_db)):
    # 1. Verify user and movie exist
    user = db.query(User).filter(User.id == dto.userId).first()
    movie = db.query(Movie).filter(Movie.id == dto.movieId).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {dto.userId} not found")
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie {dto.movieId} not found")

    # 2. Upsert in Postgres
    rating_record = db.query(Rating).filter(
        Rating.userId == dto.userId,
        Rating.movieId == dto.movieId
    ).first()

    if rating_record:
        rating_record.rating = dto.rating
    else:
        rating_record = Rating(userId=dto.userId, movieId=dto.movieId, rating=dto.rating)
        db.add(rating_record)

    # 3. Record Interaction
    interaction_type = "LIKE" if dto.rating >= 3.5 else "DISLIKE"
    db.add(Interaction(userId=dto.userId, movieId=dto.movieId, type=interaction_type))
    db.commit()
    db.refresh(rating_record)

    # 4. Sync to Neo4j in real-time
    try:
        if dto.rating >= 3.5:
            cypher = """
            MERGE (u:User {id: $userId})
            MERGE (m:Movie {id: $movieId})
            MERGE (u)-[r:LIKED]->(m)
            SET r.rating = $rating, r.updatedAt = datetime()
            WITH u, m
            OPTIONAL MATCH (u)-[d:DISLIKED]->(m)
            DELETE d
            """
            neo4j_client.write(cypher, {"userId": dto.userId, "movieId": dto.movieId, "rating": dto.rating})
        else:
            cypher = """
            MERGE (u:User {id: $userId})
            MERGE (m:Movie {id: $movieId})
            MERGE (u)-[d:DISLIKED]->(m)
            SET d.rating = $rating, d.updatedAt = datetime()
            WITH u, m
            OPTIONAL MATCH (u)-[l:LIKED]->(m)
            DELETE l
            """
            neo4j_client.write(cypher, {"userId": dto.userId, "movieId": dto.movieId, "rating": dto.rating})
    except Exception as e:
        # Non-fatal log
        print(f"Warning: Failed to sync rating to Neo4j: {e}")

    return RatingResponseDto(
        message="Rating recorded successfully",
        rating={
            "userId": rating_record.userId,
            "movieId": rating_record.movieId,
            "rating": rating_record.rating,
            "createdAt": rating_record.createdAt.isoformat()
        },
        interactionType=interaction_type
    )

@router.get("/user/{userId}")
def get_user_ratings(
    userId: int,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1),
    db: Session = Depends(get_db)
):
    skip = (page - 1) * limit
    ratings = (
        db.query(Rating)
        .filter(Rating.userId == userId)
        .order_by(Rating.createdAt.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    total = db.query(Rating).filter(Rating.userId == userId).count()
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "ratings": [
            {
                "id": r.id,
                "movieId": r.movieId,
                "movieTitle": r.movie.title if r.movie else f"Movie #{r.movieId}",
                "rating": r.rating,
                "createdAt": r.createdAt.isoformat()
            }
            for r in ratings
        ]
    }

