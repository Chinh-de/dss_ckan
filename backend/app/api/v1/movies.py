from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, String, desc
from app.core.database import get_db
from app.core.neo4j_client import neo4j_client
from app.models.sql_models import Movie, Rating
from app.models.schemas import MovieDto, MovieListResponse

router = APIRouter(prefix="/movies", tags=["Movies"])

@router.get("", response_model=MovieListResponse)
def get_movies(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    genre: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Movie)
    if search:
        query = query.filter(Movie.title.ilike(f"%{search}%"))
    if genre:
        # Check string inside JSON/ARRAY
        query = query.filter(func.cast(Movie.genres, String).ilike(f"%{genre}%"))

    total = query.count()
    skip = (page - 1) * limit
    movies = query.order_by(Movie.id.asc()).offset(skip).limit(limit).all()

    items = []
    for m in movies:
        items.append(MovieDto(
            id=m.id,
            movieLensId=m.movieLensId,
            title=m.title,
            fullTitle=m.fullTitle,
            overview=m.overview,
            releaseYear=m.releaseYear,
            genres=m.genres or [],
            posterUrl=m.posterUrl,
            totalRatings=len(m.ratings) if m.ratings else 0
        ))

    return MovieListResponse(
        data=items,
        total=total,
        page=page,
        limit=limit,
        totalPages=(total + limit - 1) // limit
    )

@router.get("/onboarding-candidates", response_model=List[MovieDto])
def get_onboarding_candidates(
    genres: Optional[str] = Query(None, description="Comma-separated genres"),
    limit: int = Query(12, ge=4, le=30),
    db: Session = Depends(get_db)
):
    genre_list = [g.strip() for g in genres.split(",") if g.strip()] if genres else []

    candidate_movies = []
    seen_ids = set()

    if genre_list:
        per_genre_limit = max(2, (limit // len(genre_list)) + 1)
        for g in genre_list:
            top_for_genre = (
                db.query(Movie)
                .join(Rating, Rating.movieId == Movie.id)
                .filter(func.cast(Movie.genres, String).ilike(f"%{g}%"))
                .group_by(Movie.id)
                .order_by(desc(func.count(Rating.id)))
                .limit(per_genre_limit)
                .all()
            )
            for m in top_for_genre:
                if m.id not in seen_ids:
                    seen_ids.add(m.id)
                    candidate_movies.append(m)

    # Fill remaining with all-time popular movies if below limit
    if len(candidate_movies) < limit:
        remaining = limit - len(candidate_movies)
        query = (
            db.query(Movie)
            .join(Rating, Rating.movieId == Movie.id)
            .filter(Movie.id.notin_(seen_ids) if seen_ids else True)
            .group_by(Movie.id)
            .order_by(desc(func.count(Rating.id)))
            .limit(remaining)
            .all()
        )
        for m in query:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                candidate_movies.append(m)

    return [
        MovieDto(
            id=m.id,
            movieLensId=m.movieLensId,
            title=m.title,
            fullTitle=m.fullTitle,
            overview=m.overview,
            releaseYear=m.releaseYear,
            genres=m.genres or [],
            posterUrl=m.posterUrl,
            totalRatings=len(m.ratings) if m.ratings else 0
        )
        for m in candidate_movies
    ]

@router.get("/{id}", response_model=MovieDto)
def get_movie_by_id(id: int, db: Session = Depends(get_db)):
    m = db.query(Movie).filter(Movie.id == id).first()
    if not m:
        raise HTTPException(status_code=404, detail=f"Movie {id} not found")

    ratings = db.query(Rating.rating).filter(Rating.movieId == id).all()
    avg_rating = sum([r[0] for r in ratings]) / len(ratings) if ratings else None

    return MovieDto(
        id=m.id,
        movieLensId=m.movieLensId,
        title=m.title,
        fullTitle=m.fullTitle,
        overview=m.overview,
        releaseYear=m.releaseYear,
        genres=m.genres or [],
        posterUrl=m.posterUrl,
        totalRatings=len(ratings),
        avgRating=round(avg_rating, 2) if avg_rating else None
    )

@router.get("/{id}/related")
def get_related_movies(id: int, limit: int = 6, db: Session = Depends(get_db)):
    # Query Neo4j 2-hop graph similarity
    cypher = """
    MATCH (m:Movie {id: $id})-[:DIRECTED_BY|ACTED_BY|HAS_GENRE]->(common)<-[:DIRECTED_BY|ACTED_BY|HAS_GENRE]-(other:Movie)
    WHERE other.id <> $id
    RETURN 
      other.id AS id,
      other.title AS title,
      count(DISTINCT common) AS sharedCount,
      collect(DISTINCT coalesce(common.name, common.title, common.id))[0..3] AS sharedEntities
    ORDER BY sharedCount DESC
    LIMIT $limit
    """
    records = neo4j_client.read(cypher, {"id": id, "limit": limit})
    if not records:
        return []

    movie_ids = [int(r["id"]) for r in records]
    db_movies = {m.id: m for m in db.query(Movie).filter(Movie.id.in_(movie_ids)).all()}

    results = []
    for r in records:
        m_id = int(r["id"])
        db_m = db_movies.get(m_id)
        results.append({
            "id": m_id,
            "title": db_m.title if db_m else r.get("title", f"Movie #{m_id}"),
            "genres": db_m.genres if db_m else [],
            "releaseYear": db_m.releaseYear if db_m else None,
            "posterUrl": db_m.posterUrl if db_m else None,
            "sharedCount": r.get("sharedCount", 0),
            "sharedEntities": r.get("sharedEntities", [])
        })
    return results
