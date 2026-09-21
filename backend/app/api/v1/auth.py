from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, create_access_token, decode_access_token
from app.core.neo4j_client import neo4j_client
from app.models.sql_models import User, Rating, Interaction
from app.models.schemas import UserCreate, UserLogin, TokenResponse, OnboardingRequest

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse)
def register(dto: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == dto.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    user = User(
        email=dto.email,
        name=dto.name,
        passwordHash=get_password_hash(dto.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email, "name": user.name}
    )

@router.post("/login", response_model=TokenResponse)
def login(dto: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == dto.email).first()
    if not user or not verify_password(dto.password, user.passwordHash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    token = create_access_token(user.id)
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email, "name": user.name}
    )

@router.post("/onboarding")
def onboarding(req: OnboardingRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {req.user_id} not found")

    # Record liked movies in Postgres and Neo4j
    for movie_id in req.liked_movie_ids:
        # 1. Postgres Rating & Interaction
        rating = db.query(Rating).filter(Rating.userId == req.user_id, Rating.movieId == movie_id).first()
        if not rating:
            db.add(Rating(userId=req.user_id, movieId=movie_id, rating=5.0))
        db.add(Interaction(userId=req.user_id, movieId=movie_id, type="LIKE"))

        # 2. Sync to Neo4j
        cypher = """
        MERGE (u:User {id: $userId})
        MERGE (m:Movie {id: $movieId})
        MERGE (u)-[r:LIKED]->(m)
        SET r.rating = 5.0, r.updatedAt = datetime()
        """
        neo4j_client.write(cypher, {"userId": req.user_id, "movieId": movie_id})

    db.commit()
    return {
        "message": "Onboarding completed successfully!",
        "userId": req.user_id,
        "likesCount": len(req.liked_movie_ids),
        "preferredGenres": req.preferred_genres
    }

