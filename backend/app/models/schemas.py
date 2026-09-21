from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field

# Auth schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class OnboardingRequest(BaseModel):
    user_id: int
    preferred_genres: List[str] = Field(default_factory=list)
    liked_movie_ids: List[int] = Field(default_factory=list)

# Movie schemas
class MovieDto(BaseModel):
    id: int
    movieLensId: int
    title: str
    fullTitle: Optional[str] = None
    overview: Optional[str] = None
    releaseYear: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    posterUrl: Optional[str] = None
    totalRatings: Optional[int] = 0
    avgRating: Optional[float] = None

class MovieListResponse(BaseModel):
    data: List[MovieDto]
    total: int
    page: int
    limit: int
    totalPages: int

# Rating schemas
class RatingCreateDto(BaseModel):
    userId: int
    movieId: int
    rating: float = Field(..., ge=0.5, le=5.0)

class RatingResponseDto(BaseModel):
    message: str
    rating: Dict[str, Any]
    interactionType: str

# Recommendation schemas
class RecommendationItemDto(BaseModel):
    movieId: int
    movieLensId: Optional[int] = None
    title: str
    releaseYear: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    posterUrl: Optional[str] = None
    score: float
    totalRatings: Optional[int] = 0
    reasons: List[str] = Field(default_factory=list)

class RecommendationResponseDto(BaseModel):
    userId: int
    total: int
    recommendations: List[RecommendationItemDto]

# Graph schemas
class GraphNodeDto(BaseModel):
    id: str
    label: str
    type: str
    data: Dict[str, Any]

class GraphEdgeDto(BaseModel):
    id: str
    source: str
    target: str
    type: str
    label: str

class SubgraphResponseDto(BaseModel):
    nodes: List[GraphNodeDto]
    edges: List[GraphEdgeDto]

# Explainability schemas
class ExplanationPathDto(BaseModel):
    id: str
    sourceMovieTitle: str
    relation: str
    entityName: str
    targetMovieTitle: str
    naturalLanguage: str

class ExplanationResponseDto(BaseModel):
    userId: int
    movieId: int
    score: float
    confidence: str
    executiveSummary: str
    paths: List[ExplanationPathDto]
    featureImportance: Dict[str, float]
    counterfactual: str
    subgraph: Optional[SubgraphResponseDto] = None

# User History & Directory schemas
class UserProfileDto(BaseModel):
    id: int
    name: str
    email: str
    totalRatings: int = 0
    totalLikes: int = 0
    totalDislikes: int = 0
    topGenres: List[str] = Field(default_factory=list)

class UserListResponse(BaseModel):
    total: int
    page: int
    limit: int
    users: List[UserProfileDto]

class UserHistoryItemDto(BaseModel):
    ratingId: int
    movieId: int
    movieLensId: Optional[int] = None
    title: str
    fullTitle: Optional[str] = None
    releaseYear: Optional[int] = None
    posterUrl: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    rating: float
    interactionType: str
    ratedAt: str

class UserHistoryResponse(BaseModel):
    user: UserProfileDto
    total: int
    page: int
    limit: int
    items: List[UserHistoryItemDto]

