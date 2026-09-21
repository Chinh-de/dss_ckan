export interface Movie {
  id: number;
  movieLensId?: number;
  title: string;
  fullTitle?: string;
  overview?: string;
  releaseYear?: number;
  genres: string[];
  posterUrl?: string | null;
  totalRatings?: number;
  avgRating?: number | null;
}

export interface RecommendationItem {
  movieId: number;
  movieLensId?: number;
  title: string;
  releaseYear?: number;
  genres: string[];
  posterUrl?: string | null;
  score: number;
  totalRatings?: number;
  reasons: string[];
}

export interface ExplanationPath {
  id: string;
  sourceMovieTitle: string;
  relation: string;
  entityName: string;
  targetMovieTitle: string;
  naturalLanguage: string;
}

export interface ExplanationResponse {
  userId: number;
  movieId: number;
  score: number;
  confidence: string;
  executiveSummary: string;
  paths: ExplanationPath[];
  featureImportance: Record<string, number>;
  counterfactual: string;
  subgraph?: SubgraphData;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  data: Record<string, any>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  label: string;
}

export interface SubgraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface User {
  id: number;
  email: string;
  name: string;
}

export interface UserProfile {
  id: number;
  name: string;
  email: string;
  totalRatings: number;
  totalLikes: number;
  totalDislikes: number;
  topGenres: string[];
}

export interface UserListResponse {
  total: number;
  page: number;
  limit: number;
  users: UserProfile[];
}

export interface UserHistoryItem {
  ratingId: number;
  movieId: number;
  movieLensId?: number;
  title: string;
  fullTitle?: string;
  releaseYear?: number;
  posterUrl?: string | null;
  genres: string[];
  rating: number;
  interactionType: "LIKE" | "DISLIKE";
  ratedAt: string;
}

export interface UserHistoryResponse {
  user: UserProfile;
  total: number;
  page: number;
  limit: number;
  items: UserHistoryItem[];
}

