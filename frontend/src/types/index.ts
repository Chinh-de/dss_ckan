export type DomainType = "movie" | "book" | "music";

export interface DomainInfo {
  id: DomainType;
  name: string;
  vietnameseName: string;
  itemTerm: string;
  description: string;
  itemsCount: number;
  usersCount: number;
  triplesCount: number;
  relationsCount: number;
  sampleUsers: number[];
  accentColor: string;
}

export interface DomainListResponse {
  domains: DomainInfo[];
}

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
  id: number;
  domain: DomainType;
  title: string;
  subtitle?: string;
  secondaryInfo?: string;
  posterUrl?: string | null;
  score: number;
  totalRatings?: number;
  reasons: string[];
  metadata?: Record<string, any>;
  
  // Backward compatibility fields
  movieId?: number;
  movieLensId?: number;
  genres?: string[];
  releaseYear?: number;
}

export interface RecommendationResponse {
  userId: number;
  domain: DomainType;
  total: number;
  recommendations: RecommendationItem[];
}

export interface ExplanationPath {
  id: string;
  sourceMovieTitle: string;
  relation: string;
  relationLabel?: string;
  entityName: string;
  targetMovieTitle: string;
  naturalLanguage: string;
}

export interface ExplanationResponse {
  userId: number;
  movieId: number;
  domain?: DomainType;
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

// Sparsity benchmark (measured in the notebook) and the live few-likes comparison
export interface SparsityPoint {
  ratio: number;
  mf_auc: number;
  ripplenet_auc: number;
  ckan_auc: number;
}

export interface ModelResult {
  model: string;
  auc: number;
  f1: number;
  acc: number;
  recall: Record<string, number>;
}

export interface ColdStartSimulationResponse {
  domain: DomainType;
  interactions: number;
  source: string;
  sparsity: SparsityPoint[];
  sparsityEvalUsers?: number | null;
  sparsityEvalRows?: number | null;
  models: ModelResult[];
  seedItems: string[];
  popularRecommendations: RecommendationItem[];
  ckanRecommendations: RecommendationItem[];
}

// Catalog Items
export interface ItemDto {
  id: number;
  domain: DomainType;
  title: string;
  subtitle?: string;
  category?: string;
  posterUrl?: string | null;
  releaseYear?: number;
  details?: Record<string, any>;
}

export interface ItemListResponse {
  domain: DomainType;
  total: number;
  page: number;
  limit: number;
  data: ItemDto[];
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
  sampleLikes?: string[];
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
