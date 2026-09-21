import {
  Movie,
  RecommendationItem,
  ExplanationResponse,
  SubgraphData,
  User,
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || "/api/v1";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const token = localStorage.getItem("cinegraph_token");
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options?.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export const api = {
  // Recommendations
  async getRecommendations(userId: number = 1, topK: number = 12): Promise<{ userId: number; total: number; recommendations: RecommendationItem[] }> {
    return request(`/recommendations?userId=${userId}&topK=${topK}`);
  },

  // Movies
  async getMovies(page: number = 1, limit: number = 20, search?: string, genre?: string): Promise<{ data: Movie[]; total: number; page: number; limit: number; totalPages: number }> {
    const params = new URLSearchParams({ page: `${page}`, limit: `${limit}` });
    if (search) params.append("search", search);
    if (genre) params.append("genre", genre);
    return request(`/movies?${params.toString()}`);
  },

  async getMovieById(id: number): Promise<Movie> {
    return request(`/movies/${id}`);
  },

  async getRelatedMovies(id: number, limit: number = 6): Promise<any[]> {
    return request(`/movies/${id}/related?limit=${limit}`);
  },

  async getOnboardingCandidates(genres?: string[], limit: number = 12): Promise<Movie[]> {
    const params = new URLSearchParams({ limit: `${limit}` });
    if (genres && genres.length > 0) {
      params.append("genres", genres.join(","));
    }
    return request(`/movies/onboarding-candidates?${params.toString()}`);
  },

  // Ratings & Interactions
  async rateMovie(userId: number, movieId: number, rating: number): Promise<{ message: string; interactionType: string; rating: any }> {
    return request(`/ratings`, {
      method: "POST",
      body: JSON.stringify({ userId, movieId, rating }),
    });
  },

  async getUserRatings(userId: number, page: number = 1, limit: number = 50): Promise<{ total: number; ratings: any[] }> {
    return request(`/ratings/user/${userId}?page=${page}&limit=${limit}`);
  },

  // Explainability
  async explainMovie(movieId: number, userId: number = 1): Promise<ExplanationResponse> {
    return request(`/explainability/${movieId}?userId=${userId}`);
  },

  // Knowledge Graph Subgraphs
  async getMovieSubgraph(movieId: number): Promise<SubgraphData> {
    return request(`/graph/subgraph/${movieId}`);
  },

  async getUserSubgraph(userId: number = 1, topK: number = 5): Promise<SubgraphData> {
    return request(`/graph/user-subgraph/${userId}?topK=${topK}`);
  },

  // Onboarding
  async completeOnboarding(userId: number, preferredGenres: string[], likedMovieIds: number[]): Promise<any> {
    return request(`/auth/onboarding`, {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
        preferred_genres: preferredGenres,
        liked_movie_ids: likedMovieIds,
      }),
    });
  },

  // Auth
  async register(email: string, password: string, name: string): Promise<{ access_token: string; user: User }> {
    return request(`/auth/register`, {
      method: "POST",
      body: JSON.stringify({ email, password, name }),
    });
  },

  async login(email: string, password: string): Promise<{ access_token: string; user: User }> {
    return request(`/auth/login`, {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  },

  // Users & History
  async getUsers(page: number = 1, limit: number = 20, search?: string): Promise<{ total: number; page: number; limit: number; users: any[] }> {
    const params = new URLSearchParams({ page: `${page}`, limit: `${limit}` });
    if (search) params.append("search", search);
    return request(`/users?${params.toString()}`);
  },

  async createUser(name?: string, email?: string): Promise<any> {
    return request(`/users`, {
      method: "POST",
      body: JSON.stringify({ name, email }),
    });
  },

  async getUserProfile(userId: number): Promise<any> {
    return request(`/users/${userId}`);
  },

  async getUserHistory(
    userId: number,
    filterType: "ALL" | "LIKE" | "DISLIKE" = "ALL",
    search?: string,
    genre?: string,
    page: number = 1,
    limit: number = 20
  ): Promise<any> {
    const params = new URLSearchParams({
      filter_type: filterType,
      page: `${page}`,
      limit: `${limit}`,
    });
    if (search) params.append("search", search);
    if (genre) params.append("genre", genre);
    return request(`/users/${userId}/history?${params.toString()}`);
  },
};

