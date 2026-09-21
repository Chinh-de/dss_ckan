import React, { useEffect, useState } from "react";
import { Sparkles, RefreshCw, AlertCircle, HelpCircle, Film, Heart } from "lucide-react";
import { RecommendationItem } from "../types";
import { MovieCard } from "./MovieCard";
import { api } from "../services/api";

interface RecommendationsViewProps {
  userId: number;
  onExplain: (movieId: number) => void;
  openOnboarding: () => void;
}

export const RecommendationsView: React.FC<RecommendationsViewProps> = ({
  userId,
  onExplain,
  openOnboarding,
}) => {
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const fetchRecs = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getRecommendations(userId, 16);
      setRecommendations(res.recommendations || []);
    } catch (err: any) {
      setError(err.message || "Failed to load recommendations");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecs();
  }, [userId]);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  const handleLike = async (movieId: number) => {
    try {
      await api.rateMovie(userId, movieId, 5.0);
      showToast("Đã thêm vào danh sách yêu thích. Đồ thị tri thức đã được cập nhật!");
      // Trigger background re-fetch after 1s
      setTimeout(fetchRecs, 1000);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleDislike = async (movieId: number) => {
    try {
      await api.rateMovie(userId, movieId, 1.0);
      showToast("Đã đánh dấu không thích và loại trừ khỏi danh sách gợi ý.");
      // Filter out of current state
      setRecommendations((prev) => prev.filter((m) => m.movieId !== movieId));
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleRate = async (movieId: number, rating: number) => {
    try {
      await api.rateMovie(userId, movieId, rating);
      showToast(`Đã ghi nhận đánh giá ${rating}★ cho Người dùng #${userId}`);
    } catch (err: any) {
      console.error(err);
    }
  };

  // Spotlight movie is the first one with the highest score
  const spotlight = recommendations.length > 0 ? recommendations[0] : null;
  const gridMovies = recommendations.length > 0 ? recommendations.slice(1) : [];

  return (
    <div className="space-y-8 pb-16">
      {/* Toast feedback */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 px-4 py-2.5 rounded-xl bg-card border border-primary/40 text-xs font-semibold text-white shadow-2xl shadow-black animate-in slide-in-from-bottom-3 duration-200 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Hero Spotlight (Bento Header) */}
      {spotlight && !loading && (
        <div className="relative rounded-3xl overflow-hidden glass-panel border border-white/10 p-6 sm:p-10 shadow-2xl">
          <div className="absolute inset-0 bg-gradient-to-r from-background via-background/80 to-transparent z-10" />
          {spotlight.posterUrl && (
            <img
              src={spotlight.posterUrl}
              alt={spotlight.title}
              className="absolute right-0 top-0 h-full w-2/3 object-cover opacity-20 filter blur-sm"
            />
          )}

          <div className="relative z-20 max-w-2xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/30 text-primary text-xs font-bold font-mono">
              <Sparkles className="w-3.5 h-3.5" />
              <span>PHIM PHÙ HỢP NHẤT • ĐỘ TƯƠNG ĐỒNG {Math.round(spotlight.score * 100)}%</span>
            </div>

            <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
              {spotlight.title}
            </h1>

            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300 font-mono">
              <span>{spotlight.releaseYear || "N/A"}</span>
              <span>•</span>
              <span className="text-primary font-semibold">
                {(spotlight.genres || []).join(", ")}
              </span>
            </div>

            {spotlight.reasons && spotlight.reasons[0] && (
              <p className="text-sm text-slate-300 italic border-l-2 border-primary/60 pl-3 leading-relaxed">
                "{spotlight.reasons[0]}"
              </p>
            )}

            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={() => onExplain(spotlight.movieId)}
                className="px-5 py-2.5 rounded-xl bg-primary text-black font-bold text-xs hover:brightness-110 active:scale-95 transition-all shadow-glow-gold flex items-center gap-2"
              >
                <HelpCircle className="w-4 h-4" />
                <span>Xem lý giải đề xuất</span>
              </button>
              <button
                onClick={() => handleLike(spotlight.movieId)}
                className="px-4 py-2.5 rounded-xl bg-white/5 hover:bg-accent hover:text-white border border-white/10 text-white font-semibold text-xs active:scale-95 transition-all flex items-center gap-1.5"
              >
                <Heart className="w-4 h-4" />
                <span>Thích</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Section Header & Refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white">
            Danh Sách Đề Xuất Tuyển Chọn
          </h2>
          <p className="text-xs text-slate-400">
            Tính toán bởi Mạng Chú ý Tri thức Liên kết (CKAN) & Neo4j
          </p>
        </div>

        <button
          onClick={fetchRecs}
          disabled={loading}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-xs font-semibold text-slate-300 hover:text-white hover:border-white/25 transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-primary" : ""}`} />
          <span>Làm mới gợi ý</span>
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-6 rounded-2xl bg-destructive/10 border border-destructive/30 flex flex-col items-center text-center">
          <AlertCircle className="w-8 h-8 text-destructive mb-2" />
          <p className="text-sm font-bold text-white">Không thể tạo danh sách đề xuất</p>
          <p className="text-xs text-slate-400 mt-1">{error}</p>
          <button
            onClick={openOnboarding}
            className="mt-4 px-4 py-2 rounded-xl bg-primary text-black text-xs font-bold shadow-glow-gold"
          >
            Thiết lập gu sở thích (Khởi tạo mới)
          </button>
        </div>
      )}

      {/* Loading Skeleton Grid */}
      {loading && recommendations.length === 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
            <div
              key={i}
              className="rounded-2xl bg-card border border-white/5 aspect-[2/3] animate-pulse p-4 flex flex-col justify-end"
            >
              <div className="h-4 bg-slate-800 rounded w-3/4 mb-2" />
              <div className="h-3 bg-slate-850 rounded w-1/2" />
            </div>
          ))}
        </div>
      )}

      {/* Main Movie Grid */}
      {!loading && gridMovies.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {gridMovies.map((movie) => (
            <MovieCard
              key={movie.movieId}
              movie={movie}
              userId={userId}
              onLike={handleLike}
              onDislike={handleDislike}
              onRate={handleRate}
              onExplain={onExplain}
            />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && recommendations.length === 0 && (
        <div className="p-12 text-center border border-dashed border-white/10 rounded-2xl">
          <Film className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <h3 className="text-base font-bold text-white">Chưa có dữ liệu đề xuất</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1">
            Người dùng #{userId} chưa có lịch sử tương tác. Hãy thiết lập gu sở thích để bắt đầu trải nghiệm!
          </p>
          <button
            onClick={openOnboarding}
            className="mt-4 px-5 py-2.5 rounded-xl bg-primary text-black text-xs font-bold shadow-glow-gold"
          >
            Thiết lập gu sở thích ngay
          </button>
        </div>
      )}
    </div>
  );
};

