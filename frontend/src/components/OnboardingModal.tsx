import React, { useState, useEffect } from "react";
import { X, Check, Sparkles, Film, ArrowRight, Layers, Loader2 } from "lucide-react";
import confetti from "canvas-confetti";
import { Movie } from "../types";
import { api } from "../services/api";
import { formatYear } from "../lib/utils";

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: () => void;
  userId: number;
  onComplete: () => void;
}

const AVAILABLE_GENRES = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", 
  "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"
];

export const OnboardingModal: React.FC<OnboardingModalProps> = ({
  isOpen,
  onClose,
  userId,
  onComplete,
}) => {
  const [selectedGenres, setSelectedGenres] = useState<string[]>(["Action", "Sci-Fi"]);
  const [candidateMovies, setCandidateMovies] = useState<Movie[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<number[]>([]);
  const [moviesLoading, setMoviesLoading] = useState(false);
  const [savingLoading, setSavingLoading] = useState(false);

  // Dynamically fetch candidate movies matching selected genres
  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;
    const fetchCandidates = async () => {
      setMoviesLoading(true);
      try {
        const movies = await api.getOnboardingCandidates(selectedGenres, 12);
        if (isMounted) {
          setCandidateMovies(movies);
          // Pre-select the first movie if nothing is selected yet
          if (selectedMovies.length === 0 && movies.length > 0) {
            setSelectedMovies([movies[0].id]);
          }
        }
      } catch (err) {
        console.error("Failed to load onboarding candidate movies:", err);
      } finally {
        if (isMounted) setMoviesLoading(false);
      }
    };

    const timer = setTimeout(fetchCandidates, 150);
    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [isOpen, selectedGenres]);

  if (!isOpen) return null;

  const toggleGenre = (genre: string) => {
    setSelectedGenres((prev) => {
      const exists = prev.includes(genre);
      if (exists) {
        // Keep at least 1 genre selected
        return prev.length > 1 ? prev.filter((g) => g !== genre) : prev;
      } else {
        return [...prev, genre];
      }
    });
  };

  const toggleMovie = (id: number) => {
    setSelectedMovies((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  };

  const handleFinish = async () => {
    if (selectedMovies.length === 0) {
      alert("Vui lòng chọn ít nhất 1 bộ phim bạn thích để hệ thống khởi tạo gợi ý!");
      return;
    }

    try {
      setSavingLoading(true);
      await api.completeOnboarding(userId, selectedGenres, selectedMovies);
      confetti({
        particleCount: 90,
        spread: 75,
        origin: { y: 0.6 },
        colors: ["#E5A93C", "#3B82F6", "#10B981", "#FFFFFF"],
      });
      onComplete();
      onClose();
    } catch (err: any) {
      alert(`Lỗi khởi tạo hồ sơ: ${err.message}`);
    } finally {
      setSavingLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-[#0F121A] border border-white/15 rounded-3xl p-6 sm:p-8 shadow-2xl shadow-black/95 text-white max-h-[92vh] flex flex-col">
        {/* Close Button */}
        <button
          onClick={onClose}
          aria-label="Close modal"
          className="absolute top-5 right-5 p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3.5 mb-5 pb-4 border-b border-white/10 shrink-0">
          <div className="w-11 h-11 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shadow-glow-amber shrink-0">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-lg sm:text-xl font-bold tracking-tight">Thiết Lập Gu Điện Ảnh (Taste Profile)</h2>
              <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary border border-primary/30 text-[10px] font-mono font-bold">
                User #{userId}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Chọn thể loại & phim bạn yêu thích để khởi động thuật toán lan truyền CKAN
            </p>
          </div>
        </div>

        {/* Modal Body (Scrollable) */}
        <div className="flex-1 overflow-y-auto space-y-6 pr-1 my-1">
          {/* Step 1: Genres Selection */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono flex items-center gap-2">
                <Layers className="w-4 h-4 text-primary" />
                <span>1. Chọn thể loại bạn yêu thích ({selectedGenres.length} thể loại)</span>
              </label>
              <span className="text-[10px] text-slate-400 font-mono">Bấm để bật/tắt</span>
            </div>

            <div className="flex flex-wrap gap-2">
              {AVAILABLE_GENRES.map((genre) => {
                const active = selectedGenres.includes(genre);
                return (
                  <button
                    key={genre}
                    type="button"
                    onClick={() => toggleGenre(genre)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all active:scale-95 ${
                      active
                        ? "bg-primary text-black border-primary font-bold shadow-glow-amber scale-105"
                        : "bg-slate-900/90 text-slate-300 border-white/10 hover:border-white/20 hover:text-white"
                    }`}
                  >
                    {genre}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Step 2: Dynamically Filtered Candidate Movies */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono flex items-center gap-2">
                <Film className="w-4 h-4 text-primary" />
                <span>2. Chọn các tác phẩm bạn từng xem & thích ({selectedMovies.length} phim đã chọn)</span>
              </label>
              {moviesLoading && (
                <div className="flex items-center gap-1 text-[11px] text-primary font-mono">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Đang tải phim phù hợp...</span>
                </div>
              )}
            </div>

            <p className="text-[11px] text-slate-400 mb-3">
              Danh sách phim bên dưới được tự động cập nhật theo các thể loại bạn vừa chọn ở trên:
            </p>

            {moviesLoading && candidateMovies.length === 0 ? (
              <div className="py-14 flex flex-col items-center justify-center text-center">
                <Loader2 className="w-8 h-8 text-primary animate-spin mb-2" />
                <span className="text-xs text-slate-400 font-mono">Đang tìm các tác phẩm kinh điển...</span>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {candidateMovies.map((movie) => {
                  const active = selectedMovies.includes(movie.id);
                  return (
                    <div
                      key={movie.id}
                      onClick={() => toggleMovie(movie.id)}
                      className={`relative rounded-2xl border p-2.5 cursor-pointer transition-all duration-200 flex flex-col justify-between group ${
                        active
                          ? "bg-primary/10 border-primary shadow-glow-amber ring-1 ring-primary"
                          : "bg-slate-900/70 border-white/10 hover:border-white/25 hover:bg-slate-800/60"
                      }`}
                    >
                      {/* Poster Thumbnail */}
                      <div className="relative aspect-[2/3] w-full rounded-xl overflow-hidden bg-black/60 mb-2">
                        {movie.posterUrl ? (
                          <img
                            src={movie.posterUrl}
                            alt={movie.title}
                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                            loading="lazy"
                          />
                        ) : (
                          <div className="w-full h-full flex flex-col items-center justify-center p-2 text-center bg-slate-950">
                            <Film className="w-6 h-6 text-slate-600 mb-1" />
                            <p className="text-[10px] text-slate-400 line-clamp-2">{movie.title}</p>
                          </div>
                        )}

                        {/* Check Indicator Overlay */}
                        <div
                          className={`absolute top-2 right-2 w-6 h-6 rounded-full flex items-center justify-center transition-all ${
                            active
                              ? "bg-primary text-black shadow-md scale-100"
                              : "bg-black/60 text-white/50 border border-white/20 scale-90 opacity-0 group-hover:opacity-100"
                          }`}
                        >
                          <Check className="w-3.5 h-3.5 stroke-[3]" />
                        </div>
                      </div>

                      {/* Movie Info */}
                      <div>
                        <div className="flex items-center justify-between gap-1 text-[10px] text-slate-400 font-mono mb-0.5">
                          <span>{formatYear(movie.releaseYear)}</span>
                          {movie.genres && movie.genres[0] && (
                            <span className="text-primary/90 font-semibold truncate max-w-[65px]">
                              {movie.genres[0]}
                            </span>
                          )}
                        </div>
                        <h4
                          className={`text-xs font-bold line-clamp-1 transition-colors ${
                            active ? "text-primary" : "text-white group-hover:text-slate-200"
                          }`}
                          title={movie.fullTitle || movie.title}
                        >
                          {movie.title}
                        </h4>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer / Action Button */}
        <div className="flex items-center justify-between pt-4 border-t border-white/10 shrink-0">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white transition-colors"
          >
            Để sau (Bỏ qua)
          </button>

          <button
            type="button"
            onClick={handleFinish}
            disabled={savingLoading || selectedMovies.length === 0}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-primary text-black hover:brightness-110 active:scale-95 transition-all shadow-glow-amber disabled:opacity-40 disabled:pointer-events-none"
          >
            {savingLoading ? (
              <div className="flex items-center gap-2 font-mono">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Đang khởi tạo Feed CKAN...</span>
              </div>
            ) : (
              <>
                <span>Kích hoạt Feed Cá Nhân ({selectedMovies.length} phim)</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
