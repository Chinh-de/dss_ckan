import React, { useState } from "react";
import { Heart, ThumbsDown, Star, Network, Film } from "lucide-react";
import { RecommendationItem } from "../types";
import { formatYear, formatScore } from "../lib/utils";

interface MovieCardProps {
  movie: RecommendationItem;
  userId: number;
  onLike: (movieId: number) => void;
  onDislike: (movieId: number) => void;
  onRate: (movieId: number, rating: number) => void;
  onExplain: (movieId: number) => void;
}

export const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  userId,
  onLike,
  onDislike,
  onRate,
  onExplain,
}) => {
  const [isLiked, setIsLiked] = useState(false);
  const [isDisliked, setIsDisliked] = useState(false);
  const [userRating, setUserRating] = useState<number | null>(null);
  const [hoverRating, setHoverRating] = useState<number | null>(null);

  const handleLike = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsLiked(!isLiked);
    setIsDisliked(false);
    onLike(movie.movieId);
  };

  const handleDislike = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDisliked(true);
    setIsLiked(false);
    onDislike(movie.movieId);
  };

  const handleStarClick = (star: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setUserRating(star);
    if (star >= 3.5) {
      setIsLiked(true);
      setIsDisliked(false);
    } else {
      setIsDisliked(true);
      setIsLiked(false);
    }
    onRate(movie.movieId, star);
  };

  if (isDisliked) {
    return (
      <div className="rounded-2xl border border-dashed border-destructive/30 bg-destructive/5 p-4 flex flex-col items-center justify-center text-center transition-all duration-300 min-h-[360px]">
        <ThumbsDown className="w-8 h-8 text-destructive/60 mb-2" />
        <p className="text-xs font-semibold text-destructive">Đã ẩn & Không gợi ý lại</p>
        <p className="text-[11px] text-slate-400 mt-1">Phim này sẽ không còn xuất hiện trong danh sách đề xuất của bạn.</p>
        <button
          onClick={() => setIsDisliked(false)}
          className="mt-3 text-[11px] text-slate-300 underline hover:text-white"
        >
          Hoàn tác
        </button>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-2xl overflow-hidden flex flex-col group relative">
      {/* Poster / Visual Frame */}
      <div className="relative aspect-[2/3] w-full bg-slate-900 overflow-hidden">
        {movie.posterUrl ? (
          <img
            src={movie.posterUrl}
            alt={movie.title}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-slate-800/40 to-slate-950 p-4 text-center">
            <Film className="w-10 h-10 text-slate-600 mb-2" />
            <p className="text-xs font-bold text-slate-300 line-clamp-2">{movie.title}</p>
            <p className="text-[10px] text-slate-500 font-mono mt-1">{formatYear(movie.releaseYear)}</p>
          </div>
        )}

        {/* Affinity Match Badge */}
        <div className="absolute top-2.5 right-2.5 px-2.5 py-1 rounded-full bg-black/75 backdrop-blur-md border border-primary/40 text-primary font-mono font-bold text-xs shadow-lg">
          {formatScore(movie.score)}
        </div>

        {/* Hover Action Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-end p-3.5 gap-2">
          {/* Visual Graph Reasoning button */}
          <button
            onClick={() => onExplain(movie.movieId)}
            className="w-full py-2 px-3 rounded-xl bg-primary text-black hover:bg-primary/90 text-xs font-bold shadow-lg shadow-black/80 transition-all flex items-center justify-center gap-1.5 active:scale-95"
          >
            <Network className="w-3.5 h-3.5" />
            <span>Xem đồ thị lý giải</span>
          </button>
        </div>
      </div>

      {/* Info Content */}
      <div className="p-4 flex-1 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-mono text-slate-400">
              {formatYear(movie.releaseYear)}
            </span>
            {movie.genres && movie.genres[0] && (
              <span className="text-[10px] uppercase font-semibold tracking-wider text-primary/80 truncate max-w-[120px]">
                {movie.genres[0]}
              </span>
            )}
          </div>
          <h3 className="font-bold text-sm text-white line-clamp-1 mt-0.5 group-hover:text-primary transition-colors">
            {movie.title}
          </h3>

          {/* Reasoning visual path preview pill */}
          {movie.reasons && movie.reasons[0] && (
            <button
              onClick={() => onExplain(movie.movieId)}
              className="w-full mt-2 py-1 px-2.5 rounded-lg bg-white/[0.04] hover:bg-primary/15 border border-white/5 hover:border-primary/30 transition-all flex items-center justify-between text-left group/reason"
              title="Nhấn để xem trực quan hóa đồ thị lý giải"
            >
              <div className="flex items-center gap-1.5 text-[11px] text-slate-300 truncate font-mono">
                <Network className="w-3 h-3 text-primary shrink-0" />
                <span className="truncate">{movie.reasons[0]}</span>
              </div>
              <span className="text-[10px] text-primary shrink-0 font-sans font-semibold group-hover/reason:translate-x-0.5 transition-transform flex items-center">
                Đồ thị →
              </span>
            </button>
          )}
        </div>

        {/* Interaction Bar: Like / Dislike / Star Rating */}
        <div className="pt-3 mt-3 border-t border-white/5 flex items-center justify-between">
          {/* 5-star rating */}
          <div className="flex items-center gap-0.5" title="Đánh giá phim này">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                aria-label={`Đánh giá ${star} sao`}
                onClick={(e) => handleStarClick(star, e)}
                onMouseEnter={() => setHoverRating(star)}
                onMouseLeave={() => setHoverRating(null)}
                className="p-0.5 text-slate-500 hover:text-primary transition-colors"
              >
                <Star
                  className={`w-3.5 h-3.5 ${
                    (hoverRating !== null ? star <= hoverRating : userRating !== null && star <= userRating)
                      ? "fill-primary text-primary"
                      : "text-slate-600"
                  }`}
                />
              </button>
            ))}
          </div>

          {/* Like / Dislike toggles */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={handleLike}
              aria-label="Thích phim"
              title="Thích phim này"
              className={`p-1.5 rounded-lg border transition-all active:scale-90 ${
                isLiked
                  ? "bg-accent text-white border-accent shadow-glow-crimson"
                  : "bg-slate-900/60 text-slate-400 border-white/5 hover:text-white hover:border-white/20"
              }`}
            >
              <Heart className={`w-3.5 h-3.5 ${isLiked ? "fill-white" : ""}`} />
            </button>

            <button
              onClick={handleDislike}
              aria-label="Không thích phim"
              title="Không thích và không gợi ý lại"
              className="p-1.5 rounded-lg bg-slate-900/60 text-slate-400 border border-white/5 hover:text-destructive hover:border-destructive/30 transition-all active:scale-90"
            >
              <ThumbsDown className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

