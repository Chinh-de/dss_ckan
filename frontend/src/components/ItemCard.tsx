import React, { useState } from "react";
import { Heart, ThumbsDown, Network, Film, BookOpen, Music2, ExternalLink } from "lucide-react";
import { RecommendationItem, DomainType } from "../types";

interface ItemCardProps {
  item: RecommendationItem;
  domain: DomainType;
  accentColor: string;
  userId: number;
  onLike: (itemId: number) => void;
  onDislike: (itemId: number) => void;
  onExplain: (itemId: number) => void;
}

export const ItemCard: React.FC<ItemCardProps> = ({
  item,
  domain,
  accentColor,
  userId,
  onLike,
  onDislike,
  onExplain,
}) => {
  const [isLiked, setIsLiked] = useState(false);
  const [isDisliked, setIsDisliked] = useState(false);

  const itemId = item.id ?? item.movieId;

  const handleLike = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsLiked(!isLiked);
    setIsDisliked(false);
    onLike(itemId);
  };

  const handleDislike = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDisliked(true);
    setIsLiked(false);
    onDislike(itemId);
  };

  const renderDomainIcon = () => {
    if (domain === "book") return <BookOpen className="w-8 h-8 opacity-40 text-slate-400" />;
    if (domain === "music") return <Music2 className="w-8 h-8 opacity-40 text-slate-400" />;
    return <Film className="w-8 h-8 opacity-40 text-slate-400" />;
  };

  if (isDisliked) {
    return (
      <div className="rounded-2xl border border-dashed border-rose-500/20 bg-rose-500/5 p-4 flex flex-col items-center justify-center text-center transition-all duration-300 min-h-[340px]">
        <ThumbsDown className="w-7 h-7 text-rose-400/60 mb-2" />
        <p className="text-xs font-semibold text-rose-300">Đã ẩn khỏi danh sách</p>
        <p className="text-[11px] text-slate-400 mt-1 max-w-[180px]">
          Mục này sẽ không xuất hiện trong đề xuất tiếp theo của bạn.
        </p>
        <button
          onClick={() => setIsDisliked(false)}
          className="mt-3 text-[11px] text-slate-300 underline hover:text-white"
        >
          Hoàn tác
        </button>
      </div>
    );
  }

  const scorePct = Math.round(item.score * 100);

  return (
    <div className="rounded-2xl border border-white/10 bg-[#10121A] overflow-hidden flex flex-col group relative transition-all duration-200 hover:border-white/20 hover:shadow-lg">
      {/* Poster / Visual Frame */}
      <div className="relative aspect-[3/4] w-full bg-slate-900/80 overflow-hidden">
        {item.posterUrl ? (
          <img
            src={item.posterUrl}
            alt={item.title}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            loading="lazy"
            onError={(e) => {
              // Fallback to placeholder if image link breaks
              (e.target as HTMLElement).style.display = "none";
            }}
          />
        ) : null}

        {/* Fallback graphic if no poster or failed */}
        <div className="absolute inset-0 -z-1 flex flex-col items-center justify-center bg-gradient-to-b from-slate-800/40 to-slate-950 p-4 text-center">
          {renderDomainIcon()}
          <p className="text-xs font-bold text-slate-300 line-clamp-2 mt-2">{item.title}</p>
          {item.subtitle && (
            <p className="text-[10px] text-slate-400 font-mono mt-0.5 line-clamp-1">{item.subtitle}</p>
          )}
        </div>

        {/* Affinity Score Badge */}
        <div className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded-md bg-black/80 backdrop-blur-md border border-white/15 font-mono font-bold text-xs text-white shadow-md flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: accentColor }} />
          <span>{scorePct}%</span>
        </div>

        {/* Hover Action Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-end p-3 gap-2">
          <button
            onClick={() => onExplain(itemId)}
            className="w-full py-2 px-3 rounded-xl text-black text-xs font-bold shadow-lg transition-all flex items-center justify-center gap-1.5 active:scale-95 hover:brightness-110"
            style={{ backgroundColor: accentColor }}
          >
            <Network className="w-3.5 h-3.5" />
            <span>Đồ thị lý giải</span>
          </button>
        </div>
      </div>

      {/* Info Content */}
      <div className="p-3.5 flex-1 flex flex-col justify-between space-y-2">
        <div>
          <div className="flex items-center justify-between gap-1 text-[11px] text-slate-400 font-mono">
            <span className="truncate">{item.secondaryInfo || item.releaseYear || domain.toUpperCase()}</span>
            <span
              className="text-[10px] font-mono px-1.5 py-0.2 rounded uppercase tracking-wider shrink-0 border"
              style={{
                color: accentColor,
                borderColor: `${accentColor}30`,
                backgroundColor: `${accentColor}10`,
              }}
            >
              {domain}
            </span>
          </div>

          <h3 className="font-bold text-xs text-white line-clamp-1 mt-1 group-hover:text-slate-100 transition-colors">
            {item.title}
          </h3>

          {item.subtitle && (
            <p className="text-[11px] text-slate-400 truncate mt-0.5">{item.subtitle}</p>
          )}

          {/* Reasoning preview pill */}
          {item.reasons && item.reasons[0] && (
            <button
              onClick={() => onExplain(itemId)}
              className="w-full mt-2 py-1 px-2 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] border border-white/5 hover:border-white/15 transition-all flex items-center justify-between text-left group/reason"
              title="Nhấn để xem trực quan hóa đồ thị lý giải"
            >
              <div className="flex items-center gap-1.5 text-[11px] text-slate-300 truncate font-mono">
                <Network className="w-3 h-3 shrink-0" style={{ color: accentColor }} />
                <span className="truncate">{item.reasons[0]}</span>
              </div>
              <span className="text-[10px] text-slate-400 shrink-0 font-sans group-hover/reason:translate-x-0.5 transition-transform">
                →
              </span>
            </button>
          )}
        </div>

        {/* Footer Interaction Bar */}
        <div className="pt-2 border-t border-white/5 flex items-center justify-between">
          <button
            onClick={() => onExplain(itemId)}
            className="text-[11px] text-slate-400 hover:text-white transition-colors flex items-center gap-1 font-mono"
          >
            <span>Chi tiết KG</span>
          </button>

          <div className="flex items-center gap-1">
            <button
              onClick={handleLike}
              title="Thích và cập nhật gu sở thích"
              className={`p-1.5 rounded-lg border transition-all active:scale-90 ${
                isLiked
                  ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                  : "bg-slate-900/60 text-slate-400 border-white/5 hover:text-white hover:border-white/20"
              }`}
            >
              <Heart className={`w-3.5 h-3.5 ${isLiked ? "fill-rose-400 text-rose-400" : ""}`} />
            </button>

            <button
              onClick={handleDislike}
              title="Không thích và loại trừ khỏi đề xuất"
              className="p-1.5 rounded-lg bg-slate-900/60 text-slate-400 border border-white/5 hover:text-rose-400 hover:border-rose-500/30 transition-all active:scale-90"
            >
              <ThumbsDown className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
