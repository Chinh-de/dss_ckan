import React, { useState } from "react";
import { Heart, EyeOff, Waypoints } from "lucide-react";
import { RecommendationItem, DomainType } from "../types";
import { cn } from "../lib/utils";
import { Poster, ScoreMeter, posterAspect } from "./ui";

interface ItemCardProps {
  item: RecommendationItem;
  domain: DomainType;
  rank: number;
  style?: React.CSSProperties;
  onLike: (itemId: number) => void;
  onDislike: (itemId: number) => void;
  onExplain: (itemId: number) => void;
}

export const ItemCard: React.FC<ItemCardProps> = ({ item, domain, rank, style, onLike, onDislike, onExplain }) => {
  const [isLiked, setIsLiked] = useState(false);
  const itemId = item.id ?? item.movieId!;
  const meta = [item.subtitle, item.secondaryInfo || item.releaseYear].filter(Boolean).join(" · ");
  const reason = item.reasons?.[0];

  return (
    <article className="group flex animate-fade-up flex-col" style={style}>
      <button
        onClick={() => onExplain(itemId)}
        className="relative block overflow-hidden rounded-lg text-left"
        aria-label={`Xem lý giải cho ${item.title}`}
      >
        <Poster
          src={item.posterUrl}
          title={item.title}
          domain={domain}
          className={cn("w-full", posterAspect(domain))}
          imgClassName="transition-transform duration-500 ease-out group-hover:scale-[1.04]"
        />
        <span className="num absolute left-0 top-0 rounded-br-lg bg-bg/85 px-2 py-1 text-xs text-ink backdrop-blur-sm">
          {String(rank).padStart(2, "0")}
        </span>
        <span className="absolute inset-x-0 bottom-0 flex translate-y-2 items-center gap-1.5 bg-gradient-to-t from-bg via-bg/80 to-transparent px-3 pb-2.5 pt-8 text-sm font-medium text-ink opacity-0 transition duration-300 group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100">
          <Waypoints className="h-4 w-4 text-accent" strokeWidth={1.75} />
          Xem lý giải
        </span>
      </button>

      <div className="flex flex-1 flex-col pt-3">
        <h3 className="line-clamp-2 font-sans text-[15px] font-semibold leading-snug text-ink">{item.title}</h3>
        {meta && <p className="mt-0.5 truncate text-[13px] text-ink-muted">{meta}</p>}
        {reason && <p className="mt-1.5 line-clamp-2 text-[13px] leading-snug text-ink-faint">{reason}</p>}

        <div className="mt-auto flex items-center gap-1 pt-3">
          <ScoreMeter score={item.score} className="flex-1" />
          <button
            onClick={() => {
              if (!isLiked) onLike(itemId);
              setIsLiked(true);
            }}
            aria-pressed={isLiked}
            title={isLiked ? "Đã thích" : "Thích"}
            className={cn("icon-btn h-8 w-8", isLiked && "text-neg hover:text-neg")}
          >
            <Heart className={cn("h-4 w-4", isLiked && "fill-current")} strokeWidth={1.75} />
          </button>
          <button onClick={() => onDislike(itemId)} title="Ẩn khỏi gợi ý" className="icon-btn h-8 w-8">
            <EyeOff className="h-4 w-4" strokeWidth={1.75} />
          </button>
        </div>
      </div>
    </article>
  );
};
