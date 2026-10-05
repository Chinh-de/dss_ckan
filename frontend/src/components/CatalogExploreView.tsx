import React, { useEffect, useState } from "react";
import { Search, Film, BookOpen, Music2, ChevronLeft, ChevronRight, Heart, Check } from "lucide-react";
import { ItemDto, DomainType } from "../types";
import { api } from "../services/api";

interface CatalogExploreViewProps {
  domain: DomainType;
  accentColor: string;
  userId: number;
  onLike: (itemId: number) => void;
}

export const CatalogExploreView: React.FC<CatalogExploreViewProps> = ({
  domain,
  accentColor,
  userId,
  onLike,
}) => {
  const [items, setItems] = useState<ItemDto[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [likedIds, setLikedIds] = useState<Set<number>>(new Set());

  const limit = 20;

  const fetchItems = async () => {
    try {
      setLoading(true);
      const res = await api.getCatalogItems(domain, page, limit, search || undefined);
      setItems(res.data);
      setTotal(res.total);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchItems();
  }, [domain, page]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchItems();
  };

  const handleLikeItem = (id: number) => {
    setLikedIds((prev) => new Set(prev).add(id));
    onLike(id);
  };

  const domainIcon = () => {
    if (domain === "book") return <BookOpen className="w-4 h-4" />;
    if (domain === "music") return <Music2 className="w-4 h-4" />;
    return <Film className="w-4 h-4" />;
  };

  const domainTitle = {
    movie: "Kho phim điện ảnh (MovieLens-20M)",
    book: "Thư viện sách (Book-Crossing)",
    music: "Danh mục nghệ sĩ (Last.FM)",
  }[domain];

  const totalPages = Math.ceil(total / limit) || 1;

  return (
    <div className="space-y-6 pb-16">
      {/* Header & Search Bar */}
      <div className="rounded-2xl border border-white/10 bg-[#10121A] p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span
                className="px-2.5 py-0.5 rounded-md text-[11px] font-mono font-medium tracking-wide uppercase border flex items-center gap-1.5"
                style={{
                  color: accentColor,
                  borderColor: `${accentColor}40`,
                  backgroundColor: `${accentColor}15`,
                }}
              >
                {domainIcon()}
                <span>{domainTitle}</span>
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Tổng cộng: <strong className="text-white">{total.toLocaleString()}</strong> mục
              </span>
            </div>
            <h1 className="text-xl font-bold text-white tracking-tight">
              Khám phá danh mục & Xây dựng hồ sơ sở thích
            </h1>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
              Tìm kiếm các tác phẩm bạn yêu thích để bổ sung vào hồ sơ Người dùng #{userId}. Hệ thống sẽ ngay lập tức
              cập nhật thuật toán lan truyền Attention trên Knowledge Graph mà không cần huấn luyện lại.
            </p>
          </div>

          {/* Search Form */}
          <form onSubmit={handleSearchSubmit} className="relative min-w-[280px]">
            <input
              type="text"
              placeholder={`Tìm kiếm ${domain === "music" ? "nghệ sĩ..." : domain === "book" ? "tên sách, tác giả..." : "tên phim..."}`}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-900 border border-white/10 focus:border-white/30 text-xs text-white placeholder-slate-500 focus:outline-none transition-colors"
            />
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          </form>
        </div>
      </div>

      {/* Grid of Items */}
      {loading && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 animate-pulse">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
            <div key={i} className="h-64 rounded-2xl bg-white/5 border border-white/5" />
          ))}
        </div>
      )}

      {!loading && items.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {items.map((item) => {
            const isLiked = likedIds.has(item.id);
            return (
              <div
                key={item.id}
                className="rounded-2xl border border-white/10 bg-[#10121A] overflow-hidden flex flex-col group hover:border-white/20 transition-all shadow-sm"
              >
                {/* Poster Frame */}
                <div className="relative aspect-[3/4] w-full bg-slate-900/80 overflow-hidden">
                  {item.posterUrl ? (
                    <img
                      src={item.posterUrl}
                      alt={item.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      loading="lazy"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = "none";
                      }}
                    />
                  ) : null}

                  <div className="absolute inset-0 -z-1 flex flex-col items-center justify-center p-4 text-center bg-gradient-to-b from-slate-800/30 to-slate-950">
                    {domainIcon()}
                    <p className="text-xs font-bold text-slate-300 line-clamp-2 mt-2">{item.title}</p>
                    {item.subtitle && (
                      <p className="text-[10px] text-slate-400 font-mono mt-0.5 line-clamp-1">{item.subtitle}</p>
                    )}
                  </div>
                </div>

                {/* Meta details & Like Action */}
                <div className="p-3.5 flex-1 flex flex-col justify-between space-y-2">
                  <div>
                    <span className="text-[10px] text-slate-400 font-mono block truncate">
                      {item.category || item.releaseYear || domain.toUpperCase()}
                    </span>
                    <h3 className="font-bold text-xs text-white line-clamp-1 mt-0.5 group-hover:text-slate-200">
                      {item.title}
                    </h3>
                    {item.subtitle && (
                      <p className="text-[11px] text-slate-400 truncate mt-0.5">{item.subtitle}</p>
                    )}
                  </div>

                  <div className="pt-2 border-t border-white/5 flex items-center justify-between">
                    <span className="text-[10px] font-mono text-slate-500">#{item.id}</span>
                    <button
                      onClick={() => handleLikeItem(item.id)}
                      className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all active:scale-95 flex items-center gap-1.5 border ${
                        isLiked
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                          : "bg-white/5 hover:bg-white/10 text-slate-300 border-white/10 hover:text-white"
                      }`}
                    >
                      {isLiked ? (
                        <>
                          <Check className="w-3 h-3 text-emerald-400" />
                          <span>Đã thích</span>
                        </>
                      ) : (
                        <>
                          <Heart className="w-3 h-3 text-rose-400" />
                          <span>Thích</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination Controls */}
      {!loading && totalPages > 1 && (
        <div className="flex items-center justify-between pt-4 border-t border-white/5">
          <p className="text-xs text-slate-400 font-mono">
            Trang {page} / {totalPages} (Tổng {total.toLocaleString()} mục)
          </p>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="p-2 rounded-xl bg-[#10121A] border border-white/10 text-slate-300 hover:text-white disabled:opacity-40 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-xs font-mono font-bold text-white px-2">
              {page}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="p-2 rounded-xl bg-[#10121A] border border-white/10 text-slate-300 hover:text-white disabled:opacity-40 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
