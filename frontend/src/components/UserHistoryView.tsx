import React, { useState, useEffect } from "react";
import {
  History,
  ThumbsUp,
  ThumbsDown,
  Star,
  Search,
  Filter,
  UserCheck,
  Sparkles,
  Network,
  Film,
  Calendar,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
} from "lucide-react";
import { UserHistoryItem, UserProfile } from "../types";
import { api } from "../services/api";
import { formatYear } from "../lib/utils";

interface UserHistoryViewProps {
  userId: number;
  onExplain: (movieId: number) => void;
  onChangeUser: () => void;
  onOpenOnboarding: () => void;
}

export const UserHistoryView: React.FC<UserHistoryViewProps> = ({
  userId,
  onExplain,
  onChangeUser,
  onOpenOnboarding,
}) => {
  const [historyItems, setHistoryItems] = useState<UserHistoryItem[]>([]);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState<"ALL" | "LIKE" | "DISLIKE">("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedGenre, setSelectedGenre] = useState<string>("ALL");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);

  const PAGE_SIZE = 18;

  // Load history data
  useEffect(() => {
    let isMounted = true;
    const fetchHistory = async () => {
      setLoading(true);
      try {
        const res = await api.getUserHistory(
          userId,
          filterType,
          searchQuery.trim() || undefined,
          selectedGenre === "ALL" ? undefined : selectedGenre,
          currentPage,
          PAGE_SIZE
        );
        if (isMounted) {
          setHistoryItems(res.items || []);
          setUserProfile(res.user || null);
          setTotalItems(res.total || 0);
          setTotalPages(Math.max(1, Math.ceil((res.total || 0) / PAGE_SIZE)));
        }
      } catch (err) {
        console.error("Failed to load user history:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    const timer = setTimeout(fetchHistory, searchQuery ? 250 : 0);
    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [userId, filterType, searchQuery, selectedGenre, currentPage]);

  // Reset page when filter changes
  const handleFilterChange = (type: "ALL" | "LIKE" | "DISLIKE") => {
    setFilterType(type);
    setCurrentPage(1);
  };

  const handleGenreChange = (genre: string) => {
    setSelectedGenre(genre);
    setCurrentPage(1);
  };

  return (
    <div className="space-y-8 pb-16 animate-in fade-in duration-300">
      {/* 1. Hero User Profile Banner */}
      <div className="rounded-3xl bg-gradient-to-r from-slate-900 via-[#0F131E] to-slate-900 border border-white/10 p-6 sm:p-8 shadow-2xl relative overflow-hidden">
        {/* Background glow orb */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          {/* User Info */}
          <div className="flex items-center gap-4 sm:gap-5">
            <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-gradient-to-br from-primary/20 via-primary/10 to-transparent border border-primary/40 flex items-center justify-center font-mono font-bold text-2xl sm:text-3xl text-primary shadow-glow-amber shrink-0">
              #{userId}
            </div>

            <div className="min-w-0">
              <div className="flex items-center gap-2.5 flex-wrap">
                <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight truncate">
                  {userProfile?.name || `Người Yêu Phim #${userId}`}
                </h1>
                <span className="px-2.5 py-0.5 rounded-full bg-primary/20 text-primary border border-primary/30 text-xs font-mono font-bold">
                  Đang Hoạt Động
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                {userProfile?.email || `user${userId}@moviekg.ai`}
              </p>

              {/* Top genres badge ribbon */}
              {userProfile?.topGenres && userProfile.topGenres.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 mt-2.5">
                  <span className="text-[11px] text-slate-400 font-mono">Gu yêu thích:</span>
                  {userProfile.topGenres.map((g) => (
                    <span
                      key={g}
                      className="px-2 py-0.5 rounded-md bg-white/5 border border-white/10 text-[11px] font-mono text-amber-300"
                    >
                      {g}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* User Stats & Action Buttons */}
          <div className="flex flex-col sm:flex-row md:flex-col lg:flex-row items-start sm:items-center gap-3 shrink-0">
            {/* Stat Counters */}
            <div className="flex items-center gap-3 p-2 rounded-2xl bg-slate-950/60 border border-white/5 font-mono">
              <div className="px-3 py-1.5 text-center">
                <span className="text-xs text-slate-400 block">Đã đánh giá</span>
                <span className="text-lg font-bold text-white">
                  {userProfile?.totalRatings ?? totalItems}
                </span>
              </div>
              <div className="w-[1px] h-8 bg-white/10" />
              <div className="px-3 py-1.5 text-center">
                <span className="text-xs text-emerald-400 flex items-center justify-center gap-1">
                  <ThumbsUp className="w-3 h-3" /> Like
                </span>
                <span className="text-lg font-bold text-emerald-400">
                  {userProfile?.totalLikes ?? "-"}
                </span>
              </div>
              <div className="w-[1px] h-8 bg-white/10" />
              <div className="px-3 py-1.5 text-center">
                <span className="text-xs text-red-400 flex items-center justify-center gap-1">
                  <ThumbsDown className="w-3 h-3" /> Dislike
                </span>
                <span className="text-lg font-bold text-red-400">
                  {userProfile?.totalDislikes ?? "-"}
                </span>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex items-center gap-2">
              <button
                onClick={onChangeUser}
                className="px-3.5 py-2 rounded-xl bg-primary text-black hover:bg-primary/90 text-xs font-bold transition-all shadow-md active:scale-95 flex items-center gap-1.5"
                title="Đổi sang hồ sơ người dùng khác"
              >
                <UserCheck className="w-3.5 h-3.5" />
                <span>Đổi User</span>
              </button>

              <button
                onClick={onOpenOnboarding}
                className="px-3.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-slate-200 text-xs font-medium transition-all active:scale-95 flex items-center gap-1.5"
                title="Tái tinh chỉnh gu sở thích"
              >
                <Sparkles className="w-3.5 h-3.5 text-primary" />
                <span>Gu sở thích</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Filter & Search Toolbar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 p-2 rounded-2xl bg-slate-900/60 border border-white/5">
        {/* Filter Tabs */}
        <div className="flex items-center p-1 rounded-xl bg-slate-950/80 border border-white/5 shrink-0">
          <button
            onClick={() => handleFilterChange("ALL")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filterType === "ALL"
                ? "bg-primary text-black font-bold shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Tất cả ({userProfile?.totalRatings ?? 0})
          </button>
          <button
            onClick={() => handleFilterChange("LIKE")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filterType === "LIKE"
                ? "bg-emerald-500 text-black font-bold shadow-sm"
                : "text-slate-400 hover:text-emerald-400"
            }`}
          >
            <ThumbsUp className="w-3 h-3" />
            <span>Đã thích ({userProfile?.totalLikes ?? 0})</span>
          </button>
          <button
            onClick={() => handleFilterChange("DISLIKE")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filterType === "DISLIKE"
                ? "bg-red-500 text-white font-bold shadow-sm"
                : "text-slate-400 hover:text-red-400"
            }`}
          >
            <ThumbsDown className="w-3 h-3" />
            <span>Không thích ({userProfile?.totalDislikes ?? 0})</span>
          </button>
        </div>

        {/* Search input in history */}
        <div className="relative flex-1 max-w-sm">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Tìm tên phim trong lịch sử..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full bg-slate-950/80 border border-white/10 rounded-xl pl-9 pr-4 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-primary/50"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-white"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* 3. History Movie Grid */}
      {loading ? (
        <div className="py-24 flex flex-col items-center justify-center text-center">
          <div className="w-10 h-10 rounded-full border-2 border-primary border-t-transparent animate-spin mb-4" />
          <p className="text-xs text-slate-400 font-mono">Đang tải lịch sử tương tác của User #{userId}...</p>
        </div>
      ) : historyItems.length === 0 ? (
        <div className="py-20 rounded-3xl border border-dashed border-white/10 bg-white/[0.02] flex flex-col items-center justify-center text-center p-6">
          <History className="w-12 h-12 text-slate-600 mb-3" />
          <h3 className="text-base font-bold text-white">Chưa có lịch sử tương tác phù hợp</h3>
          <p className="text-xs text-slate-400 max-w-md mt-1 mb-4">
            {searchQuery
              ? `Không tìm thấy phim nào khớp với "${searchQuery}" trong lịch sử của người dùng này.`
              : `Người dùng này chưa có dữ liệu đánh giá phim trong cơ sở dữ liệu.`}
          </p>
          <button
            onClick={onOpenOnboarding}
            className="px-4 py-2 rounded-xl bg-primary text-black font-bold text-xs shadow-glow-amber transition-transform active:scale-95"
          >
            Đánh giá sở thích ngay (Taste Profile)
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {historyItems.map((item) => {
              const isLike = item.interactionType === "LIKE";
              return (
                <div
                  key={item.ratingId}
                  className="glass-card rounded-2xl overflow-hidden flex flex-col group relative bg-slate-900/60 border border-white/10 hover:border-primary/40 transition-all duration-300"
                >
                  {/* Poster Thumbnail */}
                  <div className="relative aspect-[2/3] w-full bg-slate-950 overflow-hidden">
                    {item.posterUrl ? (
                      <img
                        src={item.posterUrl}
                        alt={item.title}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-slate-900 to-slate-950 p-3 text-center">
                        <Film className="w-8 h-8 text-slate-600 mb-1" />
                        <p className="text-[11px] font-bold text-slate-300 line-clamp-2">{item.title}</p>
                      </div>
                    )}

                    {/* Rating Badge Overlay */}
                    <div
                      className={`absolute top-2 right-2 px-2 py-0.5 rounded-lg backdrop-blur-md border text-[11px] font-mono font-bold flex items-center gap-1 shadow-lg ${
                        isLike
                          ? "bg-emerald-950/80 border-emerald-500/40 text-emerald-400"
                          : "bg-red-950/80 border-red-500/40 text-red-400"
                      }`}
                    >
                      {isLike ? (
                        <ThumbsUp className="w-3 h-3" />
                      ) : (
                        <ThumbsDown className="w-3 h-3" />
                      )}
                      <span>{item.rating.toFixed(1)}</span>
                    </div>

                    {/* Hover Action to Explain Graph */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-end p-3">
                      <button
                        onClick={() => onExplain(item.movieId)}
                        className="w-full py-1.5 px-2.5 rounded-xl bg-primary text-black hover:bg-primary/90 text-xs font-bold shadow-lg transition-all flex items-center justify-center gap-1 active:scale-95"
                      >
                        <Network className="w-3.5 h-3.5" />
                        <span>Xem liên kết KG</span>
                      </button>
                    </div>
                  </div>

                  {/* Movie Info */}
                  <div className="p-3 flex-1 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between gap-1 text-[10px] text-slate-400 font-mono mb-1">
                        <span>{formatYear(item.releaseYear)}</span>
                        {item.genres && item.genres[0] && (
                          <span className="text-primary/80 uppercase font-semibold truncate max-w-[80px]">
                            {item.genres[0]}
                          </span>
                        )}
                      </div>
                      <h4
                        className="font-bold text-xs text-white line-clamp-1 group-hover:text-primary transition-colors"
                        title={item.fullTitle || item.title}
                      >
                        {item.title}
                      </h4>
                    </div>

                    {/* Reason button to explain */}
                    <button
                      onClick={() => onExplain(item.movieId)}
                      className="mt-2.5 w-full py-1 px-2 rounded-lg bg-white/5 hover:bg-primary/15 border border-white/5 hover:border-primary/30 transition-colors flex items-center justify-between text-[10px] text-slate-300 font-mono"
                    >
                      <span className="flex items-center gap-1 text-slate-400">
                        <Network className="w-3 h-3 text-primary" />
                        <span>Đồ thị</span>
                      </span>
                      <span className="text-primary font-semibold">Khám phá →</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* 4. Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between pt-4 border-t border-white/10 text-xs font-mono">
              <span className="text-slate-400">
                Hiển thị {historyItems.length} / {totalItems} phim đã đánh giá
              </span>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="p-1.5 rounded-lg bg-slate-900 border border-white/10 text-slate-300 hover:text-white disabled:opacity-30 disabled:pointer-events-none transition-colors"
                  title="Trang trước"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>

                <span className="text-slate-200 font-semibold px-2">
                  Trang {currentPage} / {totalPages}
                </span>

                <button
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="p-1.5 rounded-lg bg-slate-900 border border-white/10 text-slate-300 hover:text-white disabled:opacity-30 disabled:pointer-events-none transition-colors"
                  title="Trang sau"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
