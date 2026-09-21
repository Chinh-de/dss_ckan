import React, { useEffect, useState } from "react";
import { Search, Film, Star, ChevronLeft, ChevronRight, X, ArrowRight, Share2 } from "lucide-react";
import { Movie } from "../types";
import { api } from "../services/api";
import { formatYear } from "../lib/utils";

interface MovieExploreViewProps {
  userId: number;
  onLike: (movieId: number) => void;
}

const GENRE_FILTERS = [
  { key: "All", label: "Tất cả" },
  { key: "Action", label: "Hành động" },
  { key: "Adventure", label: "Phiêu lưu" },
  { key: "Animation", label: "Hoạt hình" },
  { key: "Comedy", label: "Hài hước" },
  { key: "Crime", label: "Hình sự" },
  { key: "Drama", label: "Chính kịch" },
  { key: "Fantasy", label: "Giả tưởng" },
  { key: "Mystery", label: "Bí ẩn" },
  { key: "Romance", label: "Lãng mạn" },
  { key: "Sci-Fi", label: "Khoa học viễn tưởng" },
  { key: "Thriller", label: "Giật gân" }
];

export const MovieExploreView: React.FC<MovieExploreViewProps> = ({ userId, onLike }) => {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [selectedGenre, setSelectedGenre] = useState("All");
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);

  // Detail drawer
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null);
  const [relatedMovies, setRelatedMovies] = useState<any[]>([]);
  const [drawerLoading, setDrawerLoading] = useState(false);

  const fetchMovies = async () => {
    try {
      setLoading(true);
      const genreParam = selectedGenre === "All" ? undefined : selectedGenre;
      const res = await api.getMovies(page, 20, search || undefined, genreParam);
      setMovies(res.data);
      setTotalPages(res.totalPages);
      setTotal(res.total);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMovies();
  }, [page, selectedGenre]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchMovies();
  };

  const handleSelectMovie = async (movie: Movie) => {
    setSelectedMovie(movie);
    try {
      setDrawerLoading(true);
      const related = await api.getRelatedMovies(movie.id, 6);
      setRelatedMovies(related);
    } catch (err) {
      console.error(err);
    } finally {
      setDrawerLoading(false);
    }
  };

  return (
    <div className="space-y-6 pb-16">
      {/* Search & Genre Controls */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white">Khám Phá Thư Viện Điện Ảnh</h2>
            <p className="text-xs text-slate-400">
              Tra cứu hơn 16.000 tác phẩm được kết nối trên đồ thị tri thức
            </p>
          </div>

          <form onSubmit={handleSearchSubmit} className="relative w-full sm:w-72">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Tìm kiếm theo tên phim..."
              className="w-full pl-9 pr-4 py-2 rounded-xl bg-card border border-white/10 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </form>
        </div>

        {/* Genre filter chips */}
        <div className="flex flex-wrap gap-1.5 pt-1">
          {GENRE_FILTERS.map((g) => {
            const active = selectedGenre === g.key;
            return (
              <button
                key={g.key}
                onClick={() => {
                  setSelectedGenre(g.key);
                  setPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                  active
                    ? "bg-primary text-black border-primary shadow-glow-gold"
                    : "bg-slate-900/60 text-slate-400 border-white/5 hover:border-white/20 hover:text-white"
                }`}
              >
                {g.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Movies Grid */}
      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
            <div key={i} className="aspect-[2/3] rounded-2xl bg-card border border-white/5 animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {movies.map((m) => (
            <div
              key={m.id}
              onClick={() => handleSelectMovie(m)}
              className="glass-card rounded-2xl overflow-hidden cursor-pointer group flex flex-col justify-between"
            >
              <div className="aspect-[2/3] bg-slate-900 overflow-hidden relative">
                {m.posterUrl ? (
                  <img
                    src={m.posterUrl}
                    alt={m.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center bg-slate-900 p-4 text-center">
                    <Film className="w-8 h-8 text-slate-600 mb-2" />
                    <p className="text-xs font-bold text-slate-300 line-clamp-2">{m.title}</p>
                  </div>
                )}
                <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur-md text-[10px] font-mono text-slate-300 border border-white/10">
                  {formatYear(m.releaseYear)}
                </div>
              </div>

              <div className="p-3">
                <h4 className="text-xs font-bold text-white line-clamp-1 group-hover:text-primary transition-colors">
                  {m.title}
                </h4>
                <p className="text-[10px] text-slate-400 truncate mt-0.5">
                  {(m.genres || []).join(", ")}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      <div className="flex items-center justify-between pt-6 border-t border-white/5 text-xs text-slate-400">
        <p>
          Hiển thị trang <span className="text-white font-bold">{page}</span> /{" "}
          <span className="text-white font-bold">{totalPages}</span> (Tổng số {total} tác phẩm)
        </p>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="p-2 rounded-lg bg-card border border-white/10 text-white disabled:opacity-30 hover:bg-white/5 transition-all"
            title="Trang trước"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="p-2 rounded-lg bg-card border border-white/10 text-white disabled:opacity-30 hover:bg-white/5 transition-all"
            title="Trang kế tiếp"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Movie Detail Drawer (Slide-out) */}
      {selectedMovie && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="w-full max-w-md bg-card border-l border-white/10 h-full p-6 overflow-y-auto space-y-6 animate-in slide-in-from-right duration-300">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-mono tracking-wider text-primary border border-primary/30 px-2 py-0.5 rounded bg-primary/10">
                Thông Tin Tác Phẩm
              </span>
              <button
                onClick={() => setSelectedMovie(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-white"
                title="Đóng"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Poster & Title */}
            <div>
              <h2 className="text-2xl font-bold text-white">{selectedMovie.title}</h2>
              <div className="flex items-center gap-2 text-xs text-slate-400 font-mono mt-1">
                <span>{selectedMovie.releaseYear || "N/A"}</span>
                <span>•</span>
                <span>{(selectedMovie.genres || []).join(", ")}</span>
              </div>
            </div>

            {selectedMovie.overview && (
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-1">
                  Tóm Tắt Nội Dung
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed">{selectedMovie.overview}</p>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  onLike(selectedMovie.id);
                  alert(`Đã thêm "${selectedMovie.title}" vào phim yêu thích! Đồ thị tri thức đã cập nhật.`);
                }}
                className="flex-1 py-2 rounded-xl bg-primary text-black font-bold text-xs shadow-glow-gold hover:brightness-110 active:scale-95 transition-all"
              >
                Thêm Vào Phim Yêu Thích
              </button>
            </div>

            {/* Related Movies on Neo4j Graph */}
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2 mb-3">
                <Share2 className="w-4 h-4 text-primary" />
                <span>Các Tác Phẩm Liên Quan Trên Đồ Thị Tri Thức</span>
              </h4>

              {drawerLoading ? (
                <div className="py-6 text-center text-xs text-slate-500 font-mono">
                  Đang truy vấn các mối quan hệ trên Neo4j...
                </div>
              ) : relatedMovies.length > 0 ? (
                <div className="space-y-2">
                  {relatedMovies.map((rel) => (
                    <div
                      key={rel.id}
                      onClick={() => handleSelectMovie(rel)}
                      className="p-3 rounded-xl bg-slate-900/60 border border-white/5 hover:border-primary/40 cursor-pointer transition-all flex items-center justify-between"
                    >
                      <div>
                        <p className="text-xs font-semibold text-white">{rel.title}</p>
                        <p className="text-[10px] text-slate-400 font-mono">
                          Chung thực thể: {(rel.sharedEntities || []).join(", ")}
                        </p>
                      </div>
                      <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500">Không tìm thấy tác phẩm liên quan trên đồ thị.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

