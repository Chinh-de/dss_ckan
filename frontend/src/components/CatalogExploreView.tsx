import React, { useEffect, useState } from "react";
import { Search, ChevronLeft, ChevronRight, Heart, Check, X, AlertCircle } from "lucide-react";
import { ItemDto, DomainType } from "../types";
import { api } from "../services/api";
import { DOMAIN_META } from "../lib/theme";
import { cn } from "../lib/utils";
import { EmptyState, PageHeader, Poster, posterAspect } from "./ui";

interface CatalogExploreViewProps {
  domain: DomainType;
  userId: number;
}

const LIMIT = 20;
const GRID = "grid grid-cols-2 gap-x-5 gap-y-9 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5";

export const CatalogExploreView: React.FC<CatalogExploreViewProps> = ({ domain, userId }) => {
  const [items, setItems] = useState<ItemDto[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchInput, setSearchInput] = useState("");
  // `query` is the submitted search; the request is keyed on it, not on every keystroke.
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [likedIds, setLikedIds] = useState<Set<number>>(new Set());
  const [likeError, setLikeError] = useState<string | null>(null);
  const meta = DOMAIN_META[domain];

  useEffect(() => {
    setSearchInput("");
    setQuery("");
    setPage(1);
  }, [domain]);

  useEffect(() => {
    setLikedIds(new Set());
    setLikeError(null);
  }, [domain, userId]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getCatalogItems(domain, page, LIMIT, query || undefined)
      .then((res) => {
        if (cancelled) return;
        setItems(res.data);
        setTotal(res.total);
      })
      .catch((err) => {
        if (cancelled) return;
        setItems([]);
        setTotal(0);
        setError(err.message || "Không tải được thư viện.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [domain, page, query]);

  const submitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    setQuery(searchInput.trim());
  };

  const clearSearch = () => {
    setSearchInput("");
    setQuery("");
    setPage(1);
  };

  const handleLike = async (id: number) => {
    setLikeError(null);
    setLikedIds((prev) => new Set(prev).add(id));
    try {
      await api.submitFeedback(domain, userId, id, "LIKE");
    } catch (err: any) {
      setLikedIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setLikeError(`Không lưu được lượt thích: ${err.message}`);
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / LIMIT));

  return (
    <div>
      <PageHeader
        eyebrow={`Thư viện · ${meta.dataset}`}
        title="Thích vài mục để định hình hồ sơ"
        description={
          <>
            Mỗi lượt thích được thêm vào tập hạt giống của người dùng #{userId}. CKAN lan truyền lại trên đồ thị tri thức
            ngay ở lần gợi ý kế tiếp, không cần huấn luyện lại mô hình.
          </>
        }
        actions={
          <form onSubmit={submitSearch} role="search" className="relative w-full sm:w-80">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-ink-faint" strokeWidth={1.75} />
            <input
              type="search"
              aria-label={meta.searchHint}
              placeholder={meta.searchHint}
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="field px-9 [&::-webkit-search-cancel-button]:hidden"
            />
            {searchInput && (
              <button
                type="button"
                onClick={clearSearch}
                aria-label="Xoá tìm kiếm"
                className="absolute right-1.5 top-1.5 inline-flex h-6 w-6 items-center justify-center rounded-md text-ink-faint hover:text-ink"
              >
                <X className="h-4 w-4" strokeWidth={1.75} />
              </button>
            )}
          </form>
        }
      />

      <div className="mt-6 flex flex-wrap items-baseline justify-between gap-2 text-[13px] text-ink-muted">
        <p>
          {query ? (
            <>
              <span className="num text-ink">{total.toLocaleString("vi-VN")}</span> kết quả cho “{query}”
            </>
          ) : (
            <>
              <span className="num text-ink">{total.toLocaleString("vi-VN")}</span> {meta.item} trong tập dữ liệu
            </>
          )}
        </p>
        {likedIds.size > 0 && (
          <p>
            Đã thích <span className="num text-ink">{likedIds.size}</span> mục trong phiên này
          </p>
        )}
      </div>

      {(error || likeError) && (
        <div className="notice notice-error mt-4" role="alert">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-neg" strokeWidth={1.75} />
          <p>{error || likeError}</p>
        </div>
      )}

      {loading && items.length === 0 && (
        <div className={cn(GRID, "mt-6")} aria-busy="true">
          {Array.from({ length: 10 }, (_, i) => (
            <div key={i}>
              <div className={cn("skeleton w-full", posterAspect(domain))} />
              <div className="skeleton mt-3 h-4 w-4/5" />
              <div className="skeleton mt-2 h-3 w-1/2" />
            </div>
          ))}
        </div>
      )}

      {items.length > 0 && (
        <div className={cn(GRID, "mt-6 transition-opacity duration-300", loading && "opacity-50")}>
          {items.map((item, i) => {
            const isLiked = likedIds.has(item.id);
            return (
              <article
                key={item.id}
                className="group flex animate-fade-up flex-col"
                style={{ animationDelay: `${Math.min(i, 9) * 35}ms` }}
              >
                <Poster
                  src={item.posterUrl}
                  title={item.title}
                  domain={domain}
                  className={cn("w-full rounded-lg", posterAspect(domain))}
                  imgClassName="transition-transform duration-500 ease-out group-hover:scale-[1.04]"
                />
                <div className="flex flex-1 flex-col pt-3">
                  <h3 className="line-clamp-2 font-sans text-[15px] font-semibold leading-snug">{item.title}</h3>
                  <p className="mt-0.5 truncate text-[13px] text-ink-muted">
                    {[item.subtitle, item.category || item.releaseYear].filter(Boolean).join(" · ")}
                  </p>
                  <div className="mt-auto flex items-center justify-between pt-3">
                    <span className="num text-xs text-ink-faint">#{item.id}</span>
                    <button
                      onClick={() => handleLike(item.id)}
                      disabled={isLiked}
                      className={cn(
                        "btn h-8 px-2.5 text-[13px] disabled:opacity-100",
                        isLiked ? "bg-pos/15 text-pos" : "btn-ghost"
                      )}
                    >
                      {isLiked ? (
                        <Check className="h-3.5 w-3.5" strokeWidth={2} />
                      ) : (
                        <Heart className="h-3.5 w-3.5" strokeWidth={1.75} />
                      )}
                      {isLiked ? "Đã thích" : "Thích"}
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {!loading && !error && items.length === 0 && (
        <div className="mt-6">
          <EmptyState
            title={query ? `Không tìm thấy “${query}”` : "Thư viện đang trống"}
            action={
              query ? (
                <button onClick={clearSearch} className="btn btn-ghost">
                  Xoá tìm kiếm
                </button>
              ) : undefined
            }
          >
            {query
              ? "Thử từ khoá ngắn hơn hoặc tên gốc tiếng Anh; dữ liệu giữ nguyên tiêu đề của tập gốc."
              : `Backend chưa trả về mục nào cho tập ${meta.dataset}.`}
          </EmptyState>
        </div>
      )}

      {totalPages > 1 && (
        <nav aria-label="Phân trang" className="mt-12 flex items-center justify-between border-t border-line pt-5">
          <p className="num text-[13px] text-ink-muted">
            Trang {page} / {totalPages.toLocaleString("vi-VN")}
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1 || loading}
              className="btn btn-ghost"
            >
              <ChevronLeft className="h-4 w-4" strokeWidth={1.75} />
              Trước
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || loading}
              className="btn btn-ghost"
            >
              Sau
              <ChevronRight className="h-4 w-4" strokeWidth={1.75} />
            </button>
          </div>
        </nav>
      )}
    </div>
  );
};
