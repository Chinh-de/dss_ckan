import React, { useEffect, useRef, useState } from "react";
import { RefreshCw, AlertCircle, Heart, Waypoints } from "lucide-react";
import { RecommendationItem, DomainType, DomainInfo } from "../types";
import { ItemCard } from "./ItemCard";
import { api } from "../services/api";
import { DOMAIN_META } from "../lib/theme";
import { cn } from "../lib/utils";
import { EmptyState, PageHeader, Poster, ScoreMeter, StatRow, posterAspect } from "./ui";

interface RecommendationsViewProps {
  domain: DomainType;
  userId: number;
  info?: DomainInfo;
  onExplain: (itemId: number) => void;
  onOpenUsers: () => void;
}

const GRID = "grid grid-cols-2 gap-x-5 gap-y-9 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5";

export const RecommendationsView: React.FC<RecommendationsViewProps> = ({
  domain,
  userId,
  info,
  onExplain,
  onOpenUsers,
}) => {
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [spotlightLiked, setSpotlightLiked] = useState(false);
  const toastTimer = useRef<number>();
  const meta = DOMAIN_META[domain];

  const fetchRecs = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getRecommendations(domain, userId, 15);
      setRecommendations(res.recommendations || []);
      setSpotlightLiked(false);
    } catch (err: any) {
      setError(err.message || "Không tải được danh sách gợi ý.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setRecommendations([]);
    fetchRecs();
  }, [domain, userId]);

  useEffect(() => () => window.clearTimeout(toastTimer.current), []);

  const showToast = (msg: string) => {
    setToast(msg);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 3500);
  };

  const handleLike = async (itemId: number) => {
    try {
      await api.submitFeedback(domain, userId, itemId, "LIKE");
      showToast("Đã lưu lượt thích. Làm mới để xem xếp hạng cập nhật.");
    } catch (err: any) {
      showToast(`Không lưu được lượt thích: ${err.message}`);
    }
  };

  const handleDislike = async (itemId: number) => {
    try {
      await api.submitFeedback(domain, userId, itemId, "DISLIKE");
      setRecommendations((prev) => prev.filter((item) => (item.id ?? item.movieId) !== itemId));
      showToast("Đã ẩn mục này khỏi danh sách gợi ý.");
    } catch (err: any) {
      showToast(`Không ẩn được mục này: ${err.message}`);
    }
  };

  const spotlight = recommendations[0];
  const gridItems = recommendations.slice(1);
  const spotlightId = spotlight ? spotlight.id ?? spotlight.movieId! : null;
  const isInitialLoad = loading && recommendations.length === 0;

  return (
    <div>
      {toast && (
        <div
          role="status"
          className="fixed bottom-6 left-1/2 z-40 flex max-w-[calc(100vw-2rem)] -translate-x-1/2 animate-fade-in items-center gap-2.5 rounded-lg border border-line-strong bg-raised px-4 py-2.5 text-sm shadow-lift"
        >
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
          {toast}
        </div>
      )}

      <PageHeader
        eyebrow={`Bài toán 1 · Xếp hạng Top-K · ${meta.dataset}`}
        title={`Gợi ý ${meta.item} cho người dùng #${userId}`}
        description="CKAN lan truyền sở thích của người dùng qua các bộ ba trong đồ thị tri thức rồi chấm điểm từng ứng viên. Mỗi gợi ý đi kèm đường dẫn tri thức giải thích vì sao nó được chọn."
        actions={
          <>
            <button onClick={onOpenUsers} className="btn btn-ghost">
              Đổi người dùng
            </button>
            <button onClick={fetchRecs} disabled={loading} className="btn btn-ghost">
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} strokeWidth={1.75} />
              Làm mới
            </button>
          </>
        }
      >
        <StatRow
          stats={[
            { label: `Số ${meta.item}`, value: info?.itemsCount },
            { label: "Người dùng", value: info?.usersCount },
            { label: "Bộ ba KG", value: info?.triplesCount },
            { label: "Loại quan hệ", value: info?.relationsCount },
          ]}
        />
      </PageHeader>

      {error && (
        <div className="notice notice-error mt-8" role="alert">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-neg" strokeWidth={1.75} />
          <div className="flex-1">
            <p>{error}</p>
            <button onClick={fetchRecs} className="mt-1 text-sm font-medium underline underline-offset-4">
              Thử lại
            </button>
          </div>
        </div>
      )}

      {isInitialLoad && (
        <div className="mt-10" aria-busy="true" aria-label="Đang tải gợi ý">
          <div className="grid gap-8 md:grid-cols-[220px_1fr]">
            <div className={cn("skeleton w-40 md:w-full", posterAspect(domain))} />
            <div className="space-y-4 pt-2">
              <div className="skeleton h-4 w-40" />
              <div className="skeleton h-10 w-3/4" />
              <div className="skeleton h-4 w-1/2" />
              <div className="skeleton h-20 w-full max-w-xl" />
            </div>
          </div>
          <div className={cn(GRID, "mt-14")}>
            {Array.from({ length: 10 }, (_, i) => (
              <div key={i}>
                <div className={cn("skeleton w-full", posterAspect(domain))} />
                <div className="skeleton mt-3 h-4 w-4/5" />
                <div className="skeleton mt-2 h-3 w-1/2" />
              </div>
            ))}
          </div>
        </div>
      )}

      {spotlight && spotlightId !== null && (
        <section
          aria-label="Gợi ý phù hợp nhất"
          className={cn(
            "relative mt-10 overflow-hidden rounded-2xl border border-line bg-surface transition-opacity duration-300",
            loading && "opacity-60"
          )}
        >
          {/* Ambient wash taken from the cover itself */}
          <div className="pointer-events-none absolute inset-0" aria-hidden="true">
            {spotlight.posterUrl?.startsWith("http") && (
              <img
                key={spotlight.posterUrl}
                src={spotlight.posterUrl.replace(/^http:\/\//, "https://")}
                alt=""
                onError={(e) => (e.currentTarget.style.display = "none")}
                className="h-full w-full scale-150 object-cover opacity-25 blur-3xl"
              />
            )}
            <div className="absolute inset-0 bg-gradient-to-r from-surface/40 via-surface/85 to-surface" />
          </div>

          <div className="relative grid gap-7 p-5 sm:p-8 md:grid-cols-[minmax(0,230px)_1fr] md:gap-10 lg:p-10">
            <Poster
              src={spotlight.posterUrl}
              title={spotlight.title}
              domain={domain}
              eager
              className={cn("w-40 animate-fade-up rounded-xl shadow-lift md:w-full", posterAspect(domain))}
            />

            <div className="flex min-w-0 animate-fade-up flex-col justify-center" style={{ animationDelay: "80ms" }}>
              <p className="flex items-baseline gap-3 text-[13px] text-ink-muted">
                <span className="num text-2xl leading-none text-accent">01</span>
                Phù hợp nhất trong {recommendations.length} gợi ý
              </p>

              <h2 className="mt-3 text-4xl font-semibold leading-[1.05] tracking-[-0.03em] sm:text-5xl">
                {spotlight.title}
              </h2>

              <p className="mt-2 text-ink-muted">
                {[spotlight.subtitle, spotlight.secondaryInfo].filter(Boolean).join(" · ")}
              </p>

              <ScoreMeter score={spotlight.score} className="mt-5 max-w-xs" />

              {spotlight.reasons?.length > 0 && (
                <div className="mt-6 max-w-xl border-l-2 border-accent/60 pl-4">
                  <p className="text-[13px] text-ink-faint">Vì sao được gợi ý</p>
                  <ul className="mt-1 space-y-1 text-ink">
                    {spotlight.reasons.slice(0, 2).map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="mt-7 flex flex-wrap items-center gap-2">
                <button onClick={() => onExplain(spotlightId)} className="btn btn-primary h-10 px-4">
                  <Waypoints className="h-4 w-4" strokeWidth={2} />
                  Xem đường dẫn tri thức
                </button>
                <button
                  onClick={() => {
                    if (!spotlightLiked) handleLike(spotlightId);
                    setSpotlightLiked(true);
                  }}
                  aria-pressed={spotlightLiked}
                  className="btn btn-quiet h-10"
                >
                  <Heart className={cn("h-4 w-4", spotlightLiked && "fill-current text-neg")} strokeWidth={1.75} />
                  {spotlightLiked ? "Đã thích" : "Thích"}
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {gridItems.length > 0 && (
        <section className={cn("mt-14 transition-opacity duration-300", loading && "opacity-60")}>
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold tracking-tight">Xếp hạng tiếp theo</h2>
            <p className="num text-[13px] text-ink-faint">
              02–{String(recommendations.length).padStart(2, "0")}
            </p>
          </div>

          <div className={cn(GRID, "mt-6")}>
            {gridItems.map((item, i) => (
              <ItemCard
                key={item.id ?? item.movieId}
                item={item}
                domain={domain}
                rank={i + 2}
                style={{ animationDelay: `${Math.min(i, 9) * 45}ms` }}
                onLike={handleLike}
                onDislike={handleDislike}
                onExplain={onExplain}
              />
            ))}
          </div>
        </section>
      )}

      {!loading && !error && recommendations.length === 0 && (
        <div className="mt-10">
          <EmptyState
            title="Chưa có gợi ý cho người dùng này"
            action={
              <button onClick={onOpenUsers} className="btn btn-primary">
                Chọn người dùng khác
              </button>
            }
          >
            Người dùng #{userId} chưa có lịch sử tương tác trong tập {meta.dataset}. Chọn một hồ sơ mẫu hoặc thích vài{" "}
            {meta.item} trong thư viện để bắt đầu.
          </EmptyState>
        </div>
      )}
    </div>
  );
};
