import React, { useEffect, useState } from "react";
import { RefreshCw, AlertCircle, HelpCircle, Heart, Database, GitBranch, Layers } from "lucide-react";
import { RecommendationItem, DomainType, DomainInfo } from "../types";
import { ItemCard } from "./ItemCard";
import { api } from "../services/api";

interface RecommendationsViewProps {
  domain: DomainType;
  accentColor: string;
  userId: number;
  currentDomainInfo?: DomainInfo;
  onExplain: (itemId: number) => void;
  openUserSelect: () => void;
}

export const RecommendationsView: React.FC<RecommendationsViewProps> = ({
  domain,
  accentColor,
  userId,
  currentDomainInfo,
  onExplain,
  openUserSelect,
}) => {
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const fetchRecs = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getRecommendations(domain, userId, 15);
      setRecommendations(res.recommendations || []);
    } catch (err: any) {
      setError(err.message || "Không thể tải danh sách đề xuất.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecs();
  }, [domain, userId]);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleLike = async (itemId: number) => {
    try {
      await api.submitFeedback(domain, userId, itemId, "LIKE");
      showToast("Đã lưu lượt thích. Mô hình sẽ cập nhật ưu tiên ở lần tính toán tiếp theo.");
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleDislike = async (itemId: number) => {
    try {
      await api.submitFeedback(domain, userId, itemId, "DISLIKE");
      showToast("Đã ẩn mục này khỏi danh sách đề xuất.");
      setRecommendations((prev) => prev.filter((item) => (item.id ?? item.movieId) !== itemId));
    } catch (err: any) {
      console.error(err);
    }
  };

  const spotlight = recommendations.length > 0 ? recommendations[0] : null;
  const gridItems = recommendations.length > 0 ? recommendations.slice(1) : [];

  return (
    <div className="space-y-6 pb-16">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 px-4 py-2.5 rounded-xl bg-[#10121A] border border-white/20 text-xs font-mono font-medium text-white shadow-2xl animate-in slide-in-from-bottom-2 duration-200 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: accentColor }} />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Domain Stats & Overview Banner */}
      <div className="rounded-2xl border border-white/10 bg-[#10121A] p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span
                className="px-2.5 py-0.5 rounded-md text-[11px] font-mono font-bold uppercase tracking-wider border"
                style={{
                  color: accentColor,
                  borderColor: `${accentColor}40`,
                  backgroundColor: `${accentColor}15`,
                }}
              >
                Bài toán 1: Xếp hạng Top-K
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Người dùng: <strong className="text-white">#{userId}</strong>
              </span>
            </div>
            <h1 className="text-xl font-bold text-white tracking-tight">
              Gợi ý cá nhân hóa đa miền: {currentDomainInfo?.vietnameseName || domain}
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Xếp hạng bởi mạng Collaborative Knowledge-aware Attention Network (CKAN) kết hợp Knowledge Graph.
            </p>
          </div>

          {/* Quick Domain Stats Badges */}
          <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
            <div className="px-3 py-1.5 rounded-xl bg-white/[0.03] border border-white/5 flex items-center gap-2">
              <Database className="w-3.5 h-3.5 text-slate-400" />
              <span>
                <strong className="text-white">{currentDomainInfo?.itemsCount.toLocaleString() || "..."}</strong>{" "}
                <span className="text-slate-500">mục</span>
              </span>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-white/[0.03] border border-white/5 flex items-center gap-2">
              <GitBranch className="w-3.5 h-3.5 text-slate-400" />
              <span>
                <strong className="text-white">{currentDomainInfo?.triplesCount.toLocaleString() || "..."}</strong>{" "}
                <span className="text-slate-500">triples KG</span>
              </span>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-white/[0.03] border border-white/5 flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              <span>
                <strong className="text-white">{currentDomainInfo?.relationsCount || "..."}</strong>{" "}
                <span className="text-slate-500">quan hệ</span>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Hero Spotlight (Top 1 Pick) */}
      {spotlight && !loading && (
        <div className="rounded-3xl border border-white/10 bg-[#10121A] p-6 sm:p-8 relative overflow-hidden shadow-sm">
          <div className="relative z-10 max-w-2xl space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md text-xs font-mono font-bold border"
              style={{
                color: accentColor,
                borderColor: `${accentColor}40`,
                backgroundColor: `${accentColor}15`,
              }}
            >
              <span>VỊ TRÍ #1 • ĐIỂM TƯƠNG ĐỒNG {(spotlight.score * 100).toFixed(1)}%</span>
            </div>

            <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white leading-tight">
              {spotlight.title}
            </h2>

            {spotlight.subtitle && (
              <p className="text-xs text-slate-300 font-mono">
                {spotlight.subtitle} {spotlight.secondaryInfo ? `• ${spotlight.secondaryInfo}` : ""}
              </p>
            )}

            {spotlight.reasons && spotlight.reasons[0] && (
              <p className="text-xs text-slate-300 border-l-2 pl-3 leading-relaxed" style={{ borderColor: accentColor }}>
                {spotlight.reasons[0]}
              </p>
            )}

            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={() => onExplain(spotlight.id ?? spotlight.movieId!)}
                className="px-4 py-2 rounded-xl text-black font-bold text-xs hover:brightness-110 active:scale-95 transition-all flex items-center gap-1.5"
                style={{ backgroundColor: accentColor }}
              >
                <HelpCircle className="w-3.5 h-3.5" />
                <span>Xem đường dẫn tri thức</span>
              </button>
              <button
                onClick={() => handleLike(spotlight.id ?? spotlight.movieId!)}
                className="px-3.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white font-medium text-xs active:scale-95 transition-all flex items-center gap-1.5"
              >
                <Heart className="w-3.5 h-3.5 text-rose-400" />
                <span>Thích</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Grid Header & Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold tracking-tight text-white">
            Danh mục đề xuất tiếp theo
          </h2>
          <p className="text-[11px] text-slate-400 font-mono">
            Hiển thị {recommendations.length} kết quả xếp hạng cao nhất
          </p>
        </div>

        <button
          onClick={fetchRecs}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-medium text-slate-300 hover:text-white transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Làm mới danh sách</span>
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading Skeletons */}
      {loading && recommendations.length === 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 animate-pulse">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
            <div key={i} className="rounded-2xl bg-white/5 border border-white/5 aspect-[3/4] p-4 flex flex-col justify-end">
              <div className="h-3 bg-white/10 rounded w-3/4 mb-1.5" />
              <div className="h-2.5 bg-white/10 rounded w-1/2" />
            </div>
          ))}
        </div>
      )}

      {/* Grid of Recommended Items */}
      {!loading && gridItems.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {gridItems.map((item) => (
            <ItemCard
              key={item.id ?? item.movieId}
              item={item}
              domain={domain}
              accentColor={accentColor}
              userId={userId}
              onLike={handleLike}
              onDislike={handleDislike}
              onExplain={onExplain}
            />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && recommendations.length === 0 && (
        <div className="p-12 text-center border border-dashed border-white/10 rounded-2xl bg-[#10121A]/50">
          <Database className="w-10 h-10 text-slate-600 mx-auto mb-2" />
          <h3 className="text-sm font-bold text-white">Chưa có kết quả đề xuất</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1">
            Không tìm thấy mục phù hợp cho Người dùng #{userId}. Hãy chọn người dùng khác trong danh sách demo.
          </p>
          <button
            onClick={openUserSelect}
            className="mt-4 px-4 py-2 rounded-xl text-black text-xs font-bold"
            style={{ backgroundColor: accentColor }}
          >
            Chọn người dùng demo
          </button>
        </div>
      )}
    </div>
  );
};
