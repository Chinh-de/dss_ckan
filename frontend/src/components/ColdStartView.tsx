import React, { useState, useEffect } from "react";
import {
  Sliders,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Database,
  ArrowRight,
  ShieldCheck,
  RefreshCw,
  GitBranch,
} from "lucide-react";
import { api } from "../services/api";
import { DomainType, ColdStartSimulationResponse } from "../types";

interface ColdStartViewProps {
  domain: DomainType;
  accentColor: string;
}

export const ColdStartView: React.FC<ColdStartViewProps> = ({ domain, accentColor }) => {
  const [interactions, setInteractions] = useState<number>(3);
  const [data, setData] = useState<ColdStartSimulationResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSimulation = async (count: number) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.simulateColdStart(domain, count);
      setData(res);
    } catch (err: any) {
      console.error(err);
      setError("Không thể tải dữ liệu thử nghiệm khởi động lạnh.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSimulation(interactions);
  }, [domain, interactions]);

  const domainLabels: Record<DomainType, { name: string; sparsity: string; term: string }> = {
    movie: { name: "Điện ảnh (MovieLens)", sparsity: "99.44%", term: "phim" },
    book: { name: "Sách (Book-Crossing)", sparsity: "99.94%", term: "cuốn sách" },
    music: { name: "Âm nhạc (Last.FM)", sparsity: "99.41%", term: "nghệ sĩ" },
  };

  const domainInfo = domainLabels[domain];

  return (
    <div className="space-y-6 pb-12">
      {/* Header Banner */}
      <div className="rounded-2xl border border-white/10 bg-[#10121A] p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span
                className="px-2.5 py-0.5 rounded-md text-[11px] font-mono font-medium tracking-wide uppercase border"
                style={{
                  color: accentColor,
                  borderColor: `${accentColor}40`,
                  backgroundColor: `${accentColor}15`,
                }}
              >
                Bài toán 3: Thử nghiệm độ thưa & Khởi động lạnh
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Độ thưa dữ liệu: <strong className="text-slate-200">{domainInfo.sparsity}</strong>
              </span>
            </div>
            <h1 className="text-xl font-bold text-white tracking-tight">
              Đánh giá suy giảm hiệu năng: Collaborative Filtering vs CKAN
            </h1>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
              Mô phỏng hành vi gợi ý khi người dùng chỉ có rất ít lượt tương tác (1 - 5 {domainInfo.term}).
              Quan sát trực quan cách mạng lưới Knowledge Graph bù đắp thông tin bị thiếu hụt trên ma trận tương tác.
            </p>
          </div>

          <button
            onClick={() => fetchSimulation(interactions)}
            disabled={isLoading}
            className="self-start md:self-auto flex items-center gap-2 px-3.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-medium text-slate-200 transition-all active:scale-95 disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
            <span>Tải lại dữ liệu</span>
          </button>
        </div>

        {/* Interactive Slider Control */}
        <div className="mt-6 pt-5 border-t border-white/5">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-slate-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-slate-400" />
              <span>Số lượng tương tác lịch sử giả định (N):</span>
              <span
                className="px-2 py-0.5 rounded font-mono font-bold text-xs border"
                style={{
                  color: accentColor,
                  borderColor: `${accentColor}50`,
                  backgroundColor: `${accentColor}15`,
                }}
              >
                {interactions} {domainInfo.term}
              </span>
            </label>
            <span className="text-[11px] text-slate-400 font-mono">
              {interactions <= 3 ? "Vùng Khởi động Lạnh cực đoan (Cold-Start)" : "Vùng Dữ liệu thưa tiêu chuẩn"}
            </span>
          </div>

          <div className="space-y-2">
            <input
              type="range"
              min="1"
              max="20"
              value={interactions}
              onChange={(e) => setInteractions(parseInt(e.target.value, 10))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-primary"
              style={{ accentColor }}
            />
            <div className="flex justify-between text-[11px] text-slate-400 font-mono">
              <button
                onClick={() => setInteractions(1)}
                className={`hover:text-white transition-colors ${interactions === 1 ? "text-white font-bold" : ""}`}
              >
                1 (Cực thưa)
              </button>
              <button
                onClick={() => setInteractions(3)}
                className={`hover:text-white transition-colors ${interactions === 3 ? "text-white font-bold" : ""}`}
              >
                3 (Cold-Start)
              </button>
              <button
                onClick={() => setInteractions(5)}
                className={`hover:text-white transition-colors ${interactions === 5 ? "text-white font-bold" : ""}`}
              >
                5
              </button>
              <button
                onClick={() => setInteractions(10)}
                className={`hover:text-white transition-colors ${interactions === 10 ? "text-white font-bold" : ""}`}
              >
                10 (10% Data)
              </button>
              <button
                onClick={() => setInteractions(20)}
                className={`hover:text-white transition-colors ${interactions === 20 ? "text-white font-bold" : ""}`}
              >
                20 (Bão hòa)
              </button>
            </div>
          </div>
        </div>
      </div>

      {isLoading && !data && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 animate-pulse">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-28 rounded-2xl bg-white/5 border border-white/5" />
          ))}
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {data && (
        <>
          {/* Bento Grid: 4 Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Metric 1: ROC-AUC */}
            <div className="p-4 rounded-2xl bg-[#10121A] border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>ROC-AUC</span>
                <span className="font-mono font-bold text-emerald-400 text-[11px] bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                  +{data.currentMetrics.delta_auc_pct}%
                </span>
              </div>
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[10px] text-slate-400 block font-mono">CF Baseline</span>
                  <span className="font-mono font-bold text-lg text-slate-400">
                    {data.currentMetrics.cf_auc.toFixed(4)}
                  </span>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-600 mb-1" />
                <div className="text-right">
                  <span className="text-[10px] block font-mono" style={{ color: accentColor }}>
                    CKAN (KG)
                  </span>
                  <span className="font-mono font-bold text-xl text-white">
                    {data.currentMetrics.ckan_auc.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>

            {/* Metric 2: Recall@10 */}
            <div className="p-4 rounded-2xl bg-[#10121A] border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Recall@10</span>
                <span className="font-mono font-bold text-emerald-400 text-[11px] bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                  +{data.currentMetrics.delta_recall_pct}%
                </span>
              </div>
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[10px] text-slate-400 block font-mono">CF Baseline</span>
                  <span className="font-mono font-bold text-lg text-slate-400">
                    {data.currentMetrics.cf_recall10.toFixed(4)}
                  </span>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-600 mb-1" />
                <div className="text-right">
                  <span className="text-[10px] block font-mono" style={{ color: accentColor }}>
                    CKAN (KG)
                  </span>
                  <span className="font-mono font-bold text-xl text-white">
                    {data.currentMetrics.ckan_recall10.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>

            {/* Metric 3: F1-Score */}
            <div className="p-4 rounded-2xl bg-[#10121A] border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>F1-Score</span>
                <span className="font-mono text-slate-400 text-[11px]">Độ đo cân bằng</span>
              </div>
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[10px] text-slate-400 block font-mono">CF Baseline</span>
                  <span className="font-mono font-bold text-lg text-slate-400">
                    {data.currentMetrics.cf_f1.toFixed(4)}
                  </span>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-600 mb-1" />
                <div className="text-right">
                  <span className="text-[10px] block font-mono" style={{ color: accentColor }}>
                    CKAN (KG)
                  </span>
                  <span className="font-mono font-bold text-xl text-white">
                    {data.currentMetrics.ckan_f1.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>

            {/* Metric 4: NDCG@10 */}
            <div className="p-4 rounded-2xl bg-[#10121A] border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>NDCG@10</span>
                <span className="font-mono text-slate-400 text-[11px]">Chất lượng thứ hạng</span>
              </div>
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[10px] text-slate-400 block font-mono">CF Baseline</span>
                  <span className="font-mono font-bold text-lg text-slate-400">
                    {data.currentMetrics.cf_ndcg10.toFixed(4)}
                  </span>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-600 mb-1" />
                <div className="text-right">
                  <span className="text-[10px] block font-mono" style={{ color: accentColor }}>
                    CKAN (KG)
                  </span>
                  <span className="font-mono font-bold text-xl text-white">
                    {data.currentMetrics.ckan_ndcg10.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Technical Analysis & Explanation Card */}
          <div className="rounded-2xl border border-white/10 bg-[#10121A] p-5">
            <div className="flex items-start gap-3">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border"
                style={{
                  color: accentColor,
                  borderColor: `${accentColor}40`,
                  backgroundColor: `${accentColor}15`,
                }}
              >
                <GitBranch className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                  Nguyên lý khắc phục điểm nghẽn độ thưa (Sparsity Bottleneck)
                </h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  {data.explanation}
                </p>
              </div>
            </div>
          </div>

          {/* Side-by-Side Live Recommendations Simulation */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: CF Baseline Recommendations */}
            <div className="rounded-2xl border border-white/10 bg-[#10121A] p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
                  <h3 className="text-sm font-bold text-white">Collaborative Filtering (CF Baseline)</h3>
                </div>
                <span className="text-[11px] font-mono text-slate-400 bg-white/5 px-2 py-0.5 rounded">
                  Chỉ dùng ma trận tương tác
                </span>
              </div>

              <div className="space-y-2.5">
                {data.cfRecommendations.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-white/[0.02] border border-white/5 flex items-center justify-between gap-3 text-xs"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="font-mono text-slate-400 text-xs w-4">#{idx + 1}</span>
                      <div className="min-w-0">
                        <p className="font-medium text-slate-300 truncate">{item.title}</p>
                        <p className="text-[11px] text-slate-400 truncate">
                          {item.reasons[0] || "Gợi ý đại trà mặc định"}
                        </p>
                      </div>
                    </div>
                    <span className="font-mono text-[11px] text-slate-400 px-2 py-0.5 rounded bg-slate-800/80 shrink-0">
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 text-[11px] text-slate-400 flex items-start gap-2">
                <AlertCircle className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
                <span>
                  Hạn chế: CF dựa vào các cặp user có chung đánh giá. Với {interactions} tương tác,
                  vector đặc trưng không tìm được điểm giao, dẫn đến gợi ý sản phẩm ngẫu nhiên hoặc đại trà.
                </span>
              </div>
            </div>

            {/* Right: CKAN Recommendations */}
            <div
              className="rounded-2xl border p-5 space-y-4 bg-[#10121A]"
              style={{ borderColor: `${accentColor}30` }}
            >
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: accentColor }} />
                  <h3 className="text-sm font-bold text-white">CKAN (Collaborative Knowledge Attention)</h3>
                </div>
                <span
                  className="text-[11px] font-mono px-2 py-0.5 rounded border"
                  style={{
                    color: accentColor,
                    borderColor: `${accentColor}40`,
                    backgroundColor: `${accentColor}15`,
                  }}
                >
                  Lan truyền KG Attention
                </span>
              </div>

              <div className="space-y-2.5">
                {data.ckanRecommendations.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-white/[0.03] border border-white/5 flex items-center justify-between gap-3 text-xs"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="font-mono text-xs w-4" style={{ color: accentColor }}>
                        #{idx + 1}
                      </span>
                      <div className="min-w-0">
                        <p className="font-medium text-white truncate">{item.title}</p>
                        <p className="text-[11px] text-slate-400 truncate">
                          {item.reasons[0] || "Kết nối tri thức qua mạng KG"}
                        </p>
                      </div>
                    </div>
                    <span
                      className="font-mono text-[11px] font-bold px-2 py-0.5 rounded border shrink-0"
                      style={{
                        color: accentColor,
                        borderColor: `${accentColor}40`,
                        backgroundColor: `${accentColor}15`,
                      }}
                    >
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>

              <div className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20 text-[11px] text-emerald-300 flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  Ưu thế: Ngay cả khi chỉ có {interactions} tương tác, CKAN vẫn truy xuất các thuộc tính liên quan
                  trên đồ thị tri thức để tìm ra các sản phẩm cùng nhóm tác giả, đạo diễn hoặc phong cách.
                </span>
              </div>
            </div>
          </div>

          {/* Benchmark Trajectory Table */}
          <div className="rounded-2xl border border-white/10 bg-[#10121A] p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white">
                  Bảng thông số thực nghiệm theo số lượng tương tác (N = 1 đến 20)
                </h3>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Đo đạc thực tế trên tập kiểm thử độc lập của dataset {domainInfo.name}
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="text-[11px] text-slate-400 border-b border-white/5 font-mono uppercase">
                  <tr>
                    <th className="py-2.5 px-3">Tương tác (N)</th>
                    <th className="py-2.5 px-3">CF AUC</th>
                    <th className="py-2.5 px-3">CKAN AUC</th>
                    <th className="py-2.5 px-3 text-emerald-400">Δ AUC (%)</th>
                    <th className="py-2.5 px-3">CF Recall@10</th>
                    <th className="py-2.5 px-3">CKAN Recall@10</th>
                    <th className="py-2.5 px-3 text-emerald-400">Δ Recall (%)</th>
                    <th className="py-2.5 px-3">CKAN NDCG@10</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 font-mono">
                  {data.trajectory.map((row, idx) => {
                    const isSelected = row.interactions === interactions;
                    return (
                      <tr
                        key={idx}
                        className={`transition-colors ${
                          isSelected ? "bg-white/[0.06] font-semibold" : "hover:bg-white/[0.02]"
                        }`}
                      >
                        <td className="py-2.5 px-3 flex items-center gap-1.5">
                          {isSelected && (
                            <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: accentColor }} />
                          )}
                          <span>N = {row.interactions}</span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">{row.cf_auc.toFixed(4)}</td>
                        <td className="py-2.5 px-3 text-white">{row.ckan_auc.toFixed(4)}</td>
                        <td className="py-2.5 px-3 text-emerald-400 font-bold">+{row.delta_auc_pct}%</td>
                        <td className="py-2.5 px-3 text-slate-400">{row.cf_recall10.toFixed(4)}</td>
                        <td className="py-2.5 px-3 text-white">{row.ckan_recall10.toFixed(4)}</td>
                        <td className="py-2.5 px-3 text-emerald-400 font-bold">+{row.delta_recall_pct}%</td>
                        <td className="py-2.5 px-3 text-slate-300">{row.ckan_ndcg10.toFixed(4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
