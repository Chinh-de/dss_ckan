import React, { useState, useEffect } from "react";
import { AlertCircle } from "lucide-react";
import { api } from "../services/api";
import { DomainType, ColdStartSimulationResponse, ModelResult, RecommendationItem } from "../types";
import { BASELINE_HEX, DOMAIN_ACCENT, DOMAIN_META, RIPPLENET_HEX } from "../lib/theme";
import { cn } from "../lib/utils";
import { PageHeader } from "./ui";
import { ChartSeries, TrajectoryChart } from "./TrajectoryChart";

interface ColdStartViewProps {
  domain: DomainType;
}

const SEED_STEPS = [1, 2, 3, 5, 10, 20];
const MODEL_ORDER = ["MostPopular", "MF", "RippleNet", "CKAN"];

const pct = (ratio: number) => `${Math.round(ratio * 100)}%`;
const signedPct = (from: number, to: number) => {
  const delta = ((to - from) / from) * 100;
  return `${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(1)}%`;
};

const RecList: React.FC<{ items: RecommendationItem[]; emphasised?: boolean; scoreTitle: string }> = ({
  items,
  emphasised,
  scoreTitle,
}) => (
  <ol className="mt-4 divide-y divide-line">
    {items.map((item, idx) => (
      <li key={`${item.id}-${idx}`} className="flex items-baseline gap-3 py-2.5">
        <span className={cn("num w-5 shrink-0 text-xs", emphasised ? "text-accent" : "text-ink-faint")}>
          {String(idx + 1).padStart(2, "0")}
        </span>
        <div className="min-w-0 flex-1">
          <p className={cn("truncate", emphasised ? "font-medium text-ink" : "text-ink-muted")}>{item.title}</p>
          {item.reasons[0] && <p className="truncate text-[13px] text-ink-faint">{item.reasons[0]}</p>}
        </div>
        <span className="num shrink-0 text-xs text-ink-muted" title={scoreTitle}>
          {(item.score * 100).toFixed(1)}%
        </span>
      </li>
    ))}
  </ol>
);

export const ColdStartView: React.FC<ColdStartViewProps> = ({ domain }) => {
  const [interactions, setInteractions] = useState(3);
  const [ratioIndex, setRatioIndex] = useState(0);
  const [data, setData] = useState<ColdStartSimulationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const meta = DOMAIN_META[domain];

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    api
      .simulateColdStart(domain, interactions)
      .then((res) => !cancelled && setData(res))
      .catch((err) => !cancelled && setError(err.message || "Không tải được dữ liệu thử nghiệm."))
      .finally(() => !cancelled && setIsLoading(false));
    return () => {
      cancelled = true;
    };
  }, [domain, interactions]);

  const sparsity = data?.sparsity ?? [];
  const point = sparsity[Math.min(ratioIndex, sparsity.length - 1)];
  const first = sparsity[0];
  const last = sparsity[sparsity.length - 1];
  const models = MODEL_ORDER.map((name) => data?.models.find((m) => m.model === name)).filter(
    (m): m is ModelResult => !!m
  );
  const best = (key: "auc" | "f1" | "acc") => Math.max(...models.map((m) => m[key]));
  const bestRecall = (k: string) => Math.max(...models.map((m) => m.recall[k] ?? 0));

  // One sentence on how CKAN and MF compare as data grows, derived from the measured points.
  const crossover = sparsity.findIndex((_, i) => sparsity.slice(i).every((p) => p.ckan_auc >= p.mf_auc));
  let trend = "";
  if (first && last) {
    const gapFirst = first.ckan_auc - first.mf_auc;
    const gapLast = last.ckan_auc - last.mf_auc;
    if (crossover === 0) {
      trend =
        gapLast < gapFirst
          ? "CKAN dẫn trước ở mọi mức; lợi thế lớn nhất lúc thiếu tương tác và thu hẹp dần khi dữ liệu dày lên."
          : "CKAN dẫn trước ở mọi mức, và khoảng cách rộng ra khi có thêm dữ liệu.";
    } else if (crossover > 0) {
      trend = `Lúc thiếu dữ liệu MF tốt hơn; CKAN chỉ vượt lên từ mức ${pct(sparsity[crossover].ratio)}.`;
    } else {
      trend = "Khi dữ liệu dày, MF đuổi kịp CKAN; lợi thế của CKAN nằm ở các mức dữ liệu thấp.";
    }
  }

  const series: ChartSeries[] = [
    { key: "ckan", label: "CKAN", legend: "CKAN", color: DOMAIN_ACCENT[domain], values: sparsity.map((p) => p.ckan_auc) },
    {
      key: "ripple",
      label: "RippleNet",
      legend: "RippleNet",
      color: RIPPLENET_HEX[domain],
      dash: "2 4",
      values: sparsity.map((p) => p.ripplenet_auc),
    },
    {
      key: "mf",
      label: "MF",
      legend: "MF (không dùng đồ thị tri thức)",
      color: BASELINE_HEX,
      dash: "6 4",
      values: sparsity.map((p) => p.mf_auc),
    },
  ];

  return (
    <div>
      <PageHeader
        eyebrow={`Bài toán 3 · Độ thưa dữ liệu · ${meta.dataset}`}
        title="Mô hình chịu thiếu dữ liệu tốt đến đâu"
        description={
          <>
            Ma trận tương tác của tập {meta.dataset} thưa tới <span className="num text-ink">{meta.sparsity}</span>. Thí
            nghiệm giữ lại 10% đến 100% tập huấn luyện rồi đo ROC-AUC trên cùng một nhóm người dùng, cho MF và hai mô hình
            dùng đồ thị tri thức là RippleNet và CKAN.
          </>
        }
      />

      {error && (
        <div className="notice notice-error mt-8" role="alert">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-neg" strokeWidth={1.75} />
          <p>{error}</p>
        </div>
      )}

      {isLoading && !data && (
        <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3" aria-busy="true">
          {[0, 1, 2].map((i) => (
            <div key={i} className="skeleton h-24" />
          ))}
        </div>
      )}

      {data && sparsity.length === 0 && (
        <div className="notice mt-8">
          <p>
            Backend chưa có số liệu benchmark cho tập này. Chạy{" "}
            <span className="num">python -m app.scripts.extract_benchmark_results</span> sau khi chạy notebook.
          </p>
        </div>
      )}

      {data && point && first && last && (
        <>
          <div className="mt-7 flex flex-wrap items-center gap-x-4 gap-y-2">
            <span id="ratio-label" className="text-sm text-ink-muted">
              Phần tập huấn luyện được giữ lại
            </span>
            <div className="seg" role="group" aria-labelledby="ratio-label">
              {sparsity.map((p, i) => (
                <button
                  key={p.ratio}
                  onClick={() => setRatioIndex(i)}
                  aria-pressed={ratioIndex === i}
                  className="seg-item num min-w-12 justify-center"
                >
                  {pct(p.ratio)}
                </button>
              ))}
            </div>
          </div>

          {/* Headline: test AUC at the selected ratio */}
          <dl className="mt-6 grid grid-cols-1 border-y border-line sm:grid-cols-3">
            {[
              { name: "CKAN", value: point.ckan_auc, main: true },
              { name: "RippleNet", value: point.ripplenet_auc, main: false },
              { name: "MF", value: point.mf_auc, main: false },
            ].map((m, i) => (
              <div key={m.name} className={cn("py-6 pr-4", i > 0 && "border-t border-line sm:border-l sm:border-t-0 sm:pl-6")}>
                <dt className="text-[13px] text-ink-muted">
                  ROC-AUC của {m.name} với {pct(point.ratio)} dữ liệu
                </dt>
                <dd className="mt-1 flex flex-wrap items-baseline gap-x-2.5">
                  <span className={cn("num text-3xl tracking-tight", m.main ? "text-ink" : "text-ink-muted")}>
                    {m.value.toFixed(4)}
                  </span>
                  {m.name !== "MF" && (
                    <span className={cn("num text-sm", m.value >= point.mf_auc ? "text-pos" : "text-neg")}>
                      {signedPct(point.mf_auc, m.value)} so với MF
                    </span>
                  )}
                </dd>
              </div>
            ))}
          </dl>

          <div className="mt-10 grid gap-10 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
            <section aria-labelledby="chart-title">
              <h2 id="chart-title" className="mb-5 text-xl font-semibold tracking-tight">
                ROC-AUC theo lượng dữ liệu huấn luyện
              </h2>
              <TrajectoryChart
                series={series}
                xLabels={sparsity.map((p) => pct(p.ratio))}
                xTitle="Tập huấn luyện được giữ lại"
                selectedIndex={ratioIndex}
                onSelect={setRatioIndex}
                ariaLabel="ROC-AUC của CKAN, RippleNet và MF theo phần tập huấn luyện được giữ lại. Số liệu đầy đủ ở bảng bên dưới."
              />
            </section>

            <section aria-labelledby="read-title" className="lg:border-l lg:border-line lg:pl-10">
              <h2 id="read-title" className="text-xl font-semibold tracking-tight">
                Đọc kết quả
              </h2>
              <ul className="mt-3 space-y-3 text-ink-muted">
                <li>
                  Chỉ với {pct(first.ratio)} dữ liệu, CKAN đạt <span className="num text-ink">{first.ckan_auc.toFixed(4)}</span>,
                  MF đạt <span className="num text-ink">{first.mf_auc.toFixed(4)}</span>: chênh{" "}
                  <span className="num text-ink">{signedPct(first.mf_auc, first.ckan_auc)}</span>.
                </li>
                <li>
                  Với đủ {pct(last.ratio)} dữ liệu, chênh lệch là{" "}
                  <span className="num text-ink">{signedPct(last.mf_auc, last.ckan_auc)}</span> (CKAN{" "}
                  <span className="num">{last.ckan_auc.toFixed(4)}</span>, MF <span className="num">{last.mf_auc.toFixed(4)}</span>
                  ). {trend}
                </li>
                <li>
                  So với RippleNet ở mức {pct(first.ratio)}: CKAN{" "}
                  {first.ckan_auc >= first.ripplenet_auc ? "cao hơn" : "thấp hơn"} (
                  <span className="num">{first.ckan_auc.toFixed(4)}</span> và{" "}
                  <span className="num">{first.ripplenet_auc.toFixed(4)}</span>).
                </li>
              </ul>
              <p className="mt-4 text-[13px] text-ink-faint">
                Ở mỗi mức, lịch sử người dùng và các tập bộ ba được dựng lại chỉ từ phần dữ liệu được giữ.
                {data.sparsityEvalUsers != null && (
                  <>
                    {" "}AUC đo trên <span className="num">{data.sparsityEvalUsers.toLocaleString("vi-VN")}</span> người
                    dùng đã có lượt thích ở mức {pct(first.ratio)}
                    {data.sparsityEvalRows != null && (
                      <>
                        {" "}(<span className="num">{data.sparsityEvalRows.toLocaleString("vi-VN")}</span> mẫu kiểm thử)
                      </>
                    )}
                    .
                  </>
                )}
                {data.source && (
                  <>
                    {" "}Số đo lấy nguyên từ notebook <span className="num">{data.source}</span>.
                  </>
                )}
              </p>
            </section>
          </div>

          <section className="mt-14 border-t border-line pt-10" aria-labelledby="sparsity-table">
            <h2 id="sparsity-table" className="text-xl font-semibold tracking-tight">
              ROC-AUC tại từng mức dữ liệu
            </h2>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[560px] text-sm">
                <thead>
                  <tr className="border-b border-line text-[13px] text-ink-faint">
                    <th scope="col" className="py-2.5 pl-3 pr-4 text-left font-normal">Dữ liệu huấn luyện</th>
                    <th scope="col" className="px-4 py-2.5 text-right font-normal">MF</th>
                    <th scope="col" className="px-4 py-2.5 text-right font-normal">RippleNet</th>
                    <th scope="col" className="px-4 py-2.5 text-right font-normal">CKAN</th>
                    <th scope="col" className="py-2.5 pl-4 pr-3 text-right font-normal">CKAN so với MF</th>
                  </tr>
                </thead>
                <tbody className="num">
                  {sparsity.map((row, i) => (
                    <tr
                      key={row.ratio}
                      onClick={() => setRatioIndex(i)}
                      aria-selected={i === ratioIndex}
                      className={cn(
                        "cursor-pointer border-b border-line transition-colors duration-200",
                        i === ratioIndex ? "bg-accent/10 text-ink" : "text-ink-muted hover:bg-surface"
                      )}
                    >
                      <th scope="row" className="py-2.5 pl-3 pr-4 text-left font-normal text-ink">{pct(row.ratio)}</th>
                      <td className="px-4 py-2.5 text-right">{row.mf_auc.toFixed(4)}</td>
                      <td className="px-4 py-2.5 text-right">{row.ripplenet_auc.toFixed(4)}</td>
                      <td className="px-4 py-2.5 text-right text-ink">{row.ckan_auc.toFixed(4)}</td>
                      <td className={cn("py-2.5 pl-4 pr-3 text-right", row.ckan_auc >= row.mf_auc ? "text-pos" : "text-neg")}>
                        {signedPct(row.mf_auc, row.ckan_auc)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {models.length > 0 && (
            <section className="mt-14 border-t border-line pt-10" aria-labelledby="full-table">
              <h2 id="full-table" className="text-xl font-semibold tracking-tight">
                Kết quả với toàn bộ dữ liệu
              </h2>
              <p className="mt-1 max-w-[64ch] text-[13px] text-ink-muted">
                Bốn mô hình trên cùng tập kiểm thử. Giá trị tốt nhất mỗi cột được tô sáng; MostPopular không cá nhân hoá
                nên F1 thấp dù AUC có thể cao.
              </p>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[640px] text-sm">
                  <thead>
                    <tr className="border-b border-line text-[13px] text-ink-faint">
                      <th scope="col" className="py-2.5 pl-3 pr-4 text-left font-normal">Mô hình</th>
                      <th scope="col" className="px-4 py-2.5 text-right font-normal">ROC-AUC</th>
                      <th scope="col" className="px-4 py-2.5 text-right font-normal">F1</th>
                      <th scope="col" className="px-4 py-2.5 text-right font-normal">Accuracy</th>
                      <th scope="col" className="px-4 py-2.5 text-right font-normal">Recall@10</th>
                      <th scope="col" className="py-2.5 pl-4 pr-3 text-right font-normal">Recall@50</th>
                    </tr>
                  </thead>
                  <tbody className="num">
                    {models.map((m) => {
                      const cell = (value: number, top: number) => (
                        <td className={cn("px-4 py-2.5 text-right last:pr-3", value === top ? "font-medium text-ink" : "text-ink-muted")}>
                          {value.toFixed(4)}
                        </td>
                      );
                      return (
                        <tr key={m.model} className="border-b border-line">
                          <th scope="row" className="py-2.5 pl-3 pr-4 text-left font-sans font-normal text-ink">{m.model}</th>
                          {cell(m.auc, best("auc"))}
                          {cell(m.f1, best("f1"))}
                          {cell(m.acc, best("acc"))}
                          {cell(m.recall["10"] ?? 0, bestRecall("10"))}
                          {cell(m.recall["50"] ?? 0, bestRecall("50"))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <section className="mt-14 border-t border-line pt-10" aria-labelledby="live-title">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h2 id="live-title" className="text-xl font-semibold tracking-tight">
                  Thử trực tiếp: người dùng chỉ có vài lượt thích
                </h2>
                <p className="mt-1 max-w-[64ch] text-[13px] text-ink-muted">
                  Phần này mô hình chạy ngay lúc bạn bấm. Chỉ giữ N lượt thích đầu của người dùng mẫu #
                  {meta.defaultUser}, rồi so danh sách phổ biến nhất với gợi ý của CKAN.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span id="n-label" className="text-sm text-ink-muted">
                  Số lượt thích (N)
                </span>
                <div className="seg" role="group" aria-labelledby="n-label">
                  {SEED_STEPS.map((n) => (
                    <button
                      key={n}
                      onClick={() => setInteractions(n)}
                      aria-pressed={interactions === n}
                      className="seg-item num min-w-10 justify-center"
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {data.seedItems.length > 0 && (
              <p className="mt-4 text-[13px] text-ink-muted">
                <span className="text-ink-faint">Đã thích: </span>
                {data.seedItems.slice(0, 6).join(" · ")}
                {data.seedItems.length > 6 && ` · và ${data.seedItems.length - 6} mục khác`}
              </p>
            )}

            <div className={cn("mt-6 grid gap-x-10 gap-y-10 transition-opacity duration-300 lg:grid-cols-2", isLoading && "opacity-60")}>
              <div className="min-w-0">
                <h3 className="text-lg font-semibold tracking-tight">Phổ biến nhất</h3>
                <p className="mt-1 text-[13px] text-ink-muted">
                  Không cá nhân hoá: ai cũng nhận danh sách này, bất kể đã thích gì. Cột phải là tỷ lệ người dùng đã
                  thích.
                </p>
                <RecList items={data.popularRecommendations} scoreTitle="Tỷ lệ người dùng trong tập dữ liệu đã thích" />
              </div>
              <div className="min-w-0">
                <h3 className="text-lg font-semibold tracking-tight">CKAN</h3>
                <p className="mt-1 text-[13px] text-ink-muted">
                  Từ {data.seedItems.length} {meta.item} trên, CKAN đi theo các quan hệ trong đồ thị tri thức. Cột phải
                  là điểm dự đoán của mô hình.
                </p>
                <RecList items={data.ckanRecommendations} emphasised scoreTitle="Điểm dự đoán của CKAN" />
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
};
