import React, { useEffect, useMemo, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { AlertCircle, ArrowRight, Maximize2, Minimize2, ScanSearch, X } from "lucide-react";
import { ExplanationPath, ExplanationResponse, DomainType, SubgraphData } from "../types";
import { api } from "../services/api";
import { DOMAIN_ACCENT, DOMAIN_META, NODE_STYLE } from "../lib/theme";
import { cn } from "../lib/utils";
import { GraphCanvas, GraphCanvasHandle, GraphLegend, kindsIn } from "./GraphCanvas";
import { Poster } from "./ui";

interface ExplainModalProps {
  itemId: number | null;
  domain: DomainType;
  userId: number;
  onClose: () => void;
  onOpenFullGraph?: () => void;
}

const PATH_PREVIEW = 4;

// "film.film.set_designer" -> "set designer"
const relationLabel = (relation: string) => (relation.split(".").pop() || relation).replace(/_/g, " ").toLowerCase();

/** Older responses carry only paths; rebuild a small subgraph from them so the canvas still has data. */
function subgraphFromPaths(exp: ExplanationResponse): SubgraphData {
  const userKey = `user-${exp.userId}`;
  const targetKey = `target-${exp.movieId}`;
  const nodes: SubgraphData["nodes"] = [
    { id: userKey, label: `Người dùng #${exp.userId}`, type: "User", data: {} },
    { id: targetKey, label: exp.paths[0]?.targetMovieTitle || `#${exp.movieId}`, type: "ItemRecommended", data: {} },
  ];
  const edges: SubgraphData["edges"] = [];
  const seen = new Set(nodes.map((n) => n.id));

  exp.paths.forEach((p, i) => {
    const srcKey = `liked-${p.sourceMovieTitle}`;
    const entKey = `ent-${p.entityName}`;
    if (!seen.has(srcKey)) {
      seen.add(srcKey);
      nodes.push({ id: srcKey, label: p.sourceMovieTitle, type: "ItemLiked", data: {} });
      edges.push({ id: `e-u-${i}`, source: userKey, target: srcKey, type: "LIKED", label: "đã thích" });
    }
    if (!seen.has(entKey)) {
      seen.add(entKey);
      nodes.push({ id: entKey, label: p.entityName, type: "Entity", data: {} });
    }
    const label = relationLabel(p.relation);
    edges.push({ id: `e-s-${i}`, source: srcKey, target: entKey, type: p.relation, label });
    edges.push({ id: `e-t-${i}`, source: targetKey, target: entKey, type: p.relation, label });
  });

  return { nodes, edges };
}

const Dot: React.FC<{ color: string }> = ({ color }) => (
  <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />
);

const PathRow: React.FC<{ path: ExplanationPath }> = ({ path }) => (
  <li className="py-3">
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
      <li className="flex items-center gap-1.5 font-medium">
        <Dot color={NODE_STYLE.liked.color} />
        {path.sourceMovieTitle}
      </li>
      <li className="flex items-center gap-2 text-ink-faint" title={path.relation}>
        <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.75} />
        <span className="text-[13px]">{path.relationLabel || relationLabel(path.relation)}</span>
        <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.75} />
      </li>
      <li className="flex items-center gap-1.5 text-ink-muted">
        <Dot color={NODE_STYLE.entity.color} />
        {path.entityName}
      </li>
      <li className="flex items-center gap-2">
        <ArrowRight className="h-3.5 w-3.5 text-ink-faint" strokeWidth={1.75} />
        <span className="flex items-center gap-1.5 font-medium">
          <Dot color={NODE_STYLE.recommended.color} />
          {path.targetMovieTitle}
        </span>
      </li>
    </ol>
    <p className="mt-1 text-[13px] text-ink-faint">{path.naturalLanguage}</p>
  </li>
);

export const ExplainModal: React.FC<ExplainModalProps> = ({ itemId, domain, userId, onClose, onOpenFullGraph }) => {
  const [data, setData] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isWide, setIsWide] = useState(false);
  const [showAllPaths, setShowAllPaths] = useState(false);
  const canvasRef = useRef<GraphCanvasHandle>(null);
  const meta = DOMAIN_META[domain];

  useEffect(() => {
    setData(null);
    setError(null);
    setIsWide(false);
    setShowAllPaths(false);
    if (itemId === null) return;

    let cancelled = false;
    setLoading(true);
    api
      .explainItem(itemId, domain, userId)
      .then((res) => !cancelled && setData(res))
      .catch((err) => !cancelled && setError(err.message || "Không tải được lý giải."))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [itemId, domain, userId]);

  const subgraph = useMemo(() => {
    if (!data) return null;
    return data.subgraph?.nodes?.length ? data.subgraph : subgraphFromPaths(data);
  }, [data]);

  const targetNode = data?.subgraph?.nodes?.find((n) => n.type.endsWith("Recommended"));
  const title = targetNode?.label || data?.paths[0]?.targetMovieTitle || (itemId !== null ? `#${itemId}` : "");
  const paths = data?.paths ?? [];
  const visiblePaths = showAllPaths ? paths : paths.slice(0, PATH_PREVIEW);
  const importance = Object.entries(data?.featureImportance || {}).sort((a, b) => b[1] - a[1]);

  return (
    <Dialog.Root open={itemId !== null} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 animate-fade-in bg-bg/70 backdrop-blur-sm" />
        <Dialog.Content
          aria-describedby={undefined}
          tabIndex={-1}
          onOpenAutoFocus={(e) => {
            // Focus the panel itself rather than its first toolbar button.
            e.preventDefault();
            (e.currentTarget as HTMLElement).focus();
          }}
          onEscapeKeyDown={(e) => {
            // First Escape collapses the widened panel, the second one closes it.
            if (isWide) {
              e.preventDefault();
              setIsWide(false);
            }
          }}
          className={cn(
            "fixed inset-y-0 right-0 z-50 flex w-full animate-sheet-in flex-col border-l border-line-strong bg-surface shadow-sheet transition-[max-width] duration-300 focus:outline-none",
            isWide ? "max-w-full" : "max-w-2xl"
          )}
        >
          <header className="flex items-start gap-4 border-b border-line px-5 py-4 sm:px-7">
            {targetNode?.data?.posterUrl && (
              <Poster
                src={targetNode.data.posterUrl}
                title={title}
                domain={domain}
                className="hidden h-[72px] w-12 shrink-0 rounded-md sm:block"
              />
            )}
            <div className="min-w-0 flex-1">
              <p className="eyebrow">Vì sao gợi ý {meta.item} này cho người dùng #{userId}</p>
              <Dialog.Title className="mt-1 truncate font-display text-2xl font-semibold tracking-tight">
                {loading ? "Đang tải…" : title}
              </Dialog.Title>
              {targetNode?.data?.releaseYear && <p className="num text-[13px] text-ink-muted">{targetNode.data.releaseYear}</p>}
            </div>
            <button
              onClick={() => setIsWide((v) => !v)}
              className="icon-btn hidden sm:inline-flex"
              title={isWide ? "Thu hẹp (Esc)" : "Mở rộng"}
              aria-label={isWide ? "Thu hẹp bảng lý giải" : "Mở rộng bảng lý giải"}
            >
              {isWide ? <Minimize2 className="h-4 w-4" strokeWidth={1.75} /> : <Maximize2 className="h-4 w-4" strokeWidth={1.75} />}
            </button>
            <Dialog.Close className="icon-btn" aria-label="Đóng">
              <X className="h-5 w-5" strokeWidth={1.75} />
            </Dialog.Close>
          </header>

          <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-7">
            {loading && (
              <div className="space-y-4" aria-busy="true">
                <div className="skeleton h-16" />
                <div className="skeleton h-80" />
                <div className="skeleton h-12" />
                <div className="skeleton h-12" />
              </div>
            )}

            {error && (
              <div className="notice notice-error" role="alert">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-neg" strokeWidth={1.75} />
                <p>{error}</p>
              </div>
            )}

            {data && subgraph && (
              <div className={cn("animate-fade-up", isWide && "grid gap-8 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]")}>
                <div>
                  <div className="border-l-2 border-accent/60 pl-4">
                    <p className="text-ink">{data.executiveSummary}</p>
                    <p className="mt-1 text-[13px] text-ink-faint">
                      Độ tin cậy: {data.confidence}
                      {data.score > 0 && (
                        <>
                          {" · "}điểm CKAN <span className="num">{data.score.toFixed(3)}</span>
                        </>
                      )}
                    </p>
                  </div>

                  <section className="mt-7" aria-labelledby="explain-graph">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                      <h3 id="explain-graph" className="text-lg font-semibold tracking-tight">
                        Đồ thị con
                      </h3>
                      <button onClick={() => canvasRef.current?.fit()} className="btn btn-quiet h-8 px-2.5 text-[13px]">
                        <ScanSearch className="h-4 w-4" strokeWidth={1.75} />
                        Căn giữa
                      </button>
                    </div>
                    <GraphCanvas
                      ref={canvasRef}
                      data={subgraph}
                      accent={DOMAIN_ACCENT[domain]}
                      charge={-320}
                      linkDistance={85}
                      className={cn("rounded-xl border border-line bg-bg", isWide ? "h-[min(64vh,640px)]" : "h-[360px]")}
                    />
                    <GraphLegend kinds={kindsIn(subgraph)} className="mt-3" />
                  </section>
                </div>

                <div className={cn(!isWide && "mt-9")}>
                  <section aria-labelledby="explain-paths">
                    <h3 id="explain-paths" className="text-lg font-semibold tracking-tight">
                      Đường dẫn tri thức
                      {paths.length > 0 && <span className="num ml-2 text-sm font-normal text-ink-faint">{paths.length}</span>}
                    </h3>
                    {paths.length > 0 ? (
                      <>
                        <ul className="mt-1 divide-y divide-line">
                          {visiblePaths.map((p, i) => (
                            <PathRow key={p.id || i} path={p} />
                          ))}
                        </ul>
                        {paths.length > PATH_PREVIEW && (
                          <button
                            onClick={() => setShowAllPaths((v) => !v)}
                            className="mt-1 text-sm font-medium text-accent underline-offset-4 hover:underline"
                          >
                            {showAllPaths ? "Thu gọn" : `Xem cả ${paths.length} đường dẫn`}
                          </button>
                        )}
                      </>
                    ) : (
                      <p className="mt-2 text-sm text-ink-muted">
                        Không có đường dẫn trực tiếp trong đồ thị tri thức. Gợi ý này đến từ độ tương đồng hành vi giữa
                        những người dùng (embedding cộng tác).
                      </p>
                    )}
                  </section>

                  {importance.length > 0 && (
                    <section className="mt-9" aria-labelledby="explain-attr">
                      <h3 id="explain-attr" className="text-lg font-semibold tracking-tight">
                        Tỷ lệ đường dẫn theo loại quan hệ
                      </h3>
                      <dl className="mt-3 space-y-2.5">
                        {importance.map(([feature, pct]) => (
                          <div key={feature} className="grid grid-cols-[minmax(0,10rem)_1fr_3rem] items-center gap-3 text-sm">
                            <dt className="truncate text-ink-muted" title={feature}>
                              {feature}
                            </dt>
                            <dd className="h-1.5 overflow-hidden rounded-full bg-line-strong">
                              <div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, pct)}%` }} />
                            </dd>
                            <dd className="num text-right text-ink">{pct}%</dd>
                          </div>
                        ))}
                      </dl>
                    </section>
                  )}

                  {data.counterfactual && (
                    <section className="mt-9" aria-labelledby="explain-cf">
                      <h3 id="explain-cf" className="text-lg font-semibold tracking-tight">
                        Nếu bỏ lượt thích này
                      </h3>
                      <p className="mt-2 text-sm text-ink-muted">{data.counterfactual}</p>
                    </section>
                  )}
                </div>
              </div>
            )}
          </div>

          {onOpenFullGraph && (
            <footer className="flex items-center justify-end border-t border-line px-5 py-3 sm:px-7">
              <button onClick={onOpenFullGraph} className="btn btn-ghost">
                Mở đồ thị đầy đủ của người dùng
                <ArrowRight className="h-4 w-4" strokeWidth={1.75} />
              </button>
            </footer>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
