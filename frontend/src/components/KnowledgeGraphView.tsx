import React, { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Maximize2, Minimize2, Pause, Play, ScanSearch, X } from "lucide-react";
import { SubgraphData, DomainType } from "../types";
import { api } from "../services/api";
import { CORE_KINDS, DOMAIN_ACCENT, DOMAIN_META, NODE_STYLE, NodeKind } from "../lib/theme";
import { cn } from "../lib/utils";
import { CanvasNode, GraphCanvas, GraphCanvasHandle, GraphLegend, kindsIn } from "./GraphCanvas";
import { PageHeader, Poster } from "./ui";

interface KnowledgeGraphViewProps {
  userId: number;
  domain: DomainType;
}

export const KnowledgeGraphView: React.FC<KnowledgeGraphViewProps> = ({ userId, domain }) => {
  const [data, setData] = useState<SubgraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [onlyKind, setOnlyKind] = useState<NodeKind | null>(null);
  const [selectedNode, setSelectedNode] = useState<CanvasNode | null>(null);
  const [particlesActive, setParticlesActive] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const canvasRef = useRef<GraphCanvasHandle>(null);
  const meta = DOMAIN_META[domain];

  useEffect(() => {
    if (!isFullscreen) return;
    const handleKeyDown = (e: KeyboardEvent) => e.key === "Escape" && setIsFullscreen(false);
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  useEffect(() => {
    let cancelled = false;

    const load = async (): Promise<SubgraphData | null> => {
      if (domain === "movie") {
        try {
          const res = await api.getUserSubgraph(userId, 8);
          if (res?.nodes?.length > 0) return res;
        } catch {
          // Neo4j is optional: fall through to the CKAN explainer below.
        }
      }
      // Book, music, or an empty Neo4j graph: explain the top recommendation instead.
      const recs = await api.getRecommendations(domain, userId, 3);
      const top = recs?.recommendations?.[0];
      if (!top) return null;
      const exp = await api.explainItem(top.id ?? top.movieId!, domain, userId);
      return exp?.subgraph ?? null;
    };

    setLoading(true);
    setError(null);
    setSelectedNode(null);
    setOnlyKind(null);
    load()
      .then((res) => !cancelled && setData(res))
      .catch((err) => {
        if (cancelled) return;
        setData(null);
        setError(err.message || "Không tải được đồ thị.");
      })
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [userId, domain]);

  const kinds = useMemo(() => kindsIn(data), [data]);
  const filterKinds = kinds.filter((k) => !CORE_KINDS.includes(k));
  const nodeData = selectedNode?.raw?.data;

  const toolbar = (
    <div className="flex flex-wrap items-center gap-2">
      {filterKinds.length > 1 && (
        <div className="seg" role="group" aria-label="Lọc theo loại thực thể">
          <button onClick={() => setOnlyKind(null)} aria-pressed={onlyKind === null} className="seg-item">
            Tất cả
          </button>
          {filterKinds.map((kind) => (
            <button key={kind} onClick={() => setOnlyKind(kind)} aria-pressed={onlyKind === kind} className="seg-item">
              {NODE_STYLE[kind].label}
            </button>
          ))}
        </div>
      )}
      <button
        onClick={() => setParticlesActive((v) => !v)}
        aria-pressed={particlesActive}
        className="btn btn-ghost"
        title="Bật hoặc tắt hạt chuyển động trên cạnh"
      >
        {particlesActive ? <Pause className="h-4 w-4" strokeWidth={1.75} /> : <Play className="h-4 w-4" strokeWidth={1.75} />}
        Luồng hạt
      </button>
      <button onClick={() => canvasRef.current?.fit()} className="btn btn-ghost">
        <ScanSearch className="h-4 w-4" strokeWidth={1.75} />
        Căn giữa
      </button>
      <button
        onClick={() => setIsFullscreen((v) => !v)}
        className="btn btn-ghost"
        title={isFullscreen ? "Thu nhỏ (Esc)" : undefined}
      >
        {isFullscreen ? <Minimize2 className="h-4 w-4" strokeWidth={1.75} /> : <Maximize2 className="h-4 w-4" strokeWidth={1.75} />}
        {isFullscreen ? "Thu nhỏ" : "Toàn màn hình"}
      </button>
    </div>
  );

  return (
    <div>
      <PageHeader
        eyebrow={`Bài toán 2 · Đường dẫn tri thức · ${meta.dataset}`}
        title={`Đồ thị quanh người dùng #${userId}`}
        description={`Người dùng nối tới những ${meta.item} đã thích; từ đó các thực thể chung trong đồ thị tri thức dẫn sang những ${meta.item} được gợi ý. Rê chuột vào một nút để chỉ giữ lại các liên kết của nó.`}
      />

      <div
        className={cn(
          isFullscreen ? "fixed inset-0 z-40 flex animate-fade-in flex-col gap-3 bg-bg p-4" : "mt-6 flex flex-col gap-3"
        )}
      >
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
          <GraphLegend kinds={kinds} />
          {toolbar}
        </div>

        <div
          className={cn(
            "relative rounded-xl border border-line bg-surface",
            isFullscreen ? "min-h-0 flex-1" : "h-[clamp(420px,68vh,720px)]"
          )}
        >
          <GraphCanvas
            ref={canvasRef}
            data={data}
            onlyKind={onlyKind}
            accent={DOMAIN_ACCENT[domain]}
            particles={particlesActive}
            charge={-450}
            linkDistance={110}
            onNodeClick={setSelectedNode}
            className="absolute inset-0 rounded-xl"
          />

          {loading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 rounded-xl bg-surface/90" aria-busy="true">
              <div className="flex items-end gap-1.5" aria-hidden="true">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="h-2.5 w-2.5 animate-pulse rounded-full bg-accent"
                    style={{ animationDelay: `${i * 180}ms` }}
                  />
                ))}
              </div>
              <p className="text-sm text-ink-muted">Đang truy vấn đồ thị con…</p>
            </div>
          )}

          {!loading && error && (
            <div className="absolute inset-0 flex items-center justify-center p-6">
              <div className="notice notice-error max-w-md" role="alert">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-neg" strokeWidth={1.75} />
                <p>{error}</p>
              </div>
            </div>
          )}

          {!loading && !error && (!data || data.nodes.length === 0) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
              <h3 className="text-lg font-semibold">Chưa có đồ thị để hiển thị</h3>
              <p className="mt-1.5 max-w-sm text-sm text-ink-muted">
                Người dùng #{userId} chưa có gợi ý nào trong tập {meta.dataset}, nên không có đường dẫn tri thức để vẽ.
              </p>
            </div>
          )}

          {selectedNode && (
            <aside className="absolute bottom-3 right-3 z-10 w-72 max-w-[calc(100%-1.5rem)] animate-pop-in rounded-xl border border-line-strong bg-raised/95 p-4 shadow-lift backdrop-blur-xl">
              <div className="flex items-start justify-between gap-2">
                <span className="flex items-center gap-2 text-[13px] text-ink-muted">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: selectedNode.color }} />
                  {NODE_STYLE[selectedNode.kind].label}
                </span>
                <button onClick={() => setSelectedNode(null)} aria-label="Đóng chi tiết nút" className="icon-btn -mr-2 -mt-2 h-8 w-8">
                  <X className="h-4 w-4" strokeWidth={1.75} />
                </button>
              </div>

              <div className="mt-2 flex gap-3">
                {nodeData?.posterUrl && (
                  <Poster src={nodeData.posterUrl} title={selectedNode.name} domain={domain} className="h-24 w-16 shrink-0 rounded-md" />
                )}
                <div className="min-w-0">
                  <h4 className="font-display text-lg font-semibold leading-tight">{selectedNode.name}</h4>
                  <dl className="mt-2 space-y-0.5 text-[13px]">
                    {nodeData?.releaseYear && (
                      <div className="flex gap-2">
                        <dt className="text-ink-faint">Năm</dt>
                        <dd className="num">{nodeData.releaseYear}</dd>
                      </div>
                    )}
                    <div className="flex gap-2">
                      <dt className="text-ink-faint">Mã</dt>
                      <dd className="num truncate text-ink-muted">{selectedNode.id}</dd>
                    </div>
                  </dl>
                </div>
              </div>

              {nodeData?.genres?.length > 0 && (
                <p className="mt-3 text-[13px] text-ink-muted">{nodeData.genres.slice(0, 4).join(" · ")}</p>
              )}
            </aside>
          )}
        </div>

        <p className="text-[13px] text-ink-faint">Kéo để di chuyển nút, cuộn để phóng to, bấm vào nút để xem chi tiết.</p>
      </div>
    </div>
  );
};
