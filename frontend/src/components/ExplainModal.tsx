import React, { useEffect, useState, useRef, useCallback, useMemo } from "react";
import {
  X,
  Sparkles,
  Network,
  BarChart3,
  ArrowRight,
  ShieldCheck,
  Maximize2,
  Minimize2,
  RotateCcw,
  ListTree,
  Film,
  User as UserIcon,
  Compass,
} from "lucide-react";
import ForceGraph2D from "react-force-graph-2d";
import { ExplanationResponse } from "../types";
import { api } from "../services/api";

interface ExplainModalProps {
  movieId: number | null;
  userId: number;
  onClose: () => void;
  onOpenFullGraph?: () => void;
}

export const ExplainModal: React.FC<ExplainModalProps> = ({
  movieId,
  userId,
  onClose,
  onOpenFullGraph,
}) => {
  const [data, setData] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState<"visual-graph" | "narrative">("visual-graph");
  const [isFullscreen, setIsFullscreen] = useState(false);

  // ForceGraph references & container sizing
  const fgRef = useRef<any>(null);
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [graphDimensions, setGraphDimensions] = useState({ width: 620, height: 320 });
  const [hoveredNode, setHoveredNode] = useState<any | null>(null);

  useEffect(() => {
    if (movieId !== null) {
      setLoading(true);
      api.explainMovie(movieId, userId)
        .then(setData)
        .catch((err) => console.error("Failed to fetch explanation:", err))
        .finally(() => setLoading(false));
    } else {
      setData(null);
    }
  }, [movieId, userId]);

  // Escape key exits fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFullscreen) {
        setIsFullscreen(false);
        setTimeout(() => fgRef.current?.zoomToFit(400, 50), 300);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    setIsFullscreen((prev) => !prev);
    setTimeout(() => {
      fgRef.current?.zoomToFit(400, 50);
    }, 300);
  };

  // Measure container dimensions for the embedded ForceGraph
  useEffect(() => {
    if (!graphContainerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setGraphDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height || 360,
        });
      }
    });
    observer.observe(graphContainerRef.current);
    return () => observer.disconnect();
  }, [activeView, isFullscreen, data]);

  // Prepare graph data from backend subgraph or fallback paths
  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };

    // Prefer full subgraph provided by backend
    if (data.subgraph && data.subgraph.nodes && data.subgraph.nodes.length > 0) {
      const nodes = data.subgraph.nodes.map((n) => {
        let color = "#94A3B8";
        let radius = 6;

        if (n.type === "User") {
          color = "#E5A93C"; // Gold Amber
          radius = 9;
        } else if (n.type === "MovieLiked") {
          color = "#EF4444"; // Liked Movie Coral
          radius = 7.5;
        } else if (n.type === "MovieRecommended") {
          color = "#3B82F6"; // Recommended Movie Azure
          radius = 8;
        } else if (n.type === "Director") {
          color = "#A855F7"; // Violet
          radius = 6;
        } else if (n.type === "Genre") {
          color = "#F59E0B"; // Warm Amber
          radius = 6;
        } else if (n.type === "Actor") {
          color = "#10B981"; // Emerald
          radius = 6;
        } else if (n.type === "Writer") {
          color = "#06B6D4"; // Cyan
          radius = 5.5;
        } else if (n.type === "Producer") {
          color = "#F97316"; // Orange
          radius = 5.5;
        }

        return {
          id: n.id,
          name: n.label,
          type: n.type,
          color,
          radius,
          val: radius,
          raw: n,
        };
      });

      const links = data.subgraph.edges.map((e, idx) => ({
        id: e.id || `l-${idx}`,
        source: e.source,
        target: e.target,
        label: e.label || e.type,
        type: e.type,
      }));

      return { nodes, links };
    }

    // Fallback: build visual graph from paths
    const nodeMap: Record<string, any> = {};
    const linkList: any[] = [];

    const userKey = `user-${data.userId}`;
    nodeMap[userKey] = {
      id: userKey,
      name: `User #${data.userId}`,
      type: "User",
      color: "#E5A93C",
      radius: 9,
      val: 9,
    };

    const targetKey = `target-movie-${data.movieId}`;
    nodeMap[targetKey] = {
      id: targetKey,
      name: data.paths[0]?.targetMovieTitle || `Phim đề xuất #${data.movieId}`,
      type: "MovieRecommended",
      color: "#3B82F6",
      radius: 8,
      val: 8,
    };

    data.paths.forEach((p, i) => {
      const srcKey = `liked-${i}`;
      if (!nodeMap[srcKey]) {
        nodeMap[srcKey] = {
          id: srcKey,
          name: p.sourceMovieTitle,
          type: "MovieLiked",
          color: "#EF4444",
          radius: 7,
          val: 7,
        };
        linkList.push({
          id: `e-u-${i}`,
          source: userKey,
          target: srcKey,
          label: "LIKED",
          type: "LIKED",
        });
      }

      const entKey = `ent-${p.entityName}`;
      let entColor = "#10B981";
      if (p.relation.includes("DIRECT")) entColor = "#A855F7";
      else if (p.relation.includes("GENRE")) entColor = "#F59E0B";

      if (!nodeMap[entKey]) {
        nodeMap[entKey] = {
          id: entKey,
          name: p.entityName,
          type: "Entity",
          color: entColor,
          radius: 6,
          val: 6,
        };
      }

      linkList.push({
        id: `e-src-${i}`,
        source: srcKey,
        target: entKey,
        label: p.relation,
        type: p.relation,
      });

      linkList.push({
        id: `e-tgt-${i}`,
        source: targetKey,
        target: entKey,
        label: p.relation,
        type: p.relation,
      });
    });

    return {
      nodes: Object.values(nodeMap),
      links: linkList,
    };
  }, [data]);

  // Adjust D3 force simulation when graph opens
  useEffect(() => {
    if (activeView === "visual-graph" && graphData.nodes.length > 0 && fgRef.current) {
      fgRef.current.d3Force("charge")?.strength(-320);
      fgRef.current.d3Force("link")?.distance(85);
      fgRef.current.d3ReheatSimulation();

      const timer = setTimeout(() => {
        fgRef.current?.zoomToFit(400, 45);
      }, 350);
      return () => clearTimeout(timer);
    }
  }, [activeView, graphData]);

  // Canvas Node rendering
  const drawNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isHovered = hoveredNode === node;
      const radius = node.radius || 6;

      ctx.save();

      // Outer glow
      if (isHovered) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 4, 0, 2 * Math.PI, false);
        ctx.fillStyle = "rgba(229, 169, 60, 0.4)";
        ctx.fill();
      }

      // Circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
      ctx.fillStyle = node.color;
      ctx.fill();

      // Border
      ctx.lineWidth = isHovered ? 2 : 1.2;
      ctx.strokeStyle = isHovered ? "#FFFFFF" : "rgba(255, 255, 255, 0.5)";
      ctx.stroke();

      // Label text
      const fontSize = Math.max(9 / globalScale, 2.8);
      ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const text = node.name || node.id;
      const textY = node.y + radius + fontSize + 2;

      const textWidth = ctx.measureText(text).width;
      const paddingX = 3;
      const paddingY = 1.2;

      ctx.fillStyle = "rgba(11, 14, 23, 0.9)";
      ctx.beginPath();
      ctx.roundRect(
        node.x - textWidth / 2 - paddingX,
        textY - fontSize / 2 - paddingY,
        textWidth + paddingX * 2,
        fontSize + paddingY * 2,
        3
      );
      ctx.fill();
      ctx.strokeStyle = isHovered ? "rgba(229, 169, 60, 0.6)" : "rgba(255, 255, 255, 0.12)";
      ctx.stroke();

      ctx.fillStyle = isHovered ? "#F8FAFC" : "#E2E8F0";
      ctx.fillText(text, node.x, textY);

      ctx.restore();
    },
    [hoveredNode]
  );

  // Canvas Link rendering with relationship labels
  const drawLinkCanvas = useCallback(
    (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const source = link.source;
      const target = link.target;
      if (!source || !target || typeof source.x !== "number" || typeof target.x !== "number") return;

      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2;
      const label = link.label || link.type;
      if (!label) return;

      const fontSize = Math.max(7.5 / globalScale, 2.3);
      ctx.save();
      ctx.font = `600 ${fontSize}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
      ctx.beginPath();
      ctx.roundRect(midX - textWidth / 2 - 2.5, midY - fontSize / 2 - 1, textWidth + 5, fontSize + 2, 2.5);
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      ctx.stroke();

      ctx.fillStyle = "#94A3B8";
      ctx.fillText(label, midX, midY);
      ctx.restore();
    },
    []
  );

  if (movieId === null) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center ${
        isFullscreen ? "p-0" : "p-4"
      } bg-black/85 backdrop-blur-md animate-in fade-in duration-200`}
    >
      <div
        className={`relative ${
          isFullscreen
            ? "w-screen h-screen max-w-none rounded-none p-4 sm:p-6 flex flex-col bg-[#07090E] overflow-hidden"
            : "w-full max-w-3xl bg-[#0F121A] border border-white/15 rounded-3xl p-6 sm:p-8 shadow-2xl shadow-black/95 max-h-[92vh] overflow-y-auto"
        } text-white transition-all duration-300`}
      >
        {/* Modal Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-white/10">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="w-11 h-11 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shadow-glow-amber shrink-0">
              <Network className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-xl font-bold tracking-tight">Trực quan hóa Suy luận Đồ thị</h2>
                <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary border border-primary/30 text-[10px] font-mono font-bold shrink-0">
                  CKAN Multi-Hop
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5 truncate">
                Đường dẫn liên kết từ phim bạn đã thích đến phim được đề xuất
              </p>
            </div>
          </div>

          {/* Right Controls: View Mode Switcher + Close Button (Flex Aligned) */}
          <div className="flex items-center gap-2.5 self-end sm:self-auto shrink-0">
            {/* View Mode Toggle Tabs */}
            <div className="flex items-center p-1 rounded-xl bg-slate-900/90 border border-white/10">
              <button
                onClick={() => setActiveView("visual-graph")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeView === "visual-graph"
                    ? "bg-primary text-black shadow-md font-bold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Network className="w-3.5 h-3.5" />
                <span>Đồ thị Trực quan</span>
              </button>
              <button
                onClick={() => setActiveView("narrative")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeView === "narrative"
                    ? "bg-primary text-black shadow-md font-bold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <ListTree className="w-3.5 h-3.5" />
                <span>Chi tiết & Chỉ số</span>
              </button>
            </div>

            {/* Close button */}
            <button
              onClick={onClose}
              aria-label="Close explanation modal"
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-all active:scale-95 bg-slate-900/80 border border-white/10 shrink-0"
              title="Đóng"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {loading ? (
          <div className="py-20 flex flex-col items-center justify-center text-center">
            <div className="w-9 h-9 rounded-full border-2 border-primary border-t-transparent animate-spin mb-3" />
            <p className="text-xs text-slate-400 font-mono">Đang truy vấn đường dẫn tri thức từ Neo4j & CKAN...</p>
          </div>
        ) : data ? (
          <div className="space-y-6">
            {/* Executive Summary Card with Target Movie Poster */}
            {(() => {
              const targetNode = data.subgraph?.nodes?.find((n) => n.type === "MovieRecommended");
              const poster = targetNode?.data?.posterUrl;
              return (
                <div className="p-4 rounded-2xl bg-gradient-to-r from-primary/10 via-slate-900 to-slate-900 border border-primary/30 flex items-center gap-4 shadow-lg">
                  {poster && (
                    <div className="w-16 h-24 rounded-xl overflow-hidden bg-black/60 border border-white/15 shrink-0 shadow-md hidden sm:block">
                      <img
                        src={poster}
                        alt={targetNode?.label || "Poster"}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] uppercase font-bold tracking-wider text-primary font-mono flex items-center gap-1.5">
                        <ShieldCheck className="w-4 h-4 text-primary shrink-0" />
                        <span>Tổng quan Lý giải • Độ tin cậy: {data.confidence}</span>
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 shrink-0">
                        User #{data.userId}
                      </span>
                    </div>
                    {targetNode && (
                      <div className="flex items-center gap-2 mt-1">
                        <h3 className="text-base font-bold text-white truncate">
                          {targetNode.label}
                        </h3>
                        {targetNode.data?.releaseYear && (
                          <span className="text-xs font-semibold text-primary font-mono px-1.5 py-0.5 rounded bg-primary/10 border border-primary/20 shrink-0">
                            {targetNode.data.releaseYear}
                          </span>
                        )}
                      </div>
                    )}
                    <p className="text-xs font-medium text-slate-300 mt-1 leading-relaxed">
                      {data.executiveSummary}
                    </p>
                  </div>
                </div>
              );
            })()}

            {/* TAB 1: VISUAL INTERACTIVE GRAPH */}
            {activeView === "visual-graph" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    <span>Mô hình Đường dẫn Tri thức Tương tác</span>
                  </span>

                  <div className="flex items-center gap-2">
                    {/* Fullscreen toggle button */}
                    <button
                      onClick={toggleFullscreen}
                      className="text-[11px] text-slate-300 hover:text-primary px-2.5 py-1 rounded-lg bg-white/5 border border-white/10 flex items-center gap-1 font-mono transition-colors"
                      title={isFullscreen ? "Thu nhỏ (Esc)" : "Toàn màn hình"}
                    >
                      {isFullscreen ? (
                        <>
                          <Minimize2 className="w-3.5 h-3.5 text-primary" />
                          <span>Thu nhỏ</span>
                        </>
                      ) : (
                        <>
                          <Maximize2 className="w-3.5 h-3.5 text-primary" />
                          <span>Toàn màn hình</span>
                        </>
                      )}
                    </button>

                    {/* Reset zoom */}
                    <button
                      onClick={() => fgRef.current?.zoomToFit(400, 45)}
                      className="text-[11px] text-slate-400 hover:text-primary px-2.5 py-1 rounded-lg bg-white/5 border border-white/10 flex items-center gap-1 font-mono transition-colors"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Căn chỉnh lại</span>
                    </button>
                  </div>
                </div>

                {/* Embedded Force Graph Canvas */}
                <div
                  ref={graphContainerRef}
                  className={`w-full ${
                    isFullscreen ? "flex-1 min-h-[420px]" : "h-[360px]"
                  } rounded-2xl bg-[#07090E] border border-white/10 overflow-hidden relative shadow-inner`}
                >
                  <ForceGraph2D
                    ref={fgRef}
                    width={graphDimensions.width}
                    height={graphDimensions.height}
                    graphData={graphData}
                    nodeCanvasObject={drawNode}
                    nodePointerAreaPaint={(node: any, color, ctx) => {
                      const radius = (node.radius || 6) + 4;
                      ctx.fillStyle = color;
                      ctx.beginPath();
                      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
                      ctx.fill();
                    }}
                    onNodeHover={(node) => setHoveredNode(node)}
                    linkCanvasObjectMode={() => "after"}
                    linkCanvasObject={drawLinkCanvas}
                    linkCurvature={0.08}
                    linkDirectionalParticles={2}
                    linkDirectionalParticleWidth={2}
                    linkDirectionalParticleSpeed={0.007}
                    linkDirectionalParticleColor={() => "#E5A93C"}
                    linkDirectionalArrowLength={4.5}
                    linkDirectionalArrowRelPos={0.65}
                    linkColor={(link: any) => {
                      if (link.type === "LIKED") return "rgba(239, 68, 68, 0.6)";
                      if (link.type?.includes("DIRECT")) return "rgba(168, 85, 247, 0.6)";
                      if (link.type?.includes("GENRE")) return "rgba(245, 158, 11, 0.6)";
                      if (link.type?.includes("ACT")) return "rgba(16, 185, 129, 0.6)";
                      return "rgba(100, 116, 139, 0.4)";
                    }}
                    linkWidth={1.2}
                    cooldownTicks={100}
                  />

                  {/* Legend Overlay inside Canvas */}
                  <div className="absolute top-2.5 left-2.5 flex flex-wrap items-center gap-2 bg-[#0F121A]/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10 text-[10px] font-mono">
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#E5A93C]" />
                      <span className="text-slate-300">Bạn (User)</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#EF4444]" />
                      <span className="text-slate-300">Phim đã thích</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#3B82F6]" />
                      <span className="text-slate-300">Phim gợi ý</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#A855F7]" />
                      <span className="text-slate-300">Đạo diễn</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#10B981]" />
                      <span className="text-slate-300">Diễn viên</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#F59E0B]" />
                      <span className="text-slate-300">Thể loại</span>
                    </div>
                  </div>
                </div>

                {/* Visual Step-by-Step Flow Cards */}
                <div className="space-y-2">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block mb-1">
                    Chuỗi liên kết cụ thể (Multi-hop Walk):
                  </span>
                  {data.paths && data.paths.length > 0 ? (
                    data.paths.slice(0, 3).map((p, i) => (
                      <div
                        key={p.id || i}
                        className="p-3 rounded-xl bg-slate-900/80 border border-white/5 flex flex-col gap-1.5"
                      >
                        <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono">
                          <span className="px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20 font-semibold">
                            🎬 {p.sourceMovieTitle}
                          </span>
                          <ArrowRight className="w-3.5 h-3.5 text-primary shrink-0" />
                          <span className="px-2 py-0.5 rounded bg-white/5 text-primary border border-white/10 text-[10px] uppercase">
                            {p.relation}
                          </span>
                          <ArrowRight className="w-3.5 h-3.5 text-primary shrink-0" />
                          <span className="px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30 font-semibold">
                            {p.entityName}
                          </span>
                          <ArrowRight className="w-3.5 h-3.5 text-primary shrink-0" />
                          <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold">
                            ⭐ {p.targetMovieTitle}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-300 italic pl-1">{p.naturalLanguage}</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-xs text-slate-400 italic">
                      Được đề xuất dựa trên phân phối tương đồng hành vi người dùng (Collaborative Embedding).
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* TAB 2: DETAILED NARRATIVE & METRICS */}
            {activeView === "narrative" && (
              <div className="space-y-5">
                {/* Full list of reasoning paths */}
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2 mb-3">
                    <ListTree className="w-4 h-4 text-primary" />
                    <span>Tất cả các đường dẫn tri thức (Knowledge Paths)</span>
                  </h3>
                  <div className="space-y-2.5 max-h-[220px] overflow-y-auto pr-1">
                    {data.paths.map((p, i) => (
                      <div
                        key={p.id || i}
                        className="p-3 rounded-xl bg-slate-900/70 border border-white/5 text-xs text-slate-300"
                      >
                        <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px] text-slate-400 mb-1">
                          <span className="text-white font-semibold">{p.sourceMovieTitle}</span>
                          <ArrowRight className="w-3 h-3 text-primary shrink-0" />
                          <span className="px-1.5 py-0.5 rounded bg-white/5 text-primary border border-white/10 text-[10px]">
                            {p.relation}
                          </span>
                          <ArrowRight className="w-3 h-3 text-primary shrink-0" />
                          <span className="text-amber-300 font-semibold">{p.entityName}</span>
                          <ArrowRight className="w-3 h-3 text-primary shrink-0" />
                          <span className="text-blue-400 font-semibold">{p.targetMovieTitle}</span>
                        </div>
                        <p className="text-[11px] text-slate-300 italic">{p.naturalLanguage}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Feature Importance Attribution */}
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2 mb-2.5">
                    <BarChart3 className="w-4 h-4 text-primary" />
                    <span>Mức độ đóng góp của các thuộc tính (Attribution Breakdown)</span>
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {Object.entries(data.featureImportance || {}).map(([feature, pct]) => (
                      <div key={feature} className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1.5">
                        <div className="flex justify-between text-[11px] text-slate-300 font-mono">
                          <span>{feature}</span>
                          <span className="text-primary font-bold">{pct}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden border border-white/5">
                          <div
                            className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all duration-500"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Counterfactual Sensitivity */}
                <div className="p-3.5 rounded-xl bg-slate-900/50 border border-white/5 text-xs text-slate-400">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-slate-300 font-mono block mb-1">
                    Phân tích phản thực nghiệm (Counterfactual Analysis)
                  </span>
                  <p className="leading-relaxed">{data.counterfactual}</p>
                </div>
              </div>
            )}

            {/* Bottom Footer Actions */}
            <div className="pt-4 border-t border-white/10 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="text-[11px] text-slate-400 font-mono flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5 text-primary" />
                <span>Kéo thả node trên đồ thị để kiểm tra các mối liên kết</span>
              </div>

              {onOpenFullGraph && (
                <button
                  onClick={onOpenFullGraph}
                  className="px-4 py-2 rounded-xl bg-white/10 hover:bg-primary hover:text-black text-white text-xs font-bold transition-all flex items-center justify-center gap-2 active:scale-95 shadow-lg border border-white/10"
                >
                  <Network className="w-4 h-4" />
                  <span>Mở toàn cảnh trên Knowledge Graph</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-400">Không có dữ liệu lý giải cho phim này.</p>
        )}
      </div>
    </div>
  );
};
