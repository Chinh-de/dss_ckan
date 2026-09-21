import React, { useEffect, useState, useRef, useMemo, useCallback } from "react";
import ForceGraph2D, { ForceGraphMethods } from "react-force-graph-2d";
import {
  Share2,
  Film,
  User,
  Tag,
  Clapperboard,
  Sparkles,
  Layers,
  Compass,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  Info,
  ChevronRight,
  Filter,
  Play,
  Pause,
} from "lucide-react";
import { SubgraphData } from "../types";
import { api } from "../services/api";

interface KnowledgeGraphViewProps {
  userId: number;
}

type FilterType = "ALL" | "DIRECTOR" | "GENRE" | "ACTOR";

export const KnowledgeGraphView: React.FC<KnowledgeGraphViewProps> = ({ userId }) => {
  const [data, setData] = useState<SubgraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [filterType, setFilterType] = useState<FilterType>("ALL");
  const [selectedNode, setSelectedNode] = useState<any | null>(null);
  const [hoverNode, setHoverNode] = useState<any | null>(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set<string>());
  const [highlightLinks, setHighlightLinks] = useState(new Set<string>());
  const [particlesActive, setParticlesActive] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const fgRef = useRef<ForceGraphMethods>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 900, height: 680 });

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

  // Update canvas size responsively
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth || 900,
          height: containerRef.current.clientHeight || 680,
        });
      }
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [isFullscreen]);

  const fetchUserGraph = async () => {
    try {
      setLoading(true);
      const res = await api.getUserSubgraph(userId, 8);
      setData(res);
    } catch (err: any) {
      console.error("Error fetching user subgraph:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUserGraph();
  }, [userId]);

  // Transform subgraph data for Force Graph with collision physics & filtering
  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };

    let activeNodes = data.nodes;
    if (filterType !== "ALL") {
      const allowedTypes = new Set(["User", "MovieLiked", "MovieRecommended"]);
      if (filterType === "DIRECTOR") allowedTypes.add("Director");
      if (filterType === "GENRE") allowedTypes.add("Genre");
      if (filterType === "ACTOR") allowedTypes.add("Actor");

      activeNodes = data.nodes.filter((n) => allowedTypes.has(n.type));
    }

    const activeNodeIds = new Set(activeNodes.map((n) => n.id));
    const activeEdges = data.edges.filter(
      (e) => activeNodeIds.has(e.source) && activeNodeIds.has(e.target)
    );

    // Format nodes with visual radii and color tokens
    const nodes = activeNodes.map((n) => {
      let color = "#64748B";
      let val = 8;
      let label = n.label;

      // Clean label if numeric
      if (/^\d+$/.test(String(label).trim())) {
        label = `${n.type} #${label}`;
      }

      let radius = 5;
      if (n.type === "User") {
        color = "#E5A93C"; // Gold Amber
        radius = 8.5;
      } else if (n.type === "MovieLiked") {
        color = "#EF4444"; // Director Crimson
        radius = 7;
      } else if (n.type === "MovieRecommended") {
        color = "#3B82F6"; // Cyan Azure
        radius = 7;
      } else if (n.type === "Director") {
        color = "#A855F7"; // Royal Violet
        radius = 5.5;
      } else if (n.type === "Genre") {
        color = "#F59E0B"; // Warm Amber
        radius = 5.5;
      } else if (n.type === "Actor") {
        color = "#10B981"; // Emerald
        radius = 5.5;
      } else if (n.type === "Writer") {
        color = "#06B6D4"; // Cyan
        radius = 5;
      } else if (n.type === "Producer") {
        color = "#F97316"; // Orange
        radius = 5;
      }

      return {
        id: n.id,
        name: label,
        type: n.type,
        color,
        radius,
        val: radius,
        raw: n,
      };
    });

    const links = activeEdges.map((e, idx) => ({
      id: e.id || `l-${idx}`,
      source: e.source,
      target: e.target,
      label: e.label || e.type,
      type: e.type,
    }));

    return { nodes, links };
  }, [data, filterType]);

  // Configure physics forces & camera on graph load
  useEffect(() => {
    if (graphData.nodes.length > 0 && fgRef.current) {
      // Repulsion force: strong negative charge to push nodes far apart
      fgRef.current.d3Force("charge")?.strength(-450);
      // Link distance: spacious edges to give room for labels and avoid clumping
      fgRef.current.d3Force("link")?.distance(110);
      fgRef.current.d3ReheatSimulation();

      const timer = setTimeout(() => {
        fgRef.current?.zoomToFit(400, 70);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [graphData]);

  // Interactive hover highlight
  const handleNodeHover = (node: any | null) => {
    const nextHighlightNodes = new Set<string>();
    const nextHighlightLinks = new Set<string>();

    if (node) {
      nextHighlightNodes.add(node.id);
      graphData.links.forEach((link: any) => {
        const sourceId = typeof link.source === "object" ? link.source.id : link.source;
        const targetId = typeof link.target === "object" ? link.target.id : link.target;
        if (sourceId === node.id || targetId === node.id) {
          nextHighlightLinks.add(link.id);
          nextHighlightNodes.add(sourceId);
          nextHighlightNodes.add(targetId);
        }
      });
    }

    setHoverNode(node);
    setHighlightNodes(nextHighlightNodes);
    setHighlightLinks(nextHighlightLinks);
  };

  // Custom Canvas Rendering for Nodes (Cinematic Luxury Theme)
  const drawNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isHighlighted = highlightNodes.has(node.id);
      const isDimmed = hoverNode && !isHighlighted;
      const radius = node.radius || 5.5;

      ctx.save();
      ctx.globalAlpha = isDimmed ? 0.2 : 1.0;

      // 1. Outer Glow halo if hovered / highlighted
      if (isHighlighted || node === hoverNode) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 4, 0, 2 * Math.PI, false);
        ctx.fillStyle = "rgba(229, 169, 60, 0.4)";
        ctx.fill();
      }

      // 2. Main Node Circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
      ctx.fillStyle = node.color;
      ctx.fill();

      // Border stroke
      ctx.lineWidth = isHighlighted ? 2 : 1.2;
      ctx.strokeStyle = isHighlighted ? "#FFFFFF" : "rgba(255, 255, 255, 0.5)";
      ctx.stroke();

      // 3. Label Text (Only draw if node is major, hovered, highlighted, or when zoomed in)
      const isMajor = node.type === "User" || node.type?.includes("Movie");
      const shouldDraw = isHighlighted || node === hoverNode || isMajor || globalScale >= 0.85;

      if (shouldDraw) {
        const fontSize = Math.max(9.5 / globalScale, 3);
        ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        const text = node.name || node.id;
        const textY = node.y + radius + fontSize + 2;

        // Compact Pill Background for Text
        const textWidth = ctx.measureText(text).width;
        const paddingX = 3.5;
        const paddingY = 1.5;

        ctx.fillStyle = "rgba(11, 14, 23, 0.9)";
        ctx.beginPath();
        ctx.roundRect(
          node.x - textWidth / 2 - paddingX,
          textY - fontSize / 2 - paddingY,
          textWidth + paddingX * 2,
          fontSize + paddingY * 2,
          3.5
        );
        ctx.fill();
        ctx.strokeStyle = isHighlighted ? "rgba(229, 169, 60, 0.6)" : "rgba(255, 255, 255, 0.12)";
        ctx.stroke();

        // Text Foreground
        ctx.fillStyle = isHighlighted ? "#F8FAFC" : "#CBD5E1";
        ctx.fillText(text, node.x, textY);
      }

      ctx.restore();
    },
    [highlightNodes, hoverNode]
  );

  // Custom Canvas Rendering for Links to show Relationship Labels
  const drawLinkCanvas = useCallback(
    (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isHighlighted = highlightLinks.has(link.id);
      const shouldShowLabel = isHighlighted || globalScale >= 0.95;
      if (!shouldShowLabel) return;

      const source = link.source;
      const target = link.target;
      if (!source || !target || typeof source.x !== "number" || typeof target.x !== "number") return;

      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2;
      const label = link.label || link.type;
      if (!label) return;

      const fontSize = Math.max(8 / globalScale, 2.5);
      ctx.save();
      ctx.font = `600 ${fontSize}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = isHighlighted ? "rgba(229, 169, 60, 0.95)" : "rgba(15, 23, 42, 0.85)";
      ctx.beginPath();
      ctx.roundRect(midX - textWidth / 2 - 3, midY - fontSize / 2 - 1.5, textWidth + 6, fontSize + 3, 3);
      ctx.fill();
      ctx.strokeStyle = isHighlighted ? "#E5A93C" : "rgba(255, 255, 255, 0.15)";
      ctx.stroke();

      ctx.fillStyle = isHighlighted ? "#0B0E17" : "#94A3B8";
      ctx.fillText(label, midX, midY);
      ctx.restore();
    },
    [highlightLinks]
  );

  return (
    <div
      className={
        isFullscreen
          ? "fixed inset-0 z-[60] bg-[#07090E] p-4 flex flex-col space-y-3 overflow-hidden animate-in fade-in duration-200"
          : "space-y-4 pb-12"
      }
    >
      {/* Header Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[#0F121A] border border-white/10 p-4 rounded-2xl shadow-xl">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Share2 className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              Đồ Thị Tri Thức Đa Tầng (Knowledge Graph)
            </h2>
            <span className="text-[10px] font-mono bg-primary/20 text-primary border border-primary/30 px-2.5 py-0.5 rounded-full font-semibold">
              Mô Phỏng Lực D3
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Khám phá mạng lưới liên kết tri thức tự cân bằng (Force-Directed Simulation) giữa{" "}
            <strong className="text-amber-400">User #{userId}</strong>, các phim đã thích, và các thực thể giải thích.
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Entity Filter Pills */}
          <div className="flex items-center bg-black/40 border border-white/10 rounded-xl p-1 text-xs">
            {(["ALL", "DIRECTOR", "GENRE", "ACTOR"] as FilterType[]).map((ft) => (
              <button
                key={ft}
                onClick={() => setFilterType(ft)}
                className={`px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all ${
                  filterType === ft
                    ? "bg-primary text-black font-bold shadow-md"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {ft === "ALL"
                  ? "Tất Cả"
                  : ft === "DIRECTOR"
                  ? "Đạo Diễn"
                  : ft === "GENRE"
                  ? "Thể Loại"
                  : "Diễn Viên"}
              </button>
            ))}
          </div>

          {/* Animation Toggle */}
          <button
            onClick={() => setParticlesActive(!particlesActive)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs border border-white/10 transition-all ${
              particlesActive
                ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                : "bg-card text-slate-400"
            }`}
            title="Bật/Tắt hiệu ứng luồng hạt tri thức (Energy Particles)"
          >
            {particlesActive ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span className="font-mono text-[11px]">Luồng Hạt</span>
          </button>

          {/* Reset Zoom */}
          <button
            onClick={() => fgRef.current?.zoomToFit(400, 50)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs bg-card hover:bg-slate-800 text-slate-300 border border-white/10 transition-all"
            title="Căn giữa toàn bộ đồ thị"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="font-mono text-[11px]">Căn Giữa</span>
          </button>

          {/* Fullscreen Toggle */}
          <button
            onClick={() => {
              setIsFullscreen(!isFullscreen);
              setTimeout(() => fgRef.current?.zoomToFit(400, 50), 300);
            }}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs border transition-all ${
              isFullscreen
                ? "bg-primary text-black font-bold border-primary shadow-glow-amber"
                : "bg-card hover:bg-slate-800 text-slate-300 border-white/10"
            }`}
            title={isFullscreen ? "Thu nhỏ (Esc)" : "Toàn màn hình"}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            <span className="font-mono text-[11px]">{isFullscreen ? "Thu Nhỏ" : "Toàn Màn Hình"}</span>
          </button>
        </div>
      </div>

      {/* Legend Ribbon */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-300 bg-[#0A0D14] border border-white/5 px-4 py-2 rounded-xl">
        <div className="flex flex-wrap items-center gap-4 text-[11px]">
          <span className="text-slate-500 font-mono uppercase text-[10px]">Thực Thể:</span>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#E5A93C] shadow-sm shadow-amber-500/50" />
            <span className="font-medium text-slate-200">User</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#D9383A] shadow-sm shadow-red-500/50" />
            <span className="font-medium text-slate-200">Phim Đã Like</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#0284C7] shadow-sm shadow-sky-500/50" />
            <span className="font-medium text-slate-200">Phim Được Gợi Ý</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#8B5CF6] shadow-sm shadow-purple-500/50" />
            <span className="font-medium text-slate-200">Đạo Diễn</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#F59E0B] shadow-sm shadow-yellow-500/50" />
            <span className="font-medium text-slate-200">Thể Loại</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#10B981] shadow-sm shadow-emerald-500/50" />
            <span className="font-medium text-slate-200">Diễn Viên</span>
          </div>
        </div>

        <div className="text-[11px] text-slate-400 font-mono flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-primary" />
          <span>Kéo thả node để điều chỉnh • Lăn chuột để phóng to/thu nhỏ</span>
        </div>
      </div>

      {/* Force Graph Canvas */}
      <div
        ref={containerRef}
        className={`w-full ${
          isFullscreen ? "flex-1 min-h-0" : "h-[680px]"
        } rounded-2xl bg-[#07090E] border border-white/10 overflow-hidden relative shadow-2xl`}
      >
        {loading && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-background/85 backdrop-blur-md text-white">
            <div className="w-10 h-10 rounded-full border-2 border-primary border-t-transparent animate-spin mb-3" />
            <p className="text-xs text-slate-300 font-mono">Đang tính toán mô phỏng vật lý đồ thị...</p>
          </div>
        )}

        <ForceGraph2D
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={(node: any, color, ctx) => {
            const radius = (node.radius || 5.5) + 6;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
            ctx.fill();
          }}
          onNodeClick={(node) => setSelectedNode(node)}
          onNodeHover={handleNodeHover}
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={drawLinkCanvas}
          linkCurvature={0.08}
          linkDirectionalParticles={particlesActive ? 2 : 0}
          linkDirectionalParticleWidth={2.5}
          linkDirectionalParticleSpeed={0.006}
          linkDirectionalParticleColor={(link: any) =>
            highlightLinks.has(link.id) ? "#E5A93C" : "rgba(229, 169, 60, 0.7)"
          }
          linkDirectionalArrowLength={5}
          linkDirectionalArrowRelPos={0.65}
          linkColor={(link: any) => {
            if (highlightLinks.has(link.id)) return "#E5A93C";
            if (hoverNode) return "rgba(71, 85, 105, 0.15)";
            if (link.type === "LIKED") return "rgba(239, 68, 68, 0.6)";
            if (link.type?.includes("DIRECT")) return "rgba(168, 85, 247, 0.6)";
            if (link.type?.includes("GENRE")) return "rgba(245, 158, 11, 0.6)";
            if (link.type?.includes("ACT")) return "rgba(16, 185, 129, 0.6)";
            return "rgba(100, 116, 139, 0.4)";
          }}
          linkWidth={(link: any) => (highlightLinks.has(link.id) ? 2.5 : 1.2)}
          linkLabel={(link: any) => link.label}
          cooldownTicks={120}
          d3VelocityDecay={0.3}
          d3AlphaDecay={0.02}
        />

        {/* Floating Node Details Card */}
        {selectedNode && (
          <div className="absolute bottom-5 right-5 z-20 w-80 bg-[#0F121A]/95 backdrop-blur-2xl border border-white/15 rounded-2xl p-4 shadow-2xl animate-in fade-in slide-in-from-bottom-3 duration-200">
            <div className="flex items-start justify-between gap-2 mb-2.5">
              <span
                className="text-[10px] font-mono uppercase font-bold px-2 py-0.5 rounded"
                style={{
                  background: `${selectedNode.color}25`,
                  color: selectedNode.color,
                }}
              >
                {selectedNode.type}
              </span>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-slate-400 hover:text-white text-xs font-mono p-1 hover:bg-white/10 rounded-lg transition-colors"
              >
                ✕
              </button>
            </div>

            {selectedNode.raw?.data?.posterUrl && (
              <div className="w-full h-44 rounded-xl overflow-hidden mb-3 bg-black/60 border border-white/10 shadow-lg relative group">
                <img
                  src={selectedNode.raw.data.posterUrl}
                  alt={selectedNode.name}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60 pointer-events-none" />
              </div>
            )}

            <h4 className="text-base font-bold text-white mb-2 leading-tight">
              {selectedNode.name}
            </h4>
            <div className="text-xs text-slate-400 space-y-1.5 pt-1 border-t border-white/5">
              <div className="flex justify-between">
                <span className="text-slate-500">Phân loại:</span>
                <span className="text-slate-200 font-semibold">{selectedNode.type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Mã thực thể:</span>
                <span className="text-slate-300 font-mono text-[11px]">{selectedNode.id}</span>
              </div>
              {selectedNode.raw?.data?.releaseYear && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Năm phát hành:</span>
                  <span className="text-primary font-mono text-[11px] font-bold">
                    {selectedNode.raw.data.releaseYear}
                  </span>
                </div>
              )}
              {selectedNode.raw?.data?.genres && selectedNode.raw.data.genres.length > 0 && (
                <div className="flex justify-between items-center pt-0.5">
                  <span className="text-slate-500">Thể loại:</span>
                  <div className="flex flex-wrap gap-1 justify-end max-w-[170px]">
                    {selectedNode.raw.data.genres.slice(0, 3).map((g: string) => (
                      <span
                        key={g}
                        className="px-1.5 py-0.5 rounded bg-white/5 text-[10px] text-slate-300 font-mono"
                      >
                        {g}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
