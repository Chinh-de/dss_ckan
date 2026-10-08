import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import ForceGraph2D, { ForceGraphMethods } from "react-force-graph-2d";
import { GraphNode, SubgraphData } from "../types";
import { INK_HEX, NODE_STYLE, NodeKind, SURFACE_HEX, nodeKind } from "../lib/theme";
import { useElementSize } from "../lib/useElementSize";
import { cn } from "../lib/utils";

export interface CanvasNode {
  id: string;
  name: string;
  type: string;
  kind: NodeKind;
  color: string;
  radius: number;
  val: number;
  raw: GraphNode;
  x?: number;
  y?: number;
}

interface CanvasLink {
  id: string;
  source: any;
  target: any;
  label: string;
  type: string;
}

export interface GraphCanvasHandle {
  fit: () => void;
}

interface GraphCanvasProps {
  data: SubgraphData | null;
  /** When set, only the user, liked and recommended nodes plus this kind are drawn. */
  onlyKind?: NodeKind | null;
  accent: string;
  particles?: boolean;
  charge?: number;
  linkDistance?: number;
  onNodeClick?: (node: CanvasNode) => void;
  className?: string;
}

const FONT = '"Be Vietnam Pro", system-ui, sans-serif';
const CORE = new Set<NodeKind>(["user", "liked", "recommended"]);
const endId = (end: any) => (typeof end === "object" ? end.id : end);

/** Force-directed subgraph shared by the full graph page and the explanation panel. */
export const GraphCanvas = forwardRef<GraphCanvasHandle, GraphCanvasProps>(
  ({ data, onlyKind, accent, particles = true, charge = -380, linkDistance = 100, onNodeClick, className }, ref) => {
    const fgRef = useRef<ForceGraphMethods>();
    const [boxRef, { width, height }] = useElementSize<HTMLDivElement>();
    const [hoverId, setHoverId] = useState<string | null>(null);

    const fit = useCallback(() => fgRef.current?.zoomToFit(400, 56), []);
    useImperativeHandle(ref, () => ({ fit }), [fit]);

    const graphData = useMemo(() => {
      if (!data) return { nodes: [] as CanvasNode[], links: [] as CanvasLink[] };

      const nodes: CanvasNode[] = data.nodes
        .map((n) => {
          const kind = nodeKind(n.type);
          const style = NODE_STYLE[kind];
          const label = String(n.label ?? n.id).trim();
          return {
            id: n.id,
            name: /^\d+$/.test(label) ? `${style.label} #${label}` : label,
            type: n.type,
            kind,
            color: style.color,
            radius: style.radius,
            val: style.radius,
            raw: n,
          };
        })
        .filter((n) => !onlyKind || CORE.has(n.kind) || n.kind === onlyKind);

      const ids = new Set(nodes.map((n) => n.id));
      const links: CanvasLink[] = data.edges
        .filter((e) => ids.has(e.source) && ids.has(e.target))
        .map((e, idx) => ({
          id: e.id || `l-${idx}`,
          source: e.source,
          target: e.target,
          label: e.label || e.type,
          type: e.type,
        }));

      // An unlinked node drifts away from the cluster and makes zoom-to-fit shrink everything else.
      const linked = new Set(links.flatMap((l) => [l.source as string, l.target as string]));
      const connected = links.length > 0 ? nodes.filter((n) => linked.has(n.id)) : nodes;

      return { nodes: connected, links };
    }, [data, onlyKind]);

    const highlight = useMemo(() => {
      const nodes = new Set<string>();
      const links = new Set<string>();
      if (hoverId) {
        nodes.add(hoverId);
        graphData.links.forEach((l) => {
          const s = endId(l.source);
          const t = endId(l.target);
          if (s === hoverId || t === hoverId) {
            links.add(l.id);
            nodes.add(s);
            nodes.add(t);
          }
        });
      }
      return { nodes, links };
    }, [hoverId, graphData]);

    useEffect(() => {
      if (graphData.nodes.length === 0 || !fgRef.current) return;
      fgRef.current.d3Force("charge")?.strength(charge);
      fgRef.current.d3Force("link")?.distance(linkDistance);
      fgRef.current.d3ReheatSimulation();
      const timer = setTimeout(fit, 500);
      return () => clearTimeout(timer);
    }, [graphData, charge, linkDistance, fit, width > 0]);

    // Re-centre when the container is resized (fullscreen toggle, panel widening).
    useEffect(() => {
      if (width === 0) return;
      const timer = setTimeout(fit, 250);
      return () => clearTimeout(timer);
    }, [width, height, fit]);

    const drawNode = useCallback(
      (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const lit = highlight.nodes.has(node.id);
        const dimmed = hoverId !== null && !lit;
        const radius = node.radius;

        ctx.save();
        ctx.globalAlpha = dimmed ? 0.18 : 1;

        if (lit) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 4, 0, 2 * Math.PI);
          ctx.fillStyle = `${accent}55`;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = node.color;
        ctx.fill();
        ctx.lineWidth = 2 / Math.max(globalScale, 1);
        ctx.strokeStyle = SURFACE_HEX;
        ctx.stroke();

        if (lit || CORE.has(node.kind) || globalScale >= 0.85) {
          const fontSize = Math.max(11 / globalScale, 3);
          ctx.font = `${CORE.has(node.kind) ? 600 : 500} ${fontSize}px ${FONT}`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";

          const text = node.name;
          const textY = node.y + radius + fontSize * 0.5 + 4 / globalScale;
          const textWidth = ctx.measureText(text).width;
          const padX = 4 / globalScale;
          const padY = 2 / globalScale;

          ctx.fillStyle = "rgba(14, 13, 12, 0.86)";
          ctx.beginPath();
          ctx.roundRect(
            node.x - textWidth / 2 - padX,
            textY - fontSize / 2 - padY,
            textWidth + padX * 2,
            fontSize + padY * 2,
            3 / globalScale
          );
          ctx.fill();

          ctx.fillStyle = lit || CORE.has(node.kind) ? INK_HEX : "rgba(237, 233, 227, 0.72)";
          ctx.fillText(text, node.x, textY);
        }

        ctx.restore();
      },
      [highlight, hoverId, accent]
    );

    const drawLinkLabel = useCallback(
      (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const lit = highlight.links.has(link.id);
        if (!lit && (hoverId !== null || globalScale < 1.1)) return;
        const { source, target, label } = link;
        if (!label || typeof source?.x !== "number" || typeof target?.x !== "number") return;

        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;
        const fontSize = Math.max(10 / globalScale, 2.5);

        ctx.save();
        ctx.font = `500 ${fontSize}px ${FONT}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const textWidth = ctx.measureText(label).width;
        const pad = 3 / globalScale;
        ctx.fillStyle = lit ? accent : "rgba(22, 21, 19, 0.92)";
        ctx.beginPath();
        ctx.roundRect(midX - textWidth / 2 - pad, midY - fontSize / 2 - pad / 2, textWidth + pad * 2, fontSize + pad, 3 / globalScale);
        ctx.fill();
        ctx.fillStyle = lit ? SURFACE_HEX : "rgba(237, 233, 227, 0.6)";
        ctx.fillText(label, midX, midY);
        ctx.restore();
      },
      [highlight, hoverId, accent]
    );

    return (
      <div ref={boxRef} className={cn("relative overflow-hidden", className)}>
        {width > 0 && height > 0 && (
          <ForceGraph2D
            ref={fgRef}
            width={width}
            height={height}
            backgroundColor="rgba(0,0,0,0)"
            graphData={graphData}
            nodeCanvasObject={drawNode}
            nodePointerAreaPaint={(node: any, color, ctx) => {
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(node.x, node.y, node.radius + 6, 0, 2 * Math.PI);
              ctx.fill();
            }}
            onNodeClick={(node) => onNodeClick?.(node as CanvasNode)}
            onNodeHover={(node) => setHoverId(node ? (node.id as string) : null)}
            linkCanvasObjectMode={() => "after"}
            linkCanvasObject={drawLinkLabel}
            linkCurvature={0.08}
            linkDirectionalParticles={particles ? 2 : 0}
            linkDirectionalParticleWidth={2}
            linkDirectionalParticleSpeed={0.006}
            linkDirectionalParticleColor={() => accent}
            linkDirectionalArrowLength={4.5}
            linkDirectionalArrowRelPos={0.65}
            linkColor={(link: any) => {
              if (highlight.links.has(link.id)) return accent;
              if (hoverId !== null) return "rgba(237, 233, 227, 0.06)";
              return link.type === "LIKED" ? "rgba(217, 98, 79, 0.5)" : "rgba(237, 233, 227, 0.2)";
            }}
            linkWidth={(link: any) => (highlight.links.has(link.id) ? 2.2 : 1.1)}
            cooldownTicks={120}
            onEngineStop={fit}
            d3VelocityDecay={0.3}
            d3AlphaDecay={0.02}
          />
        )}
      </div>
    );
  }
);
GraphCanvas.displayName = "GraphCanvas";

export const GraphLegend: React.FC<{ kinds: NodeKind[]; className?: string }> = ({ kinds, className }) => (
  <ul className={cn("flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-ink-muted", className)}>
    {kinds.map((kind) => (
      <li key={kind} className="flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NODE_STYLE[kind].color }} />
        {NODE_STYLE[kind].label}
      </li>
    ))}
  </ul>
);

const KIND_ORDER = Object.keys(NODE_STYLE) as NodeKind[];

export function kindsIn(data: SubgraphData | null): NodeKind[] {
  if (!data) return [];
  const present = new Set(data.nodes.map((n) => nodeKind(n.type)));
  return KIND_ORDER.filter((k) => present.has(k));
}
