import React, { useState } from "react";
import { SURFACE_HEX } from "../lib/theme";
import { useElementSize } from "../lib/useElementSize";

export interface ChartSeries {
  key: string;
  label: string;
  /** Longer name for the legend. */
  legend: string;
  color: string;
  /** SVG dash pattern; identity never rests on colour alone. */
  dash?: string;
  values: number[];
}

interface TrajectoryChartProps {
  series: ChartSeries[];
  /** One tick label per value, e.g. "10%". */
  xLabels: string[];
  xTitle: string;
  selectedIndex: number;
  onSelect: (index: number) => void;
  ariaLabel: string;
}

const HEIGHT = 300;
const M = { top: 16, right: 84, bottom: 46, left: 44 };
const LABEL_GAP = 15;

/** Line chart on one shared y-axis. Columns are ordinal and evenly spaced. */
export const TrajectoryChart: React.FC<TrajectoryChartProps> = ({
  series,
  xLabels,
  xTitle,
  selectedIndex,
  onSelect,
  ariaLabel,
}) => {
  const [ref, { width }] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const all = series.flatMap((s) => s.values);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const step = hi - lo > 0.4 ? 0.2 : hi - lo > 0.2 ? 0.1 : hi - lo > 0.08 ? 0.05 : 0.02;
  const y0 = Math.floor(lo / step) * step;
  const y1 = Math.max(Math.ceil(hi / step) * step, y0 + step);
  const ticks: number[] = [];
  for (let t = y0; t <= y1 + 1e-9; t += step) ticks.push(Number(t.toFixed(2)));

  const n = xLabels.length;
  const innerW = Math.max(0, width - M.left - M.right);
  const innerH = HEIGHT - M.top - M.bottom;
  const x = (i: number) => M.left + (n > 1 ? (i / (n - 1)) * innerW : innerW / 2);
  const y = (v: number) => M.top + (1 - (v - y0) / (y1 - y0)) * innerH;
  const path = (values: number[]) => values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" ");
  const band = n > 1 ? innerW / (n - 1) : innerW;

  // End-of-line labels, pushed apart when the lines finish close together.
  const labels = series
    .map((s) => ({ s, y: y(s.values[n - 1]) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) {
    if (labels[i].y - labels[i - 1].y < LABEL_GAP) labels[i].y = labels[i - 1].y + LABEL_GAP;
  }
  const overflow = labels.length ? labels[labels.length - 1].y - (M.top + innerH) : 0;
  if (overflow > 0) labels.forEach((l) => (l.y -= overflow));

  const active = hover ?? selectedIndex;

  return (
    <div>
      <ul className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[13px] text-ink-muted">
        {series.map((s) => (
          <li key={s.key} className="flex items-center gap-2">
            <svg width="22" height="6" aria-hidden="true">
              <line x1="0" y1="3" x2="22" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} />
            </svg>
            {s.legend}
          </li>
        ))}
      </ul>

      <div ref={ref} className="relative" onPointerLeave={() => setHover(null)}>
        {width > 0 && (
          <svg width={width} height={HEIGHT} role="img" aria-label={ariaLabel}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={M.left} x2={M.left + innerW} y1={y(t)} y2={y(t)} className="stroke-line" />
                <text x={M.left - 10} y={y(t)} textAnchor="end" dominantBaseline="middle" className="num fill-ink-faint text-xs">
                  {t.toFixed(2)}
                </text>
              </g>
            ))}

            <line
              x1={x(active)}
              x2={x(active)}
              y1={M.top}
              y2={M.top + innerH}
              className="stroke-line-strong"
              strokeDasharray={hover === null ? "2 3" : undefined}
            />

            {series.map((s) => (
              <path
                key={s.key}
                d={path(s.values)}
                fill="none"
                stroke={s.color}
                strokeWidth="2"
                strokeDasharray={s.dash}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}

            {series.map((s) =>
              s.values.map((v, i) => (
                <circle
                  key={`${s.key}-${i}`}
                  cx={x(i)}
                  cy={y(v)}
                  r={i === active ? 5 : 4}
                  fill={s.color}
                  stroke={SURFACE_HEX}
                  strokeWidth="2"
                />
              ))
            )}

            {xLabels.map((label, i) => (
              <text
                key={label}
                x={x(i)}
                y={HEIGHT - M.bottom + 20}
                textAnchor="middle"
                className={`num text-xs ${i === selectedIndex ? "fill-ink font-medium" : "fill-ink-faint"}`}
              >
                {label}
              </text>
            ))}

            <text x={M.left + innerW / 2} y={HEIGHT - 4} textAnchor="middle" className="fill-ink-faint text-xs">
              {xTitle}
            </text>

            {labels.map(({ s, y: labelY }) => (
              <text key={s.key} x={x(n - 1) + 12} y={labelY} dominantBaseline="middle" className="fill-ink-muted text-xs">
                {s.label}
              </text>
            ))}

            {/* Full-height hit bands: the pointer only has to be nearest to a column. */}
            {xLabels.map((label, i) => (
              <rect
                key={label}
                x={x(i) - band / 2}
                y={M.top}
                width={band}
                height={innerH + 28}
                fill="transparent"
                tabIndex={0}
                role="button"
                aria-label={`${label}: ${series.map((s) => `${s.label} ${s.values[i].toFixed(4)}`).join(", ")}`}
                className="cursor-pointer outline-none"
                onPointerEnter={() => setHover(i)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
                onClick={() => onSelect(i)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(i);
                  }
                }}
              />
            ))}
          </svg>
        )}

        {hover !== null && width > 0 && (
          <div
            className="pointer-events-none absolute top-0 z-10 w-48 rounded-lg border border-line-strong bg-raised px-3 py-2 text-[13px] shadow-lift"
            style={{ left: Math.min(Math.max(x(hover) - 96, 0), width - 192) }}
          >
            <p className="text-ink-faint">
              {xLabels[hover]} {xTitle.toLowerCase()}
            </p>
            {[...series]
              .sort((a, b) => b.values[hover] - a.values[hover])
              .map((s) => (
                <p key={s.key} className="mt-0.5 flex items-center justify-between gap-3">
                  <span className="flex items-center gap-2 text-ink-muted">
                    <span className="h-0.5 w-3" style={{ background: s.color }} />
                    {s.label}
                  </span>
                  <span className="num font-medium text-ink">{s.values[hover].toFixed(4)}</span>
                </p>
              ))}
          </div>
        )}
      </div>
    </div>
  );
};
