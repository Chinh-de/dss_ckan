import React, { useEffect, useState } from "react";
import { Film, BookOpen, Music2, type LucideIcon } from "lucide-react";
import { DomainType } from "../types";
import { cn } from "../lib/utils";

export const DOMAIN_ICON: Record<DomainType, LucideIcon> = {
  movie: Film,
  book: BookOpen,
  music: Music2,
};

export const BrandMark: React.FC<{ className?: string }> = ({ className }) => (
  <svg viewBox="0 0 32 32" fill="none" className={className} aria-hidden="true">
    <path
      d="M10 21 16 9l7 11-13 1Z"
      className="stroke-accent"
      strokeOpacity="0.55"
      strokeWidth="1.5"
      strokeLinejoin="round"
    />
    <circle cx="16" cy="9" r="3" className="fill-accent" />
    <circle cx="10" cy="21" r="2.5" className="fill-ink" />
    <circle cx="23" cy="20" r="2.5" className="fill-ink" />
  </svg>
);

interface PageHeaderProps {
  eyebrow: string;
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  children?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({ eyebrow, title, description, actions, children }) => (
  <header className="animate-fade-up border-b border-line pb-7 pt-8 sm:pt-12">
    <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
      <div className="min-w-0">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="mt-2 max-w-3xl text-3xl font-semibold leading-[1.1] tracking-[-0.025em] sm:text-[2.6rem]">
          {title}
        </h1>
        {description && <p className="mt-3 max-w-[64ch] text-ink-muted">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
    {children}
  </header>
);

export const StatRow: React.FC<{ stats: { label: string; value?: number | string }[] }> = ({ stats }) => (
  <dl className="mt-7 grid grid-cols-2 gap-x-8 gap-y-4 sm:flex sm:flex-wrap sm:gap-x-0">
    {stats.map((s, i) => (
      <div key={s.label} className={cn("sm:pr-8", i > 0 && "sm:border-l sm:border-line sm:pl-8")}>
        <dt className="text-[13px] text-ink-faint">{s.label}</dt>
        <dd className="num mt-0.5 text-xl text-ink">
          {s.value === undefined ? "–" : typeof s.value === "number" ? s.value.toLocaleString("vi-VN") : s.value}
        </dd>
      </div>
    ))}
  </dl>
);

interface PosterProps {
  src?: string | null;
  title: string;
  domain: DomainType;
  className?: string;
  imgClassName?: string;
  eager?: boolean;
}

/** Cover image with a typographic placeholder when the source is missing or fails to load. */
export const Poster: React.FC<PosterProps> = ({ src, title, domain, className, imgClassName, eager }) => {
  const url = typeof src === "string" && src.trim().startsWith("http") ? src.trim().replace(/^http:\/\//, "https://") : null;
  const [failed, setFailed] = useState(false);

  useEffect(() => setFailed(false), [url]);

  const Icon = DOMAIN_ICON[domain];
  const initial = (title || "?").trim().charAt(0).toUpperCase();

  return (
    <div className={cn("relative overflow-hidden bg-raised", className)}>
      {url && !failed ? (
        <img
          src={url}
          alt={title}
          loading={eager ? "eager" : "lazy"}
          onError={() => setFailed(true)}
          // Amazon answers a missing cover with a 1x1 GIF instead of an error.
          onLoad={(e) => e.currentTarget.naturalWidth < 10 && setFailed(true)}
          className={cn("h-full w-full object-cover", imgClassName)}
        />
      ) : (
        <div
          role="img"
          aria-label={title}
          className="flex h-full w-full flex-col items-center justify-center gap-2 bg-gradient-to-b from-raised to-surface"
        >
          <span className="font-display text-5xl font-semibold leading-none text-accent/45">{initial}</span>
          <Icon className="h-4 w-4 text-ink-faint" strokeWidth={1.75} />
        </div>
      )}
    </div>
  );
};

export const posterAspect = (domain: DomainType) => (domain === "music" ? "aspect-square" : "aspect-[2/3]");

export const ScoreMeter: React.FC<{ score: number; className?: string }> = ({ score, className }) => {
  const pct = Math.max(0, Math.min(100, score * 100));
  return (
    <div className={cn("flex items-center gap-2.5", className)} title="Điểm dự đoán của CKAN">
      <div className="h-1 flex-1 overflow-hidden rounded-full bg-line-strong">
        <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${pct}%` }} />
      </div>
      <span className="num text-xs text-ink-muted">{pct.toFixed(1)}%</span>
    </div>
  );
};

interface EmptyStateProps {
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title, children, action }) => (
  <div className="rounded-xl border border-dashed border-line-strong px-6 py-14 text-center">
    <h3 className="text-lg font-semibold">{title}</h3>
    {children && <p className="mx-auto mt-1.5 max-w-md text-sm text-ink-muted">{children}</p>}
    {action && <div className="mt-5 flex justify-center">{action}</div>}
  </div>
);
