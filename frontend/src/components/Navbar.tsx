import React from "react";
import { ChevronDown } from "lucide-react";
import { DomainType } from "../types";
import { DOMAIN_META, DOMAIN_ORDER } from "../lib/theme";
import { cn } from "../lib/utils";
import { BrandMark, DOMAIN_ICON } from "./ui";

export type NavTabType = "recommendations" | "graph" | "coldstart" | "explore";

interface NavbarProps {
  activeTab: NavTabType;
  onTabChange: (tab: NavTabType) => void;
  domain: DomainType;
  onDomainChange: (domain: DomainType) => void;
  userId: number;
  onOpenUsers: () => void;
}

// The first three tabs are the three problems the project studies, hence the numbering.
const TABS: { id: NavTabType; index?: string; label: string }[] = [
  { id: "recommendations", index: "01", label: "Gợi ý Top-K" },
  { id: "graph", index: "02", label: "Đồ thị tri thức" },
  { id: "coldstart", index: "03", label: "Độ thưa dữ liệu" },
  { id: "explore", label: "Thư viện" },
];

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onTabChange,
  domain,
  onDomainChange,
  userId,
  onOpenUsers,
}) => {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1320px] flex-wrap items-center gap-x-6 px-4 sm:px-6 lg:flex-nowrap lg:px-10">
        <div className="flex h-14 shrink-0 items-center gap-2.5">
          <BrandMark className="h-7 w-7" />
          <div className="leading-tight">
            <span className="block font-display text-[17px] font-semibold tracking-tight">CKAN</span>
            <span className="hidden text-xs text-ink-faint xl:block">Gợi ý trên đồ thị tri thức</span>
          </div>
        </div>

        <nav
          aria-label="Điều hướng chính"
          className="order-last -mx-4 flex w-[calc(100%+2rem)] items-stretch gap-1 overflow-x-auto border-t border-line px-4 sm:-mx-6 sm:w-[calc(100%+3rem)] sm:px-6 lg:order-none lg:mx-0 lg:w-auto lg:flex-1 lg:border-t-0 lg:px-0"
        >
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "relative flex h-11 shrink-0 items-center gap-2 px-3 text-sm transition-colors duration-200 lg:h-14",
                  isActive ? "font-medium text-ink" : "text-ink-muted hover:text-ink"
                )}
              >
                {tab.index && (
                  <span className={cn("num text-xs", isActive ? "text-accent" : "text-ink-faint")}>{tab.index}</span>
                )}
                <span>{tab.label}</span>
                <span
                  className={cn(
                    "absolute inset-x-3 bottom-0 h-0.5 origin-left rounded-full bg-accent transition-transform duration-300",
                    isActive ? "scale-x-100" : "scale-x-0"
                  )}
                />
              </button>
            );
          })}
        </nav>

        <div className="ml-auto flex h-14 items-center gap-2">
          <div className="seg" role="group" aria-label="Chọn tập dữ liệu">
            {DOMAIN_ORDER.map((id) => {
              const Icon = DOMAIN_ICON[id];
              const meta = DOMAIN_META[id];
              return (
                <button
                  key={id}
                  onClick={() => onDomainChange(id)}
                  aria-pressed={domain === id}
                  title={`${meta.label} · ${meta.dataset}`}
                  className="seg-item"
                >
                  <Icon className={cn("h-4 w-4", domain === id && "text-accent")} strokeWidth={1.75} />
                  <span className="hidden md:inline">{meta.label}</span>
                </button>
              );
            })}
          </div>

          <button onClick={onOpenUsers} className="btn btn-ghost group px-2.5" title="Đổi người dùng demo">
            <span className="hidden text-ink-muted sm:inline">Người dùng</span>
            <span className="num text-ink">#{userId}</span>
            <ChevronDown className="h-4 w-4 text-ink-faint transition-colors group-hover:text-ink" strokeWidth={1.75} />
          </button>
        </div>
      </div>
    </header>
  );
};
