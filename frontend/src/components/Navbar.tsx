import React from "react";
import {
  Film,
  BookOpen,
  Music2,
  Sparkles,
  Share2,
  Sliders,
  Compass,
  ChevronDown,
  Layers,
} from "lucide-react";
import { DomainType, DomainInfo } from "../types";

export type NavTabType = "recommendations" | "graph" | "coldstart" | "explore";

interface NavbarProps {
  activeTab: NavTabType;
  setActiveTab: (tab: NavTabType) => void;
  selectedDomain: DomainType;
  setSelectedDomain: (domain: DomainType) => void;
  domains: DomainInfo[];
  currentUserId: number;
  openUserSelect: () => void;
  accentColor: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  selectedDomain,
  setSelectedDomain,
  domains,
  currentUserId,
  openUserSelect,
  accentColor,
}) => {
  const domainIcons: Record<DomainType, React.ReactNode> = {
    movie: <Film className="w-3.5 h-3.5" />,
    book: <BookOpen className="w-3.5 h-3.5" />,
    music: <Music2 className="w-3.5 h-3.5" />,
  };

  const domainOptions: { id: DomainType; label: string; sub: string }[] = [
    { id: "movie", label: "Điện ảnh", sub: "MovieLens" },
    { id: "book", label: "Sách", sub: "Book-Crossing" },
    { id: "music", label: "Âm nhạc", sub: "Last.FM" },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-[#090A0F]/90 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3 shrink-0">
          <div
            className="w-9 h-9 rounded-xl border flex items-center justify-center shadow-sm"
            style={{
              borderColor: `${accentColor}40`,
              backgroundColor: `${accentColor}15`,
              color: accentColor,
            }}
          >
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-sm tracking-tight text-white font-mono">
                CKAN • MULTI-DOMAIN DEMO
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-white/5 border border-white/10 text-slate-400">
                v2.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400 -mt-0.5 hidden md:block">
              Hệ thống Gợi ý Đa miền Dựa trên Đồ thị Tri thức (Freebase KG)
            </p>
          </div>
        </div>

        {/* Center: Domain Switcher (3 Datasets) */}
        <div className="flex items-center p-1 rounded-xl bg-[#10121A] border border-white/10">
          {domainOptions.map((opt) => {
            const isSelected = selectedDomain === opt.id;
            return (
              <button
                key={opt.id}
                onClick={() => setSelectedDomain(opt.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                  isSelected
                    ? "bg-white/15 text-white font-bold shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
                }`}
                style={
                  isSelected
                    ? {
                        color: accentColor,
                        borderColor: `${accentColor}50`,
                      }
                    : {}
                }
              >
                {domainIcons[opt.id]}
                <span className="hidden sm:inline font-sans font-medium">{opt.label}</span>
                <span className="text-[10px] text-slate-400 hidden lg:inline">({opt.sub})</span>
              </button>
            );
          })}
        </div>

        {/* Navigation Tabs: 3 Problems + Catalog */}
        <nav className="hidden xl:flex items-center gap-1 p-1 rounded-xl bg-[#10121A] border border-white/10">
          <button
            onClick={() => setActiveTab("recommendations")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "recommendations"
                ? "bg-white/15 text-white font-bold"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>1. Gợi ý Top-K</span>
          </button>

          <button
            onClick={() => setActiveTab("graph")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "graph"
                ? "bg-white/15 text-white font-bold"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <Share2 className="w-3.5 h-3.5" />
            <span>2. Đường dẫn KG</span>
          </button>

          <button
            onClick={() => setActiveTab("coldstart")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "coldstart"
                ? "bg-white/15 text-white font-bold"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>3. Thử nghiệm độ thưa</span>
          </button>

          <button
            onClick={() => setActiveTab("explore")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "explore"
                ? "bg-white/15 text-white font-bold"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Thư viện</span>
          </button>
        </nav>

        {/* User Switcher */}
        <div className="flex items-center gap-2">
          <button
            onClick={openUserSelect}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#10121A] hover:bg-slate-800/80 border border-white/10 hover:border-white/20 text-xs text-slate-200 transition-all active:scale-95 group shadow-sm font-mono"
            title="Nhấn để đổi người dùng demo"
          >
            <div
              className="w-5 h-5 rounded-md flex items-center justify-center font-mono font-bold text-[10px] border"
              style={{
                color: accentColor,
                borderColor: `${accentColor}50`,
                backgroundColor: `${accentColor}15`,
              }}
            >
              #{currentUserId}
            </div>
            <span className="font-medium hidden sm:inline text-xs">User #{currentUserId}</span>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 group-hover:text-white transition-colors" />
          </button>
        </div>
      </div>

      {/* Mobile / Tablet Tab Strip */}
      <div className="xl:hidden flex items-center justify-center gap-1 px-4 py-2 border-t border-white/5 overflow-x-auto bg-[#090A0F]">
        <button
          onClick={() => setActiveTab("recommendations")}
          className={`px-3 py-1 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
            activeTab === "recommendations" ? "bg-white/15 text-white font-bold" : "text-slate-400"
          }`}
        >
          1. Gợi ý Top-K
        </button>
        <button
          onClick={() => setActiveTab("graph")}
          className={`px-3 py-1 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
            activeTab === "graph" ? "bg-white/15 text-white font-bold" : "text-slate-400"
          }`}
        >
          2. Đường dẫn KG
        </button>
        <button
          onClick={() => setActiveTab("coldstart")}
          className={`px-3 py-1 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
            activeTab === "coldstart" ? "bg-white/15 text-white font-bold" : "text-slate-400"
          }`}
        >
          3. Thử nghiệm độ thưa
        </button>
        <button
          onClick={() => setActiveTab("explore")}
          className={`px-3 py-1 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
            activeTab === "explore" ? "bg-white/15 text-white font-bold" : "text-slate-400"
          }`}
        >
          Thư viện
        </button>
      </div>
    </header>
  );
};
