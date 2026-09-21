import React from "react";
import { Film, Compass, Share2, Sparkles, UserCheck, History, ChevronDown } from "lucide-react";

interface NavbarProps {
  activeTab: "recommendations" | "explore" | "graph" | "history";
  setActiveTab: (tab: "recommendations" | "explore" | "graph" | "history") => void;
  currentUserId: number;
  openUserSelect: () => void;
  openOnboarding: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  currentUserId,
  openUserSelect,
  openOnboarding,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/5 bg-background/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary/20 via-primary/10 to-accent/10 border border-primary/30 flex items-center justify-center shadow-glow-gold">
            <Film className="w-5 h-5 text-primary" />
          </div>
          <div>
            <span className="font-extrabold text-lg tracking-tight text-white flex items-center gap-1.5">
              DSS Demo 
            </span>
            <p className="text-[11px] text-slate-400 -mt-0.5 hidden sm:block">
              Hệ thống Đề xuất Phim Thông minh Dựa trên Đồ thị Tri thức (CKAN)
            </p>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav className="flex items-center gap-1 p-1 rounded-xl bg-slate-900/60 border border-white/5">
          <button
            onClick={() => setActiveTab("recommendations")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "recommendations"
                ? "bg-primary text-black shadow-sm font-bold"
                : "text-slate-300 hover:text-white hover:bg-white/5"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Gợi ý</span>
          </button>

          <button
            onClick={() => setActiveTab("explore")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "explore"
                ? "bg-primary text-black shadow-sm font-bold"
                : "text-slate-300 hover:text-white hover:bg-white/5"
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Thư viện</span>
          </button>

          <button
            onClick={() => setActiveTab("graph")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "graph"
                ? "bg-primary text-black shadow-sm font-bold"
                : "text-slate-300 hover:text-white hover:bg-white/5"
            }`}
          >
            <Share2 className="w-3.5 h-3.5" />
            <span>Đồ thị tri thức</span>
          </button>

          <button
            onClick={() => setActiveTab("history")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "history"
                ? "bg-primary text-black shadow-sm font-bold"
                : "text-slate-300 hover:text-white hover:bg-white/5"
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span>Lịch sử</span>
          </button>
        </nav>

        {/* User context & Onboarding */}
        <div className="flex items-center gap-3">
          {/* Onboarding launcher button */}
          <button
            onClick={openOnboarding}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/10 border border-white/10 text-slate-200 transition-all active:scale-95"
          >
            <Sparkles className="w-3.5 h-3.5 text-primary" />
            <span className="hidden md:inline">Gu sở thích</span>
          </button>

          {/* User selector button */}
          <div className="flex items-center gap-2 pl-2 border-l border-white/10">
            <button
              onClick={openUserSelect}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-white/10 hover:border-primary/40 text-xs text-slate-200 transition-all active:scale-95 group shadow-sm"
              title="Nhấn để chọn người dùng"
            >
              <div className="w-5 h-5 rounded-md bg-primary/20 text-primary border border-primary/30 flex items-center justify-center font-mono font-bold text-[10px]">
                #{currentUserId}
              </div>
              <span className="font-medium truncate max-w-[120px] hidden sm:inline">
                User #{currentUserId}
              </span>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 group-hover:text-primary transition-colors" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};


