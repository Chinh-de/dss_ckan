import React, { useState, useEffect } from "react";
import { Navbar, NavTabType } from "./components/Navbar";
import { RecommendationsView } from "./components/RecommendationsView";
import { KnowledgeGraphView } from "./components/KnowledgeGraphView";
import { ColdStartView } from "./components/ColdStartView";
import { CatalogExploreView } from "./components/CatalogExploreView";
import { UserHistoryView } from "./components/UserHistoryView";
import { UserSelectModal } from "./components/UserSelectModal";
import { OnboardingModal } from "./components/OnboardingModal";
import { ExplainModal } from "./components/ExplainModal";
import { api } from "./services/api";
import { DomainType, DomainInfo } from "./types";

const DOMAIN_ACCENT_COLORS: Record<DomainType, string> = {
  movie: "#f59e0b", // Amber gold
  book: "#10b981",  // Warm emerald
  music: "#06b6d4", // Sky cyan
};

const DOMAIN_DEFAULT_USERS: Record<DomainType, number> = {
  movie: 1,
  book: 790,
  music: 774,
};

export function App() {
  const [selectedDomain, setSelectedDomain] = useState<DomainType>(() => {
    try {
      const saved = localStorage.getItem("ckan_selected_domain") as DomainType;
      if (saved === "movie" || saved === "book" || saved === "music") {
        return saved;
      }
    } catch (e) {
      console.warn("Could not read selectedDomain from localStorage", e);
    }
    return "movie";
  });

  const [activeTab, setActiveTab] = useState<NavTabType>(() => {
    try {
      const saved = localStorage.getItem("ckan_active_tab") as NavTabType;
      if (saved === "recommendations" || saved === "graph" || saved === "coldstart" || saved === "explore") {
        return saved;
      }
    } catch (e) {
      console.warn("Could not read activeTab from localStorage", e);
    }
    return "recommendations";
  });

  const [currentUserId, setCurrentUserId] = useState<number>(() => {
    try {
      const saved = localStorage.getItem("ckan_current_user_id");
      if (saved) {
        const parsed = parseInt(saved, 10);
        if (!isNaN(parsed) && parsed > 0) return parsed;
      }
    } catch (e) {
      console.warn("Could not read currentUserId from localStorage", e);
    }
    return 1;
  });

  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [isOnboardingOpen, setIsOnboardingOpen] = useState<boolean>(false);
  const [isUserSelectOpen, setIsUserSelectOpen] = useState<boolean>(false);
  const [explainingItemId, setExplainingItemId] = useState<number | null>(null);

  // Fetch available domains info
  useEffect(() => {
    api.getDomains()
      .then((res) => {
        if (res && res.domains) {
          setDomains(res.domains);
        }
      })
      .catch((err) => console.error("Error fetching domains:", err));
  }, []);

  const handleSelectDomain = (domain: DomainType) => {
    setSelectedDomain(domain);
    try {
      localStorage.setItem("ckan_selected_domain", domain);
    } catch (e) {
      console.warn("Could not save selectedDomain to localStorage", e);
    }
    // Switch to optimal demo user for that domain
    const defaultUser = DOMAIN_DEFAULT_USERS[domain] || 1;
    setCurrentUserId(defaultUser);
    try {
      localStorage.setItem("ckan_current_user_id", defaultUser.toString());
    } catch (e) {}
  };

  const handleSelectTab = (tab: NavTabType) => {
    setActiveTab(tab);
    try {
      localStorage.setItem("ckan_active_tab", tab);
    } catch (e) {
      console.warn("Could not save activeTab to localStorage", e);
    }
  };

  const handleSelectUser = (userId: number) => {
    setCurrentUserId(userId);
    try {
      localStorage.setItem("ckan_current_user_id", userId.toString());
    } catch (e) {
      console.warn("Could not save currentUserId to localStorage", e);
    }
  };

  const handleGlobalLike = async (itemId: number) => {
    try {
      await api.submitFeedback(selectedDomain, currentUserId, itemId, "LIKE");
    } catch (e) {
      console.error(e);
    }
  };

  const accentColor = DOMAIN_ACCENT_COLORS[selectedDomain] || "#f59e0b";
  const currentDomainInfo = domains.find((d) => d.id === selectedDomain);

  return (
    <div className="min-h-screen bg-[#090A0F] text-slate-100 flex flex-col font-sans selection:bg-white/10 selection:text-white">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={handleSelectTab}
        selectedDomain={selectedDomain}
        setSelectedDomain={handleSelectDomain}
        domains={domains}
        currentUserId={currentUserId}
        openUserSelect={() => setIsUserSelectOpen(true)}
        accentColor={accentColor}
      />

      {/* Main View Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {activeTab === "recommendations" && (
          <RecommendationsView
            domain={selectedDomain}
            accentColor={accentColor}
            userId={currentUserId}
            currentDomainInfo={currentDomainInfo}
            onExplain={(id) => setExplainingItemId(id)}
            openUserSelect={() => setIsUserSelectOpen(true)}
          />
        )}

        {activeTab === "graph" && (
          <KnowledgeGraphView
            domain={selectedDomain}
            accentColor={accentColor}
            userId={currentUserId}
          />
        )}

        {activeTab === "coldstart" && (
          <ColdStartView
            domain={selectedDomain}
            accentColor={accentColor}
          />
        )}

        {activeTab === "explore" && (
          <CatalogExploreView
            domain={selectedDomain}
            accentColor={accentColor}
            userId={currentUserId}
            onLike={handleGlobalLike}
          />
        )}
      </main>

      {/* User Selection Modal */}
      <UserSelectModal
        isOpen={isUserSelectOpen}
        onClose={() => setIsUserSelectOpen(false)}
        domain={selectedDomain}
        currentUserId={currentUserId}
        onSelectUser={handleSelectUser}
        onUserCreated={(newUid) => {
          handleSelectUser(newUid);
        }}
      />

      {/* Multi-tier Explainability Modal */}
      <ExplainModal
        movieId={explainingItemId}
        domain={selectedDomain}
        userId={currentUserId}
        onClose={() => setExplainingItemId(null)}
        onOpenFullGraph={() => {
          setExplainingItemId(null);
          handleSelectTab("graph");
        }}
      />
    </div>
  );
}

export default App;
