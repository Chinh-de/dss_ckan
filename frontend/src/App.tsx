import React, { useState } from "react";
import { Navbar } from "./components/Navbar";
import { RecommendationsView } from "./components/RecommendationsView";
import { MovieExploreView } from "./components/MovieExploreView";
import { KnowledgeGraphView } from "./components/KnowledgeGraphView";
import { UserHistoryView } from "./components/UserHistoryView";
import { UserSelectModal } from "./components/UserSelectModal";
import { OnboardingModal } from "./components/OnboardingModal";
import { ExplainModal } from "./components/ExplainModal";
import { api } from "./services/api";

export function App() {
  const [activeTab, setActiveTab] = useState<"recommendations" | "explore" | "graph" | "history">(() => {
    try {
      const saved = localStorage.getItem("cinegraph_active_tab");
      if (saved === "recommendations" || saved === "explore" || saved === "graph" || saved === "history") {
        return saved;
      }
    } catch (e) {
      console.warn("Could not read activeTab from localStorage", e);
    }
    return "recommendations";
  });

  const [currentUserId, setCurrentUserId] = useState<number>(() => {
    try {
      const saved = localStorage.getItem("cinegraph_current_user_id");
      if (saved) {
        const parsed = parseInt(saved, 10);
        if (!isNaN(parsed) && parsed > 0) return parsed;
      }
    } catch (e) {
      console.warn("Could not read currentUserId from localStorage", e);
    }
    return 1;
  });

  const [isOnboardingOpen, setIsOnboardingOpen] = useState<boolean>(false);
  const [isUserSelectOpen, setIsUserSelectOpen] = useState<boolean>(false);
  const [explainingMovieId, setExplainingMovieId] = useState<number | null>(null);

  // Sync to localStorage
  const handleSelectTab = (tab: "recommendations" | "explore" | "graph" | "history") => {
    setActiveTab(tab);
    try {
      localStorage.setItem("cinegraph_active_tab", tab);
    } catch (e) {
      console.warn("Could not save activeTab to localStorage", e);
    }
  };

  const handleSelectUser = (userId: number) => {
    setCurrentUserId(userId);
    try {
      localStorage.setItem("cinegraph_current_user_id", userId.toString());
    } catch (e) {
      console.warn("Could not save currentUserId to localStorage", e);
    }
  };

  const handleGlobalLike = async (movieId: number) => {
    try {
      await api.rateMovie(currentUserId, movieId, 5.0);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans selection:bg-primary/20 selection:text-primary">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={handleSelectTab}
        currentUserId={currentUserId}
        openUserSelect={() => setIsUserSelectOpen(true)}
        openOnboarding={() => setIsOnboardingOpen(true)}
      />

      {/* Main View Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        {activeTab === "recommendations" && (
          <RecommendationsView
            userId={currentUserId}
            onExplain={(id) => setExplainingMovieId(id)}
            openOnboarding={() => setIsOnboardingOpen(true)}
          />
        )}

        {activeTab === "explore" && (
          <MovieExploreView
            userId={currentUserId}
            onLike={handleGlobalLike}
          />
        )}

        {activeTab === "graph" && (
          <KnowledgeGraphView userId={currentUserId} />
        )}

        {activeTab === "history" && (
          <UserHistoryView
            userId={currentUserId}
            onExplain={(id) => setExplainingMovieId(id)}
            onChangeUser={() => setIsUserSelectOpen(true)}
            onOpenOnboarding={() => setIsOnboardingOpen(true)}
          />
        )}
      </main>

      {/* User Selection Modal (All 2,501+ users) */}
      <UserSelectModal
        isOpen={isUserSelectOpen}
        onClose={() => setIsUserSelectOpen(false)}
        currentUserId={currentUserId}
        onSelectUser={handleSelectUser}
        onUserCreated={(newUid) => {
          handleSelectUser(newUid);
          setIsOnboardingOpen(true);
        }}
      />

      {/* Onboarding Taste Profile Modal */}
      <OnboardingModal
        isOpen={isOnboardingOpen}
        onClose={() => setIsOnboardingOpen(false)}
        userId={currentUserId}
        onComplete={() => {
          handleSelectTab("recommendations");
        }}
      />

      {/* Multi-tier Explainability Modal */}
      <ExplainModal
        movieId={explainingMovieId}
        userId={currentUserId}
        onClose={() => setExplainingMovieId(null)}
        onOpenFullGraph={() => {
          setExplainingMovieId(null);
          handleSelectTab("graph");
        }}
      />
    </div>
  );
}

export default App;


