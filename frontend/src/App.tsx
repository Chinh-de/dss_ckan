import { useState, useEffect } from "react";
import { Navbar, NavTabType } from "./components/Navbar";
import { RecommendationsView } from "./components/RecommendationsView";
import { KnowledgeGraphView } from "./components/KnowledgeGraphView";
import { ColdStartView } from "./components/ColdStartView";
import { CatalogExploreView } from "./components/CatalogExploreView";
import { UserSelectModal } from "./components/UserSelectModal";
import { ExplainModal } from "./components/ExplainModal";
import { api } from "./services/api";
import { DomainType, DomainInfo } from "./types";
import { DOMAIN_META } from "./lib/theme";

const TABS: NavTabType[] = ["recommendations", "graph", "coldstart", "explore"];

function readStored<T>(key: string, parse: (raw: string) => T | null, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw !== null) {
      const parsed = parse(raw);
      if (parsed !== null) return parsed;
    }
  } catch (e) {
    console.warn(`Could not read ${key} from localStorage`, e);
  }
  return fallback;
}

function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch (e) {
    console.warn(`Could not save ${key} to localStorage`, e);
  }
}

export function App() {
  const [selectedDomain, setSelectedDomain] = useState<DomainType>(() =>
    readStored<DomainType>("ckan_selected_domain", (raw) => (raw in DOMAIN_META ? (raw as DomainType) : null), "movie")
  );

  const [activeTab, setActiveTab] = useState<NavTabType>(() =>
    readStored<NavTabType>(
      "ckan_active_tab",
      (raw) => (TABS.includes(raw as NavTabType) ? (raw as NavTabType) : null),
      "recommendations"
    )
  );

  const [currentUserId, setCurrentUserId] = useState<number>(() =>
    readStored<number>(
      "ckan_current_user_id",
      (raw) => {
        const parsed = parseInt(raw, 10);
        return !isNaN(parsed) && parsed > 0 ? parsed : null;
      },
      1
    )
  );

  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [isUserSelectOpen, setIsUserSelectOpen] = useState(false);
  const [explainingItemId, setExplainingItemId] = useState<number | null>(null);

  useEffect(() => {
    api
      .getDomains()
      .then((res) => {
        if (res && res.domains) setDomains(res.domains);
      })
      .catch((err) => console.error("Error fetching domains:", err));
  }, []);

  // The accent lives on <html> so portalled dialogs pick it up too.
  useEffect(() => {
    document.documentElement.dataset.domain = selectedDomain;
  }, [selectedDomain]);

  const handleSelectUser = (userId: number) => {
    setCurrentUserId(userId);
    store("ckan_current_user_id", userId.toString());
  };

  const handleSelectDomain = (domain: DomainType) => {
    setSelectedDomain(domain);
    store("ckan_selected_domain", domain);
    // User ids are per dataset, so jump to that dataset's demo user.
    handleSelectUser(DOMAIN_META[domain].defaultUser);
  };

  const handleSelectTab = (tab: NavTabType) => {
    setActiveTab(tab);
    store("ckan_active_tab", tab);
  };

  const currentDomainInfo = domains.find((d) => d.id === selectedDomain);

  return (
    <div className="flex min-h-[100dvh] flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-accent focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-accent-ink"
      >
        Bỏ qua điều hướng
      </a>

      <Navbar
        activeTab={activeTab}
        onTabChange={handleSelectTab}
        domain={selectedDomain}
        onDomainChange={handleSelectDomain}
        userId={currentUserId}
        onOpenUsers={() => setIsUserSelectOpen(true)}
      />

      <main id="main" className="mx-auto w-full max-w-[1320px] flex-1 px-4 pb-24 sm:px-6 lg:px-10">
        {activeTab === "recommendations" && (
          <RecommendationsView
            domain={selectedDomain}
            userId={currentUserId}
            info={currentDomainInfo}
            onExplain={setExplainingItemId}
            onOpenUsers={() => setIsUserSelectOpen(true)}
          />
        )}

        {activeTab === "graph" && <KnowledgeGraphView domain={selectedDomain} userId={currentUserId} />}

        {activeTab === "coldstart" && <ColdStartView domain={selectedDomain} />}

        {activeTab === "explore" && <CatalogExploreView domain={selectedDomain} userId={currentUserId} />}
      </main>

      <UserSelectModal
        open={isUserSelectOpen}
        onClose={() => setIsUserSelectOpen(false)}
        domain={selectedDomain}
        currentUserId={currentUserId}
        onSelectUser={handleSelectUser}
      />

      <ExplainModal
        itemId={explainingItemId}
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
