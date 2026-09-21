import React, { useState, useEffect } from "react";
import { Search, UserCheck, X, Sparkles, ThumbsUp, ThumbsDown, Star, Check, UserPlus, ArrowRight } from "lucide-react";
import { UserProfile } from "../types";
import { api } from "../services/api";

interface UserSelectModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUserId: number;
  onSelectUser: (userId: number) => void;
  onUserCreated?: (newUserId: number) => void;
}

// Curated showcase users with diverse profiles
const SHOWCASE_USER_IDS = [1, 2, 10, 13, 39, 375, 551, 1665, 2244, 2431];

export const UserSelectModal: React.FC<UserSelectModalProps> = ({
  isOpen,
  onClose,
  currentUserId,
  onSelectUser,
  onUserCreated,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(false);
  const [filterMode, setFilterMode] = useState<"showcase" | "all">("showcase");

  // Create User Form State
  const [isCreating, setIsCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [creatingLoading, setCreatingLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) return;

    const fetchUsers = async () => {
      setLoading(true);
      try {
        if (searchQuery.trim()) {
          const res = await api.getUsers(1, 25, searchQuery.trim());
          setUsers(res.users || []);
        } else if (filterMode === "showcase") {
          // Fetch showcase active users
          const promises = SHOWCASE_USER_IDS.map((uid) =>
            api.getUserProfile(uid).catch(() => null)
          );
          const results = await Promise.all(promises);
          setUsers(results.filter((u): u is UserProfile => u !== null));
        } else {
          const res = await api.getUsers(1, 30);
          setUsers(res.users || []);
        }
      } catch (err) {
        console.error("Failed to load users:", err);
      } finally {
        setLoading(false);
      }
    };

    const timer = setTimeout(fetchUsers, searchQuery ? 250 : 0);
    return () => clearTimeout(timer);
  }, [isOpen, searchQuery, filterMode]);

  if (!isOpen) return null;

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setCreatingLoading(true);
      const created = await api.createUser(
        newName.trim() || undefined,
        newEmail.trim() || undefined
      );

      // Reset form
      setNewName("");
      setNewEmail("");
      setIsCreating(false);

      // Select new user and trigger callback
      onSelectUser(created.id);
      if (onUserCreated) {
        onUserCreated(created.id);
      }
      onClose();
    } catch (err: any) {
      alert(`Không thể tạo người dùng: ${err.message}`);
    } finally {
      setCreatingLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-[#0F121A] border border-white/15 rounded-3xl p-6 shadow-2xl shadow-black/95 max-h-[90vh] flex flex-col text-white">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 border-b border-white/10 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shadow-glow-amber">
              <UserCheck className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold">Chọn Hồ Sơ Người Dùng (User Profile)</h3>
              <p className="text-xs text-slate-400 font-mono">
                Chọn người dùng hoặc tạo User mới để thử nghiệm Cold-Start
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsCreating(!isCreating)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all active:scale-95 shadow-md ${
                isCreating
                  ? "bg-slate-800 text-slate-200 border border-white/10"
                  : "bg-primary text-black hover:bg-primary/90 shadow-glow-amber"
              }`}
            >
              <UserPlus className="w-3.5 h-3.5" />
              <span>{isCreating ? "Hủy tạo" : "Thêm User mới"}</span>
            </button>

            <button
              onClick={onClose}
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Inline Create User Form */}
        {isCreating && (
          <form
            onSubmit={handleCreateUser}
            className="my-3 p-4 rounded-2xl bg-gradient-to-r from-primary/10 via-slate-900 to-slate-900 border border-primary/30 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200 shrink-0"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-primary uppercase tracking-wider font-mono flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Tạo Người Dùng Mới (Cold-Start)</span>
              </span>
              <span className="text-[10px] text-slate-400 font-mono">ID sẽ được cấp tự động</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <div>
                <label className="block text-[11px] text-slate-300 mb-1">Tên người dùng (Tùy chọn):</label>
                <input
                  type="text"
                  placeholder="Ví dụ: Cinephile Linh..."
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full bg-slate-950 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-primary"
                  autoFocus
                />
              </div>
              <div>
                <label className="block text-[11px] text-slate-300 mb-1">Email (Tùy chọn):</label>
                <input
                  type="email"
                  placeholder="Để trống để tự động tạo..."
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  className="w-full bg-slate-950 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-primary"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={() => setIsCreating(false)}
                className="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-xs text-slate-300"
              >
                Hủy
              </button>
              <button
                type="submit"
                disabled={creatingLoading}
                className="px-4 py-1.5 rounded-xl bg-primary text-black font-bold text-xs hover:brightness-110 active:scale-95 transition-all flex items-center gap-1.5 shadow-glow-amber disabled:opacity-50"
              >
                {creatingLoading ? (
                  <span>Đang khởi tạo...</span>
                ) : (
                  <>
                    <span>Tạo & Đăng nhập</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>
          </form>
        )}

        {/* Search & Filter Controls */}
        <div className="pt-3 pb-2 space-y-2.5 shrink-0">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Tìm theo User ID (ví dụ: 1, 39, 1665...) hoặc tên..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-900/90 border border-white/10 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-primary/50 font-mono"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-white"
              >
                ✕
              </button>
            )}
          </div>

          {!searchQuery && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setFilterMode("showcase")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  filterMode === "showcase"
                    ? "bg-primary text-black font-bold shadow-sm"
                    : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                ⭐ Người dùng tiêu biểu (Showcase)
              </button>
              <button
                onClick={() => setFilterMode("all")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  filterMode === "all"
                    ? "bg-primary text-black font-bold shadow-sm"
                    : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                📋 Tất cả người dùng (Từ ID 0)
              </button>
            </div>
          )}
        </div>

        {/* User List */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 my-2">
          {loading ? (
            <div className="py-16 flex flex-col items-center justify-center text-center">
              <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin mb-3" />
              <p className="text-xs text-slate-400 font-mono">Đang tải danh sách người dùng...</p>
            </div>
          ) : users.length === 0 ? (
            <div className="py-16 text-center text-slate-400">
              <p className="text-sm font-semibold">Không tìm thấy người dùng phù hợp</p>
              <p className="text-xs text-slate-500 mt-1 font-mono">Thử tìm kiếm với số ID khác (0 đến 2500)</p>
            </div>
          ) : (
            users.map((u) => {
              const isSelected = u.id === currentUserId;
              return (
                <div
                  key={u.id}
                  onClick={() => {
                    onSelectUser(u.id);
                    onClose();
                  }}
                  className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-center justify-between gap-3 group ${
                    isSelected
                      ? "bg-primary/10 border-primary shadow-glow-amber"
                      : "bg-slate-900/60 border-white/5 hover:border-primary/40 hover:bg-slate-800/60"
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    {/* User Avatar Badge */}
                    <div
                      className={`w-11 h-11 rounded-2xl flex items-center justify-center font-mono font-bold text-sm shrink-0 border ${
                        isSelected
                          ? "bg-primary text-black border-primary shadow-md"
                          : "bg-white/5 text-primary border-white/10 group-hover:border-primary/40"
                      }`}
                    >
                      #{u.id}
                    </div>

                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="text-sm font-bold text-white group-hover:text-primary transition-colors truncate">
                          {u.name}
                        </h4>
                        {isSelected && (
                          <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary border border-primary/30 text-[10px] font-mono font-bold shrink-0 flex items-center gap-1">
                            <Check className="w-3 h-3" />
                            Đang chọn
                          </span>
                        )}
                      </div>

                      {/* Stats & Genres */}
                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400 mt-1">
                        <span className="font-mono text-slate-300">
                          {u.totalRatings} lượt đánh giá
                        </span>
                        <span>•</span>
                        <span className="text-emerald-400 flex items-center gap-0.5 font-mono">
                          <ThumbsUp className="w-3 h-3" /> {u.totalLikes}
                        </span>
                        <span>•</span>
                        <span className="text-red-400 flex items-center gap-0.5 font-mono">
                          <ThumbsDown className="w-3 h-3" /> {u.totalDislikes}
                        </span>
                      </div>

                      {/* Top genres badges */}
                      {u.topGenres && u.topGenres.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1.5">
                          {u.topGenres.map((g) => (
                            <span
                              key={g}
                              className="px-1.5 py-0.5 rounded bg-white/5 border border-white/5 text-[10px] font-mono text-slate-300"
                            >
                              {g}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectUser(u.id);
                      onClose();
                    }}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold shrink-0 transition-all ${
                      isSelected
                        ? "bg-primary text-black font-bold"
                        : "bg-white/5 text-slate-300 group-hover:bg-primary group-hover:text-black font-medium"
                    }`}
                  >
                    {isSelected ? "Đang chọn" : "Chọn User"}
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Modal Footer */}
        <div className="pt-3 border-t border-white/10 flex items-center justify-between text-xs text-slate-400 shrink-0">
          <span className="font-mono">Tổng cộng: 2,501 người dùng trong hệ thống</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-slate-200 text-xs font-medium transition-colors"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};
