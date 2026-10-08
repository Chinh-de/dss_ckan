import React, { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { AlertCircle, ArrowRight, Check, Search, UserPlus, X } from "lucide-react";
import { DomainType, UserProfile } from "../types";
import { api } from "../services/api";
import { DOMAIN_META } from "../lib/theme";
import { cn } from "../lib/utils";

interface UserSelectModalProps {
  open: boolean;
  onClose: () => void;
  domain: DomainType;
  currentUserId: number;
  onSelectUser: (userId: number) => void;
}

type ListMode = "showcase" | "all";

export const UserSelectModal: React.FC<UserSelectModalProps> = ({ open, onClose, domain, currentUserId, onSelectUser }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mode, setMode] = useState<ListMode>("showcase");

  const [isCreating, setIsCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const meta = DOMAIN_META[domain];
  const query = searchQuery.trim();
  const numericId = /^\d+$/.test(query) ? parseInt(query, 10) : null;

  useEffect(() => {
    if (!open) return;
    let cancelled = false;

    const fetchUsers = async () => {
      setLoading(true);
      setLoadError(null);
      try {
        if (query) {
          const res = await api.getUsers(1, 25, query, domain);
          if (cancelled) return;
          setUsers(res.users || []);
          setTotal(null);
        } else if (mode === "showcase") {
          const results = await Promise.all(
            meta.showcaseUsers.map((uid) =>
              api.getUserProfile(uid, domain).catch(
                (): UserProfile => ({
                  id: uid,
                  name: `Người dùng #${uid}`,
                  email: "",
                  totalRatings: 0,
                  totalLikes: 0,
                  totalDislikes: 0,
                  topGenres: [],
                })
              )
            )
          );
          if (!cancelled) setUsers(results);
        } else {
          const res = await api.getUsers(1, 30, undefined, domain);
          if (cancelled) return;
          setUsers(res.users || []);
          setTotal(res.total);
        }
      } catch (err: any) {
        if (cancelled) return;
        setUsers([]);
        setLoadError(err.message || "Không tải được danh sách người dùng.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    const timer = setTimeout(fetchUsers, query ? 250 : 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [open, query, mode, domain]);

  useEffect(() => {
    if (open) return;
    setSearchQuery("");
    setIsCreating(false);
    setCreateError(null);
  }, [open]);

  const choose = (id: number) => {
    onSelectUser(id);
    onClose();
  };

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreateError(null);
    try {
      setCreating(true);
      const created = await api.createUser(newName.trim() || undefined, newEmail.trim() || undefined, domain);
      setNewName("");
      setNewEmail("");
      choose(created.id);
    } catch (err: any) {
      setCreateError(err.message || "Không tạo được người dùng.");
    } finally {
      setCreating(false);
    }
  };

  const showDirectId = numericId !== null && numericId > 0 && !users.some((u) => u.id === numericId);

  return (
    <Dialog.Root open={open} onOpenChange={(next) => !next && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 animate-fade-in bg-bg/70 backdrop-blur-sm" />
        <Dialog.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[8vh] z-50 flex max-h-[84vh] w-[calc(100vw-2rem)] max-w-xl -translate-x-1/2 animate-fade-in flex-col rounded-2xl border border-line-strong bg-surface shadow-lift focus:outline-none"
        >
          <header className="flex items-start justify-between gap-4 px-5 pt-5 sm:px-6">
            <div>
              <Dialog.Title className="font-display text-2xl font-semibold tracking-tight">Chọn người dùng</Dialog.Title>
              <p className="mt-1 text-sm text-ink-muted">
                Tập {meta.dataset}. Gợi ý và đồ thị sẽ được tính lại cho hồ sơ bạn chọn.
              </p>
            </div>
            <Dialog.Close className="icon-btn -mr-2 -mt-1" aria-label="Đóng">
              <X className="h-5 w-5" strokeWidth={1.75} />
            </Dialog.Close>
          </header>

          <div className="space-y-3 px-5 pb-3 pt-4 sm:px-6">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-ink-faint" strokeWidth={1.75} />
              <input
                type="search"
                autoFocus
                aria-label="Tìm người dùng theo ID hoặc tên"
                placeholder="Nhập ID hoặc tên, ví dụ 1665"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="field h-10 pl-9 [&::-webkit-search-cancel-button]:hidden"
              />
            </div>

            {!query && (
              <div className="flex items-center justify-between gap-3">
                <div className="seg" role="group" aria-label="Phạm vi danh sách">
                  <button onClick={() => setMode("showcase")} aria-pressed={mode === "showcase"} className="seg-item">
                    Hồ sơ mẫu
                  </button>
                  <button onClick={() => setMode("all")} aria-pressed={mode === "all"} className="seg-item">
                    Tất cả
                  </button>
                </div>
                <button
                  onClick={() => setIsCreating((v) => !v)}
                  aria-expanded={isCreating}
                  className="btn btn-quiet h-8 px-2.5"
                >
                  <UserPlus className="h-4 w-4" strokeWidth={1.75} />
                  Người dùng mới
                </button>
              </div>
            )}

            {isCreating && !query && (
              <form onSubmit={handleCreateUser} className="animate-fade-up rounded-xl border border-line bg-bg p-4">
                <p className="text-sm text-ink-muted">
                  Người dùng mới chưa có tương tác nào: đây là cách nhanh nhất để thử kịch bản khởi động lạnh.
                </p>
                <div className={cn("mt-3 grid gap-3 sm:grid-cols-2", domain !== "movie" && "hidden")}>
                  <label className="block text-[13px] text-ink-muted">
                    Tên (không bắt buộc)
                    <input
                      type="text"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      placeholder="Trần Khánh Linh"
                      className="field mt-1"
                    />
                  </label>
                  <label className="block text-[13px] text-ink-muted">
                    Email (không bắt buộc)
                    <input
                      type="email"
                      value={newEmail}
                      onChange={(e) => setNewEmail(e.target.value)}
                      placeholder="Để trống để tự sinh"
                      className="field mt-1"
                    />
                  </label>
                </div>
                {createError && (
                  <p className="mt-3 flex items-start gap-2 text-sm text-neg" role="alert">
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.75} />
                    {createError}
                  </p>
                )}
                <div className="mt-4 flex justify-end gap-2">
                  <button type="button" onClick={() => setIsCreating(false)} className="btn btn-quiet">
                    Huỷ
                  </button>
                  <button type="submit" disabled={creating} className="btn btn-primary">
                    {creating ? "Đang tạo…" : "Tạo và dùng ngay"}
                    {!creating && <ArrowRight className="h-4 w-4" strokeWidth={2} />}
                  </button>
                </div>
              </form>
            )}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto border-t border-line px-2 py-2 sm:px-3">
            {showDirectId && (
              <button
                onClick={() => choose(numericId!)}
                className="flex w-full items-center justify-between gap-3 rounded-lg px-3 py-3 text-left transition-colors duration-200 hover:bg-raised"
              >
                <span>
                  Dùng trực tiếp ID <span className="num font-medium text-accent">#{numericId}</span>
                  <span className="block text-[13px] text-ink-faint">ID phải tồn tại trong tập {meta.dataset}.</span>
                </span>
                <ArrowRight className="h-4 w-4 text-ink-faint" strokeWidth={1.75} />
              </button>
            )}

            {loadError && (
              <div className="notice notice-error m-2" role="alert">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-neg" strokeWidth={1.75} />
                <p>{loadError}</p>
              </div>
            )}

            {loading && users.length === 0 && (
              <div className="space-y-2 p-2" aria-busy="true">
                {[0, 1, 2, 3].map((i) => (
                  <div key={i} className="skeleton h-14" />
                ))}
              </div>
            )}

            {!loading && !loadError && users.length === 0 && !showDirectId && (
              <p className="px-3 py-10 text-center text-sm text-ink-muted">
                Không có người dùng nào khớp với “{query}”. Thử một ID số khác.
              </p>
            )}

            <ul className={cn("transition-opacity duration-200", loading && "opacity-50")}>
              {users.map((u) => {
                const isSelected = u.id === currentUserId;
                return (
                  <li key={u.id}>
                    <button
                      onClick={() => choose(u.id)}
                      aria-current={isSelected ? "true" : undefined}
                      className={cn(
                        "flex w-full items-center gap-3.5 rounded-lg px-3 py-2.5 text-left transition-colors duration-200",
                        isSelected ? "bg-accent/10" : "hover:bg-raised"
                      )}
                    >
                      <span
                        className={cn(
                          "num flex h-10 min-w-[3.25rem] items-center justify-center rounded-md px-1.5 text-[13px]",
                          isSelected ? "bg-accent font-medium text-accent-ink" : "bg-raised text-ink-muted"
                        )}
                      >
                        #{u.id}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-medium text-ink">{u.name}</span>
                        <span className="block truncate text-[13px] text-ink-muted">
                          <span className="num">{u.totalLikes}</span> thích
                          {u.totalDislikes > 0 && (
                            <>
                              {" · "}
                              <span className="num">{u.totalDislikes}</span> không thích
                            </>
                          )}
                          {u.topGenres?.length > 0 && ` · ${u.topGenres.slice(0, 3).join(", ")}`}
                          {(u.sampleLikes?.length ?? 0) > 0 && ` · ${u.sampleLikes!.slice(0, 2).join(", ")}`}
                        </span>
                      </span>
                      {isSelected && <Check className="h-4 w-4 shrink-0 text-accent" strokeWidth={2} />}
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>

          {total !== null && !query && (
            <footer className="border-t border-line px-5 py-3 text-[13px] text-ink-faint sm:px-6">
              Đang hiện {users.length} trên <span className="num">{total.toLocaleString("vi-VN")}</span> người dùng. Nhập
              ID để tới thẳng một hồ sơ.
            </footer>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
