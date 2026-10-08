import { DomainType } from "../types";

// Hex mirrors of the --accent CSS variable, for canvas / SVG code that cannot read Tailwind classes.
export const DOMAIN_ACCENT: Record<DomainType, string> = {
  movie: "#CC8E26",
  book: "#3FA877",
  music: "#3D9CC9",
};

export const SURFACE_HEX = "#0E0D0C";
export const INK_HEX = "#EDE9E3";
export const BASELINE_HEX = "#6B675F";
// Second chart series. Checked against each accent with the dataviz validator: violet collides
// with the music cyan under deuteranopia, so that domain uses coral instead.
export const RIPPLENET_HEX: Record<DomainType, string> = {
  movie: "#9A6FE0",
  book: "#9A6FE0",
  music: "#D9624F",
};

interface DomainMeta {
  label: string;
  dataset: string;
  /** Singular noun for one catalogue item, lower-case. */
  item: string;
  searchHint: string;
  sparsity: string;
  defaultUser: number;
  showcaseUsers: number[];
}

export const DOMAIN_META: Record<DomainType, DomainMeta> = {
  movie: {
    label: "Điện ảnh",
    dataset: "MovieLens-20M",
    item: "phim",
    searchHint: "Tìm tên phim",
    sparsity: "99,71%",
    defaultUser: 1,
    showcaseUsers: [1, 375, 551, 1665, 2244],
  },
  book: {
    label: "Sách",
    dataset: "Book-Crossing",
    item: "cuốn sách",
    searchHint: "Tìm tên sách hoặc tác giả",
    sparsity: "99,97%",
    defaultUser: 790,
    showcaseUsers: [790, 6486, 10029, 12762, 1],
  },
  music: {
    label: "Âm nhạc",
    dataset: "Last.FM",
    item: "nghệ sĩ",
    searchHint: "Tìm nghệ sĩ",
    sparsity: "99,71%",
    defaultUser: 774,
    showcaseUsers: [774, 79, 238, 386, 1],
  },
};

export const DOMAIN_ORDER: DomainType[] = ["movie", "book", "music"];

export type NodeKind =
  | "user"
  | "liked"
  | "recommended"
  | "director"
  | "genre"
  | "actor"
  | "writer"
  | "producer"
  | "entity";

// The backend emits MovieLiked/MovieRecommended from Neo4j and ItemLiked/ItemRecommended/Entity
// from the CKAN explainer; both collapse onto the same kinds here.
export function nodeKind(type: string): NodeKind {
  const t = (type || "").toLowerCase();
  if (t === "user") return "user";
  if (t.endsWith("liked")) return "liked";
  if (t.endsWith("recommended")) return "recommended";
  if (t === "director") return "director";
  if (t === "genre") return "genre";
  if (t === "actor") return "actor";
  if (t === "writer") return "writer";
  if (t === "producer") return "producer";
  return "entity";
}

// Categorical hues checked with the dataviz palette validator against SURFACE_HEX.
// Green/pink sit close under deuteranopia, so every node also carries a text label.
export const NODE_STYLE: Record<NodeKind, { color: string; radius: number; label: string }> = {
  user: { color: INK_HEX, radius: 8.5, label: "Người dùng" },
  liked: { color: "#D9624F", radius: 7, label: "Đã thích" },
  recommended: { color: "#3F8FE0", radius: 7.5, label: "Được gợi ý" },
  director: { color: "#9A6FE0", radius: 5.5, label: "Đạo diễn" },
  genre: { color: "#BF8718", radius: 5.5, label: "Thể loại" },
  actor: { color: "#2FA36B", radius: 5.5, label: "Diễn viên" },
  writer: { color: "#D45FA8", radius: 5.5, label: "Tác giả / biên kịch" },
  producer: { color: "#A39A8C", radius: 5, label: "Nhà sản xuất" },
  entity: { color: "#8A857D", radius: 5.5, label: "Thực thể KG" },
};

export const CORE_KINDS: NodeKind[] = ["user", "liked", "recommended"];
