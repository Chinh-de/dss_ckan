import os
import json
import pickle
import heapq
import threading
from pathlib import Path
import logging
from typing import List, Dict, Set, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
from sqlalchemy.orm import Session

from app.core.config import settings, BACKEND_DIR
from app.core.neo4j_client import neo4j_client
from app.models.sql_models import Rating, Interaction, Movie
from app.models.schemas import (
    RecommendationItemDto,
    DomainInfoDto,
    ExplanationResponseDto,
    ExplanationPathDto,
    SubgraphResponseDto,
    GraphNodeDto,
    GraphEdgeDto,
    ColdStartSimulationResponseDto,
    SparsityPointDto,
    ModelResultDto,
    ItemDto,
    ItemListResponseDto,
    UserProfileDto,
)
from app.recommendation.ckan_model import CKAN
from app.recommendation.relation_labels import relation_role, relation_node_type
from app.recommendation.dynamic_propagation import (
    build_kg_dict,
    generate_user_triple_set,
)

logger = logging.getLogger("recommendation_engine")

VALID_DOMAINS = ("movie", "book", "music")
# Likes and hides made in the demo UI, kept across restarts for all three domains.
FEEDBACK_PATH = BACKEND_DIR / "data" / "session_feedback.json"
# Measured results copied out of the executed notebook by app/scripts/extract_benchmark_results.py.
BENCHMARK_PATH = BACKEND_DIR / "data" / "benchmark_results.json"
# Per-dataset export written by the benchmark notebook: ckan_model.pt, config.json, item_embeddings.npy.
SAVED_MODELS_DIR = Path(os.getenv("SAVED_MODELS_DIR", str(BACKEND_DIR.parent / "saved_models")))

class ItemNotFoundError(LookupError):
    """Raised when an item id does not exist in the requested domain."""


class ModelArgs:
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, v)

class DomainData:
    def __init__(self, name: str, data_dir: str, checkpoint_path: str):
        self.name = name
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False
        
        self.kg_np: Optional[np.ndarray] = None
        self.kg_dict: Dict[int, List[Tuple[int, int]]] = {}
        self.tail_to_heads: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.ratings_np: Optional[np.ndarray] = None
        
        self.user_history: Dict[int, Set[int]] = defaultdict(set)
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.relations: Dict[str, str] = {}
        self.all_items: List[int] = []
        
        self.n_entity = 0
        self.n_relation = 0
        self.model: Optional[CKAN] = None
        self.model_args = ModelArgs({
            "dim": 64,
            "n_layer": 1,
            "agg": "concat",
            "batch_size": 256,
            "use_cuda": False,
        })
        # In-memory dynamic feedback for live session updates
        self.session_likes: Dict[int, Set[int]] = defaultdict(set)
        self.session_dislikes: Dict[int, Set[int]] = defaultdict(set)
        # Users created from the demo UI; they start with no interactions at all.
        self.session_users: Set[int] = set()
        # How many users liked each item in the dataset (the MostPopular baseline).
        self.popularity: Dict[int, int] = defaultdict(int)
        # Triple-set sizes for the user and item sides (utss / itss in the notebook's CONFIG).
        self.utss = 32
        self.itss = 64
        # User-independent item embeddings, one row per item id. Loaded from the notebook export
        # when available, otherwise filled by RecommendationEngine._ensure_item_matrix.
        self.item_matrix: Optional[torch.Tensor] = None
        self.model_source = "legacy checkpoint"
        # Real names for KG entities, keyed by entity id (see app/scripts/build_entity_names.py).
        self.entity_names: Dict[str, str] = {}

    def load(self, device: torch.device):
        if self.is_loaded:
            return

        logger.info(f"Loading domain '{self.name}' from {self.data_dir}...")
        
        # 1. Load KG
        kg_file = os.path.join(self.data_dir, "kg_final.npy")
        if os.path.exists(kg_file):
            self.kg_np = np.load(kg_file)
            self.kg_dict = build_kg_dict(self.kg_np)
            for h, r, t in self.kg_np:
                self.tail_to_heads[int(t)].append((int(h), int(r)))
            self.n_entity = int(max(np.max(self.kg_np[:, 0]), np.max(self.kg_np[:, 2]))) + 1
            self.n_relation = int(np.max(self.kg_np[:, 1])) + 1
            logger.info(f"[{self.name}] KG loaded: {len(self.kg_dict)} heads, {self.n_entity} entities, {self.n_relation} relations.")
        
        # 2. Load Ratings & precompute user history
        rat_file = os.path.join(self.data_dir, "ratings_final.npy")
        if os.path.exists(rat_file):
            self.ratings_np = np.load(rat_file)
            for u, i, r in self.ratings_np:
                if r == 1:
                    self.user_history[int(u)].add(int(i))
                    self.popularity[int(i)] += 1
            logger.info(f"[{self.name}] Ratings loaded: {len(self.ratings_np)} ratings across {len(self.user_history)} users.")

        # 3. Load Relations
        rel_file = os.path.join(self.data_dir, "kg_relations.json")
        if os.path.exists(rel_file):
            try:
                with open(rel_file, "r", encoding="utf-8") as f:
                    rel_data = json.load(f)
                    self.relations = rel_data.get("index2rel", {})
            except Exception as e:
                logger.warning(f"[{self.name}] Could not parse relations: {e}")

        # 4. Load Metadata
        meta_candidates = [
            os.path.join(self.data_dir, f"{self.name}s_metadata.json"),
            os.path.join(self.data_dir, f"{self.name}_metadata.json"),
            os.path.join(self.data_dir, "movies_metadata.json"),
        ]
        meta_file = next((f for f in meta_candidates if os.path.exists(f)), None)
        if meta_file:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_list = json.load(f)
                    if isinstance(meta_list, list):
                        self.metadata = {int(m.get("ckan_id", idx)): m for idx, m in enumerate(meta_list)}
                    elif isinstance(meta_list, dict):
                        self.metadata = {int(k): v for k, v in meta_list.items()}
                self.all_items = sorted(list(self.metadata.keys()))
                logger.info(f"[{self.name}] Loaded {len(self.metadata)} metadata items.")

                # Enrich movie domain with TMDB poster URLs from movies_and_posters.csv
                if self.name == "movie":
                    posters_csv = os.path.join(self.data_dir, "movies_and_posters.csv")
                    if os.path.exists(posters_csv):
                        try:
                            import csv
                            with open(posters_csv, "r", encoding="utf-8") as pf:
                                rdr = csv.DictReader(pf)
                                p_map = {}
                                for r in rdr:
                                    mid = r.get("movieId")
                                    p_url = r.get("poster_url", "").strip()
                                    if mid and p_url and p_url.startswith("http"):
                                        p_map[int(mid)] = p_url.replace("/original/", "/w500/")
                            # Map to items
                            matched_posters = 0
                            for item_obj in self.metadata.values():
                                ml_id = item_obj.get("movie_id")
                                if ml_id and int(ml_id) in p_map:
                                    item_obj["posterUrl"] = p_map[int(ml_id)]
                                    matched_posters += 1
                            logger.info(f"[{self.name}] Enriched {matched_posters} movies with TMDB posters.")
                        except Exception as pe:
                            logger.warning(f"Could not load posters CSV: {pe}")

                # The Last.FM picture URLs in the dataset are dead; use the Deezer ones when fetched
                # (see app/scripts/fetch_artist_images.py), otherwise leave the item without a picture.
                if self.name == "music":
                    images_file = os.path.join(self.data_dir, "artist_images.json")
                    images = {}
                    if os.path.exists(images_file):
                        try:
                            with open(images_file, "r", encoding="utf-8") as imf:
                                images = json.load(imf)
                        except Exception as ie:
                            logger.warning(f"Could not load artist images: {ie}")
                    for item_id, item_obj in self.metadata.items():
                        item_obj["posterUrl"] = images.get(str(item_id))
                    logger.info(f"[{self.name}] {len(images)} artists have a picture.")

                # Ensure secure HTTPS for all metadata poster URLs
                for item_obj in self.metadata.values():
                    p_url = item_obj.get("posterUrl")
                    if p_url and isinstance(p_url, str) and p_url.startswith("http://"):
                        item_obj["posterUrl"] = p_url.replace("http://", "https://")

            except Exception as e:
                logger.warning(f"[{self.name}] Could not load metadata: {e}")

        names_file = os.path.join(self.data_dir, "entity_names.json")
        if os.path.exists(names_file):
            try:
                with open(names_file, "r", encoding="utf-8") as nf:
                    self.entity_names = json.load(nf)
                logger.info(f"[{self.name}] Loaded {len(self.entity_names)} entity names.")
            except Exception as e:
                logger.warning(f"[{self.name}] Could not load entity names: {e}")

        if not self.all_items and self.ratings_np is not None:
            self.all_items = sorted(list(np.unique(self.ratings_np[:, 1])))

        # 5. Load CKAN Model Checkpoint. The notebook export wins when present, so the demo serves
        # the same weights, depth and triple-set sizes that produced the reported metrics.
        notebook_dir = SAVED_MODELS_DIR / self.name
        precomputed_items = None
        if (notebook_dir / "config.json").exists() and (notebook_dir / "ckan_model.pt").exists():
            with open(notebook_dir / "config.json", "r", encoding="utf-8") as cf:
                cfg = json.load(cf)
            self.model_args = ModelArgs({
                "dim": cfg["dim"],
                "n_layer": cfg["n_layer"],
                "agg": cfg["agg"],
                "batch_size": 256,
                "use_cuda": False,
            })
            self.utss = cfg["utss"]
            self.itss = cfg["itss"]
            self.checkpoint_path = str(notebook_dir / "ckan_model.pt")
            self.model_source = f"notebook export ({notebook_dir.name}: n_layer={cfg['n_layer']}, utss={self.utss})"
            if (notebook_dir / "item_embeddings.npy").exists():
                precomputed_items = np.load(notebook_dir / "item_embeddings.npy")

        if os.path.exists(self.checkpoint_path):
            logger.info(f"[{self.name}] Loading CKAN checkpoint from {self.checkpoint_path}...")
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                # The notebook wraps the attention MLP in a sub-module; the weights are the same.
                state_dict = {k.replace("attention_layer.attention.", "attention."): v for k, v in state_dict.items()}
                self.model = CKAN(self.model_args, self.n_entity, self.n_relation)
                self.model.load_state_dict(state_dict)
                self.model.to(device)
                self.model.eval()
                if precomputed_items is not None:
                    self.item_matrix = torch.from_numpy(precomputed_items).float().to(device)
                logger.info(f"[{self.name}] CKAN model loaded successfully from {self.model_source}.")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to load checkpoint: {e}. Initializing fresh model.")
                self.model = CKAN(self.model_args, max(self.n_entity, 10000), max(self.n_relation, 100))
                self.model.to(device)
                self.model.eval()
        else:
            logger.warning(f"[{self.name}] Checkpoint {self.checkpoint_path} not found. Creating fresh model.")
            self.model = CKAN(self.model_args, max(self.n_entity, 10000), max(self.n_relation, 100))
            self.model.to(device)
            self.model.eval()

        self.is_loaded = True

class RecommendationEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecommendationEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = torch.device("cpu")
        self.domains: Dict[str, DomainData] = {}
        self.is_loaded = False
        self.benchmark: Dict[str, Any] = {}
        self._feedback_lock = threading.Lock()
        self._initialized = True

    def initialize(self):
        if self.is_loaded:
            return

        logger.info("Initializing Multi-Domain Recommendation Engine...")
        use_cuda = settings.USE_CUDA and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"Engine device: {self.device}")

        # Setup 3 domains
        data_base = BACKEND_DIR / "data"
        models_base = BACKEND_DIR / "models"

        # 1. Movie Domain
        movie_ckpt = str(models_base / "ckan_movie.pt")
        if not os.path.exists(movie_ckpt):
            movie_ckpt = str(models_base / "ckan_model.pt")
        self.domains["movie"] = DomainData(
            name="movie",
            data_dir=str(data_base / "movie"),
            checkpoint_path=movie_ckpt
        )

        # 2. Book Domain
        book_ckpt = str(models_base / "ckan_book.pt")
        if not os.path.exists(book_ckpt):
            book_ckpt = str(data_base / "book" / "ckan_checkpoint.pt")
        self.domains["book"] = DomainData(
            name="book",
            data_dir=str(data_base / "book"),
            checkpoint_path=book_ckpt
        )

        # 3. Music Domain
        music_ckpt = str(models_base / "ckan_music.pt")
        if not os.path.exists(music_ckpt):
            music_ckpt = str(data_base / "music" / "ckan_checkpoint.pt")
        self.domains["music"] = DomainData(
            name="music",
            data_dir=str(data_base / "music"),
            checkpoint_path=music_ckpt
        )

        # Initialize all 3 domains
        for dname, domain_obj in self.domains.items():
            try:
                domain_obj.load(self.device)
            except Exception as e:
                logger.error(f"Error loading domain {dname}: {e}")

        for dname, domain_obj in self.domains.items():
            try:
                self._ensure_item_matrix(domain_obj)
            except Exception as e:
                logger.error(f"Could not precompute item embeddings for {dname}: {e}")
        self._load_feedback()
        if BENCHMARK_PATH.exists():
            try:
                with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                    self.benchmark = json.load(f)
            except (OSError, ValueError) as e:
                logger.warning(f"Could not load benchmark results: {e}")
        self.is_loaded = True
        logger.info("Multi-Domain Recommendation Engine initialization complete.")

    def get_domain(self, domain_name: str = "movie") -> DomainData:
        if not self.is_loaded:
            self.initialize()
        domain = self.domains.get((domain_name or "").lower())
        if not domain:
            raise ValueError(f"Unknown domain '{domain_name}'. Expected one of {VALID_DOMAINS}.")
        if not domain.is_loaded:
            domain.load(self.device)
        return domain

    # Backwards-compatible properties for health check
    @property
    def n_entity(self) -> int:
        movie = self.get_domain("movie")
        return movie.n_entity

    @property
    def n_relation(self) -> int:
        movie = self.get_domain("movie")
        return movie.n_relation

    def get_domains_info(self) -> List[DomainInfoDto]:
        if not self.is_loaded:
            self.initialize()

        info = [
            DomainInfoDto(
                id="movie",
                name="MovieLens-20M",
                vietnameseName="Điện ảnh (Movie)",
                itemTerm="Bộ phim",
                description="Tập dữ liệu điện ảnh chuẩn MovieLens kết hợp Freebase Knowledge Graph với các quan hệ đạo diễn, diễn viên, thể loại và hãng sản xuất.",
                itemsCount=len(self.domains["movie"].all_items),
                usersCount=len(self.domains["movie"].user_history),
                triplesCount=len(self.domains["movie"].kg_np) if self.domains["movie"].kg_np is not None else 499474,
                relationsCount=self.domains["movie"].n_relation,
                sampleUsers=[1, 375, 551, 1665, 2244],
                accentColor="#f59e0b"
            ),
            DomainInfoDto(
                id="book",
                name="Book-Crossing",
                vietnameseName="Sách (Book)",
                itemTerm="Cuốn sách",
                description="Tập dữ liệu xuất bản Book-Crossing liên kết thực thể tác giả, năm phát hành, thể loại và nhà xuất bản với độ thưa ma trận lên tới 99.94%.",
                itemsCount=len(self.domains["book"].all_items),
                usersCount=len(self.domains["book"].user_history),
                triplesCount=len(self.domains["book"].kg_np) if self.domains["book"].kg_np is not None else 151500,
                relationsCount=self.domains["book"].n_relation,
                sampleUsers=[790, 6486, 10029, 12762, 1],
                accentColor="#10b981"
            ),
            DomainInfoDto(
                id="music",
                name="Last.FM",
                vietnameseName="Âm nhạc (Music)",
                itemTerm="Nghệ sĩ / Ban nhạc",
                description="Tập dữ liệu nghe nhạc Last.FM kết hợp mạng lưới tri thức về quốc gia xuất xứ, nhạc cụ biểu diễn, phong cách và nhóm nhạc liên quan.",
                itemsCount=len(self.domains["music"].all_items),
                usersCount=len(self.domains["music"].user_history),
                triplesCount=len(self.domains["music"].kg_np) if self.domains["music"].kg_np is not None else 15518,
                relationsCount=self.domains["music"].n_relation,
                sampleUsers=[774, 79, 238, 386, 1],
                accentColor="#06b6d4"
            )
        ]
        return info

    def get_user_liked_items(self, domain_obj: DomainData, user_id: int, db: Optional[Session] = None) -> List[int]:
        likes = set(domain_obj.user_history.get(user_id, set()))
        likes.update(domain_obj.session_likes.get(user_id, set()))

        # For Movie domain, also merge PostgreSQL likes if DB available
        if domain_obj.name == "movie" and db:
            try:
                pos_ratings = db.query(Rating.movieId).filter(Rating.userId == user_id, Rating.rating >= 3.5).all()
                pos_interactions = db.query(Interaction.movieId).filter(Interaction.userId == user_id, Interaction.type == "LIKE").all()
                db_likes = set([r[0] for r in pos_ratings] + [i[0] for i in pos_interactions])
                likes.update(db_likes)
            except Exception as e:
                logger.debug(f"Could not fetch PG likes: {e}")

        # Exclude session dislikes
        dislikes = domain_obj.session_dislikes.get(user_id, set())
        return list(likes - dislikes)

    def get_user_disliked_items(self, domain_obj: DomainData, user_id: int, db: Optional[Session] = None) -> Set[int]:
        dislikes = set(domain_obj.session_dislikes.get(user_id, set()))
        if domain_obj.name == "movie" and db:
            try:
                neg_ratings = db.query(Rating.movieId).filter(Rating.userId == user_id, Rating.rating < 3.5).all()
                neg_interactions = db.query(Interaction.movieId).filter(Interaction.userId == user_id, Interaction.type == "DISLIKE").all()
                dislikes.update(set([r[0] for r in neg_ratings] + [i[0] for i in neg_interactions]))
            except Exception as e:
                logger.debug(f"Could not fetch PG dislikes: {e}")
        return dislikes

    def get_user_profile(self, domain_name: str, user_id: int) -> Optional[UserProfileDto]:
        """Profile built from the dataset's own interactions (used for book and music)."""
        d = self.get_domain(domain_name)
        history = d.user_history.get(user_id)
        session = d.session_likes.get(user_id)
        if not history and not session and user_id not in d.session_users:
            return None
        likes = sorted(set(history or ()) | set(session or ()))
        hidden = d.session_dislikes.get(user_id, set())
        titles = [d.metadata[i]["title"] for i in likes if i in d.metadata and d.metadata[i].get("title")]
        return UserProfileDto(
            id=user_id,
            name=f"Người dùng #{user_id}",
            email="",
            totalRatings=len(likes) + len(hidden),
            totalLikes=len(likes),
            # ratings_final only holds positives plus sampled negatives, so real dislikes are the hidden items.
            totalDislikes=len(hidden),
            topGenres=[],
            sampleLikes=titles[:3],
        )

    def list_user_profiles(
        self, domain_name: str, page: int = 1, limit: int = 20, search: Optional[str] = None
    ) -> Tuple[int, List[UserProfileDto]]:
        d = self.get_domain(domain_name)
        ids = sorted(set(d.user_history) | {u for u, items in d.session_likes.items() if items} | d.session_users)
        s = (search or "").strip()
        if s:
            ids = [u for u in ids if s in str(u)]
            ids.sort(key=lambda u: (str(u) != s, len(str(u)), u))
        page_ids = ids[(page - 1) * limit: page * limit]
        profiles = [p for p in (self.get_user_profile(domain_name, u) for u in page_ids) if p]
        return len(ids), profiles

    def create_user(self, domain_name: str) -> UserProfileDto:
        """New empty user in a book/music dataset, numbered after the last existing one."""
        d = self.get_domain(domain_name)
        with self._feedback_lock:
            known = set(d.user_history) | set(d.session_likes) | d.session_users
            new_id = (max(known) if known else 0) + 1
            d.session_users.add(new_id)
        self._save_feedback()
        return self.get_user_profile(domain_name, new_id)

    def record_feedback(self, domain_name: str, user_id: int, item_id: int, action: str):
        d = self.get_domain(domain_name)
        if action == "LIKE":
            d.session_likes[user_id].add(item_id)
            d.session_dislikes[user_id].discard(item_id)
        elif action == "DISLIKE":
            d.session_dislikes[user_id].add(item_id)
            d.session_likes[user_id].discard(item_id)
        self._save_feedback()

    def _save_feedback(self):
        payload = {
            name: {
                "likes": {str(u): sorted(items) for u, items in d.session_likes.items() if items},
                "dislikes": {str(u): sorted(items) for u, items in d.session_dislikes.items() if items},
                "users": sorted(d.session_users),
            }
            for name, d in self.domains.items()
        }
        try:
            with self._feedback_lock:
                tmp_path = FEEDBACK_PATH.with_suffix(".json.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.replace(tmp_path, FEEDBACK_PATH)
        except OSError as e:
            logger.warning(f"Could not persist feedback: {e}")

    def _load_feedback(self):
        if not FEEDBACK_PATH.exists():
            return
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for name, saved in payload.items():
                d = self.domains.get(name)
                if not d:
                    continue
                for u, items in saved.get("likes", {}).items():
                    d.session_likes[int(u)].update(int(i) for i in items)
                for u, items in saved.get("dislikes", {}).items():
                    d.session_dislikes[int(u)].update(int(i) for i in items)
                d.session_users.update(int(u) for u in saved.get("users", []))
            logger.info(f"Restored saved feedback from {FEEDBACK_PATH.name}.")
        except (OSError, ValueError) as e:
            logger.warning(f"Could not restore feedback: {e}")

    def _aggregate(self, domain_obj: DomainData, parts: List[torch.Tensor]) -> torch.Tensor:
        """Combine the 0-hop and per-layer embeddings, in the same order as the notebook's CKAN."""
        agg = domain_obj.model.agg
        if agg == "concat":
            return torch.cat(parts, dim=-1)
        if agg == "sum":
            return torch.stack(parts, dim=0).sum(dim=0)
        return parts[-1]

    def _ensure_item_matrix(self, domain_obj: DomainData):
        """Item embeddings do not depend on the user. The notebook export ships them
        (item_embeddings.npy); without it they are rebuilt here the way the notebook's
        KnowledgeRippleSampler does: layer 0 from the item, deeper layers from the previous tails."""
        if domain_obj.item_matrix is not None:
            return
        model = domain_obj.model
        n_layer, size = model.n_layer, domain_obj.itss
        items = list(domain_obj.all_items)
        width = model.dim * (n_layer + 1) if model.agg == "concat" else model.dim
        matrix = torch.zeros((max(items) + 1 if items else 0, width), device=self.device)

        with torch.no_grad():
            for start in range(0, len(items), 1024):
                batch = items[start:start + 1024]
                layers = [([], [], []) for _ in range(n_layer)]
                for item_id in batch:
                    rng = np.random.RandomState(item_id)
                    entities = [item_id]
                    for l in range(n_layer):
                        triples = [(e, r, t) for e in entities for t, r in domain_obj.kg_dict.get(e, [])]
                        if triples:
                            idx = rng.choice(len(triples), size=size, replace=len(triples) < size)
                            h, r, t = zip(*(triples[i] for i in idx))
                        else:
                            h = r = t = (0,) * size
                        for column, values in zip(layers[l], (h, r, t)):
                            column.append(values)
                        entities = t

                parts = [model.entity_emb(torch.as_tensor(batch, dtype=torch.long, device=self.device))]
                for h, r, t in layers:
                    as_tensor = lambda rows: torch.as_tensor(rows, dtype=torch.long, device=self.device)
                    parts.append(model._knowledge_attention(
                        model.entity_emb(as_tensor(h)), model.relation_emb(as_tensor(r)), model.entity_emb(as_tensor(t))
                    ))
                matrix[torch.as_tensor(batch, dtype=torch.long, device=self.device)] = self._aggregate(domain_obj, parts)

        domain_obj.item_matrix = matrix

    def predict_scores(
        self,
        domain_obj: DomainData,
        user_triple_set: List[Any],
        candidate_items: List[int]
    ) -> np.ndarray:
        """sigmoid(user embedding . item embedding), as in the notebook's CKAN.forward."""
        if not candidate_items:
            return np.array([])

        self._ensure_item_matrix(domain_obj)
        model = domain_obj.model
        as_row = lambda values: torch.as_tensor(values, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            parts = [model.entity_emb(as_row(user_triple_set[0][0])).mean(dim=1)]
            for l in range(model.n_layer):
                parts.append(model._knowledge_attention(
                    model.entity_emb(as_row(user_triple_set[l][0])),
                    model.relation_emb(as_row(user_triple_set[l][1])),
                    model.entity_emb(as_row(user_triple_set[l][2])),
                ))
            user_emb = self._aggregate(domain_obj, parts).squeeze(0)

            rows = torch.as_tensor(candidate_items, dtype=torch.long, device=self.device)
            scores = torch.sigmoid(domain_obj.item_matrix.index_select(0, rows) @ user_emb)

        return scores.cpu().numpy()

    def recommend(
        self,
        domain_name: str = "movie",
        user_id: int = 1,
        top_k: int = 12,
        custom_liked_items: Optional[List[int]] = None,
        db: Optional[Session] = None
    ) -> List[RecommendationItemDto]:
        domain_obj = self.get_domain(domain_name)

        # 1. Get user interactions
        if custom_liked_items is not None and len(custom_liked_items) > 0:
            liked_items = list(custom_liked_items)
            disliked_items = set()
        else:
            liked_items = self.get_user_liked_items(domain_obj, user_id, db)
            disliked_items = self.get_user_disliked_items(domain_obj, user_id, db)

        seen_items = set(liked_items).union(disliked_items)

        # 2. Select candidates
        all_items = domain_obj.all_items
        candidates = [it for it in all_items if it not in seen_items]
        if not candidates:
            candidates = all_items[:100]

        # 3. Cold start handling
        if not liked_items:
            return self._popular_fallback(domain_obj, candidates, top_k)

        # 4. Dynamic user triple set
        user_ts = generate_user_triple_set(
            liked_items=liked_items,
            kg_dict=domain_obj.kg_dict,
            n_layer=domain_obj.model_args.n_layer,
            set_size=domain_obj.utss,
            # Same user and same likes give the same sample, so a refresh does not reshuffle the list.
            rng=np.random.RandomState(user_id % (2 ** 32)),
        )

        # 5. Model scoring
        scores = self.predict_scores(domain_obj, user_ts, candidates)
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 6. Build results
        results = []
        for idx in top_indices:
            item_id = candidates[idx]
            raw_score = float(scores[idx])
            meta = domain_obj.metadata.get(item_id, {})
            score = round(max(0.10, min(0.99, raw_score)), 4)

            # Discover KG reasons
            reasons = self._find_quick_kg_reasons(domain_obj, liked_items, item_id)
            if not reasons:
                reasons = [f"Phù hợp với đặc trưng sở thích trong tập {domain_obj.name.capitalize()}"]

            # Subtitle & details
            subtitle, sec_info = self._format_item_subtitles(domain_obj.name, meta)

            results.append(RecommendationItemDto(
                id=item_id,
                movieId=item_id if domain_obj.name == "movie" else None,
                movieLensId=meta.get("movie_id") if domain_obj.name == "movie" else None,
                domain=domain_obj.name,
                title=meta.get("title", f"Mục #{item_id}"),
                subtitle=subtitle,
                secondaryInfo=sec_info,
                releaseYear=meta.get("release_year"),
                genres=meta.get("genres", []),
                posterUrl=meta.get("posterUrl"),
                score=score,
                totalRatings=0,
                reasons=reasons,
                metadata=meta
            ))

        return results

    def _format_item_subtitles(self, domain: str, meta: Dict[str, Any]) -> Tuple[str, str]:
        if domain == "movie":
            year = meta.get("release_year")
            genres = ", ".join(meta.get("genres", [])[:2])
            sub = genres or "Phim điện ảnh"
            sec = str(year) if year else ""
            return sub, sec
        elif domain == "book":
            author = meta.get("author", "Tác giả chưa xác định")
            publisher = meta.get("publisher", "")
            year = meta.get("release_year")
            sec = f"{publisher} ({year})" if year else publisher
            return author, sec
        elif domain == "music":
            author = meta.get("author", "Nghệ sĩ / Ban nhạc")
            url = meta.get("url", "")
            return author, "Last.FM Profile"
        return "", ""

    def _popular_fallback(self, domain_obj: DomainData, candidates: List[int], top_k: int) -> List[RecommendationItemDto]:
        """MostPopular baseline: the items most users liked, the same list for everyone."""
        popularity = domain_obj.popularity
        n_users = max(1, len(domain_obj.user_history))
        ranked = heapq.nlargest(top_k, candidates, key=lambda i: (popularity.get(i, 0), -i))
        results = []
        for item_id in ranked:
            meta = domain_obj.metadata.get(item_id, {})
            sub, sec = self._format_item_subtitles(domain_obj.name, meta)
            liked_by = popularity.get(item_id, 0)
            results.append(RecommendationItemDto(
                id=item_id,
                movieId=item_id if domain_obj.name == "movie" else None,
                movieLensId=meta.get("movie_id") if domain_obj.name == "movie" else None,
                domain=domain_obj.name,
                title=meta.get("title", f"Mục #{item_id}"),
                subtitle=sub,
                secondaryInfo=sec,
                releaseYear=meta.get("release_year"),
                genres=meta.get("genres", []),
                posterUrl=meta.get("posterUrl"),
                # Share of the dataset's users who liked the item, not a model score.
                score=round(liked_by / n_users, 4),
                reasons=[f"Phổ biến: {liked_by:,} người dùng trong tập dữ liệu đã thích".replace(",", ".")],
                metadata=meta
            ))
        return results

    def _find_quick_kg_reasons(self, domain_obj: DomainData, liked_items: List[int], target_id: int) -> List[str]:
        target_nbrs = domain_obj.kg_dict.get(target_id, [])
        if not target_nbrs:
            return []

        reasons = []
        liked_set = set(liked_items)
        rel_map = domain_obj.relations

        for t, r2 in target_nbrs[:25]:
            for h1, r1 in domain_obj.tail_to_heads.get(t, []):
                if h1 in liked_set and h1 != target_id:
                    src_title = domain_obj.metadata.get(h1, {}).get("title", f"#{h1}")
                    rel_name = rel_map.get(str(r2), rel_map.get(str(r1), f"quan hệ {r1}"))
                    clean_rel = self._humanize_relation(rel_name)
                    r_text = f"Cùng {clean_rel} với \"{src_title}\""
                    if r_text not in reasons:
                        reasons.append(r_text)
                    if len(reasons) >= 2:
                        return reasons
        return reasons

    def _humanize_relation(self, raw_rel: str) -> str:
        return relation_role(raw_rel)

    def _entity_label(self, domain_obj: DomainData, entity_id: int, raw_rel: str) -> str:
        """Curated name when one exists, otherwise the role of the entity plus its id."""
        name = domain_obj.entity_names.get(str(entity_id))
        if name:
            return name
        role = relation_role(raw_rel)
        return f"{role[:1].upper()}{role[1:]} #{entity_id}"

    def _score_item(self, domain_obj: DomainData, liked_items: List[int], item_id: int) -> float:
        """CKAN score for one user-item pair. The triple sample is seeded so that two calls
        (with and without a liked item) are comparable."""
        user_ts = generate_user_triple_set(
            liked_items=liked_items,
            kg_dict=domain_obj.kg_dict,
            n_layer=domain_obj.model_args.n_layer,
            set_size=domain_obj.utss,
            rng=np.random.RandomState(20260),
        )
        raw = float(self.predict_scores(domain_obj, user_ts, [item_id])[0])
        return max(0.0, min(1.0, raw))

    def explain(
        self,
        domain_name: str,
        user_id: int,
        item_id: int,
        db: Optional[Session] = None
    ) -> ExplanationResponseDto:
        domain_obj = self.get_domain(domain_name)
        if item_id not in domain_obj.metadata and item_id not in domain_obj.kg_dict:
            raise ItemNotFoundError(f"Mục {item_id} không tồn tại trong tập {domain_obj.name}.")

        target_meta = domain_obj.metadata.get(item_id, {})
        target_title = target_meta.get("title", f"Mục #{item_id}")
        liked_items = self.get_user_liked_items(domain_obj, user_id, db)

        # Find 2-hop KG paths: Liked -> rel1 -> Entity <- rel2 <- Target
        paths: List[ExplanationPathDto] = []
        nodes: Dict[str, GraphNodeDto] = {}
        edges: List[GraphEdgeDto] = []
        edge_ids: Set[str] = set()
        weights: Dict[str, float] = defaultdict(float)
        primary_source_id: Optional[int] = None

        def add_edge(edge_id: str, source: str, target: str, rel_type: str, label: str):
            if edge_id not in edge_ids:
                edge_ids.add(edge_id)
                edges.append(GraphEdgeDto(id=edge_id, source=source, target=target, type=rel_type, label=label))

        user_node_id = f"user-{user_id}"
        nodes[user_node_id] = GraphNodeDto(
            id=user_node_id,
            label=f"Người dùng #{user_id}",
            type="User",
            data={"id": user_id, "name": f"Người dùng #{user_id}"}
        )

        target_node_id = f"{domain_obj.name}-{item_id}"
        nodes[target_node_id] = GraphNodeDto(
            id=target_node_id,
            label=target_title,
            type="ItemRecommended",
            data={
                "id": item_id,
                "title": target_title,
                "releaseYear": target_meta.get("release_year"),
                "genres": target_meta.get("genres", []),
                "posterUrl": target_meta.get("posterUrl"),
                "domain": domain_obj.name
            }
        )

        liked_set = set(liked_items)
        rel_map = domain_obj.relations
        max_paths = 8
        seen_links: Set[Tuple[int, int]] = set()

        for t, r2 in domain_obj.kg_dict.get(item_id, []):
            if len(paths) >= max_paths:
                break
            for h1, r1 in domain_obj.tail_to_heads.get(t, []):
                if h1 not in liked_set or h1 == item_id or (h1, t) in seen_links:
                    continue
                seen_links.add((h1, t))
                src_meta = domain_obj.metadata.get(h1, {})
                src_title = src_meta.get("title", f"Mục #{h1}")
                if primary_source_id is None:
                    primary_source_id = h1

                r1_name = rel_map.get(str(r1), f"rel_{r1}")
                r2_name = rel_map.get(str(r2), f"rel_{r2}")
                role = relation_role(r2_name)
                ent_label = self._entity_label(domain_obj, t, r2_name)

                src_node_id = f"{domain_obj.name}-{h1}"
                if src_node_id not in nodes:
                    nodes[src_node_id] = GraphNodeDto(
                        id=src_node_id,
                        label=src_title,
                        type="ItemLiked",
                        data={
                            "id": h1,
                            "title": src_title,
                            "releaseYear": src_meta.get("release_year"),
                            "genres": src_meta.get("genres", []),
                            "posterUrl": src_meta.get("posterUrl"),
                        }
                    )
                add_edge(f"e-user-{h1}", user_node_id, src_node_id, "LIKED", "đã thích")

                ent_node_id = f"entity-{t}"
                if ent_node_id not in nodes:
                    nodes[ent_node_id] = GraphNodeDto(
                        id=ent_node_id,
                        label=ent_label,
                        type=relation_node_type(r2_name),
                        data={"id": t, "name": ent_label, "relation": r2_name}
                    )
                add_edge(f"e-{src_node_id}-{ent_node_id}", src_node_id, ent_node_id, r1_name, relation_role(r1_name))
                add_edge(f"e-{target_node_id}-{ent_node_id}", target_node_id, ent_node_id, r2_name, role)

                weights[f"{role[:1].upper()}{role[1:]}"] += 1.0
                paths.append(ExplanationPathDto(
                    id=f"path-{len(paths)}",
                    sourceMovieTitle=src_title,
                    relation=r1_name,
                    relationLabel=relation_role(r1_name),
                    entityName=ent_label,
                    targetMovieTitle=target_title,
                    naturalLanguage=f"Người dùng đã thích \"{src_title}\", có chung {role} với \"{target_title}\"."
                ))
                if len(paths) >= max_paths:
                    break

        # Share of the found paths per relation type (a count, not a model attribution).
        total_w = sum(weights.values())
        importance = {k: round((v / total_w) * 100, 1) for k, v in weights.items()} if total_w > 0 else {}

        n_paths = len(paths)
        counterfactual = ""
        if not liked_items:
            score = 0.0
            confidence = "Không đánh giá được · người dùng chưa có lượt thích"
            summary = (
                f"Người dùng #{user_id} chưa có lượt thích nào trong tập {domain_obj.name}, "
                f"nên \"{target_title}\" chỉ là gợi ý mặc định, không có đường dẫn tri thức để lý giải."
            )
        else:
            score = self._score_item(domain_obj, liked_items, item_id)
            if n_paths >= 4:
                confidence = f"Cao · {n_paths} đường dẫn tri thức"
            elif n_paths >= 1:
                confidence = f"Trung bình · {n_paths} đường dẫn tri thức"
            else:
                confidence = "Thấp · không có đường dẫn trực tiếp"

            if primary_source_id is not None:
                src_title = domain_obj.metadata.get(primary_source_id, {}).get("title", f"Mục #{primary_source_id}")
                summary = (
                    f"\"{target_title}\" có {n_paths} liên kết trong đồ thị tri thức với những mục người dùng "
                    f"đã thích, bắt đầu từ \"{src_title}\"."
                )
                remaining = [i for i in liked_items if i != primary_source_id]
                if remaining:
                    without = self._score_item(domain_obj, remaining, item_id)
                    if abs(score - without) < 0.005:
                        counterfactual = (
                            f"Bỏ lượt thích \"{src_title}\" thì điểm CKAN gần như không đổi "
                            f"({score:.3f} so với {without:.3f}): gợi ý này không phụ thuộc riêng vào một lượt thích."
                        )
                    else:
                        counterfactual = (
                            f"Bỏ lượt thích \"{src_title}\" thì điểm CKAN cho \"{target_title}\" "
                            f"đổi từ {score:.3f} thành {without:.3f}."
                        )
            else:
                summary = (
                    f"Không tìm thấy đường dẫn hai bước nào trong đồ thị tri thức giữa \"{target_title}\" và các mục "
                    f"người dùng đã thích. Điểm số đến từ embedding mà CKAN học được."
                )

        return ExplanationResponseDto(
            userId=user_id,
            movieId=item_id,
            domain=domain_obj.name,
            score=round(score, 4),
            confidence=confidence,
            executiveSummary=summary,
            paths=paths,
            featureImportance=importance,
            counterfactual=counterfactual,
            subgraph=SubgraphResponseDto(nodes=list(nodes.values()), edges=edges)
        )

    def simulate_cold_start(self, domain_name: str, interactions: int = 3) -> ColdStartSimulationResponseDto:
        """Measured sparsity results from the notebook plus a live comparison: what MostPopular
        and CKAN recommend when only `interactions` of the demo user's likes are known."""
        domain_obj = self.get_domain(domain_name)
        n = max(1, min(20, interactions))

        measured = self.benchmark.get("datasets", {}).get(domain_obj.name, {})
        sparsity_data = measured.get("sparsity", {})
        aucs = sparsity_data.get("auc", {})
        sparsity = [
            SparsityPointDto(
                ratio=ratio,
                mf_auc=aucs["MF"][i],
                ripplenet_auc=aucs["RippleNet"][i],
                ckan_auc=aucs["CKAN"][i],
            )
            for i, ratio in enumerate(sparsity_data.get("ratios", []))
        ]
        models = [ModelResultDto(model=name, **values) for name, values in measured.get("models", {}).items()]

        demo_user = 1 if domain_obj.name == "movie" else (790 if domain_obj.name == "book" else 774)
        seed = sorted(domain_obj.user_history.get(demo_user, set()))[:n]
        ckan_recs = self.recommend(domain_obj.name, user_id=demo_user, top_k=5, custom_liked_items=seed) if seed else []
        seed_set = set(seed)
        popular = self._popular_fallback(
            domain_obj, [i for i in domain_obj.all_items if i not in seed_set], top_k=5
        )

        return ColdStartSimulationResponseDto(
            domain=domain_obj.name,
            interactions=n,
            source=self.benchmark.get("source", ""),
            sparsity=sparsity,
            sparsityEvalUsers=sparsity_data.get("evalUsers"),
            sparsityEvalRows=sparsity_data.get("evalRows"),
            models=models,
            seedItems=[domain_obj.metadata.get(i, {}).get("title", f"Mục #{i}") for i in seed],
            popularRecommendations=popular,
            ckanRecommendations=ckan_recs,
        )

    def get_items(
        self,
        domain_name: str = "movie",
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        category: Optional[str] = None
    ) -> ItemListResponseDto:
        domain_obj = self.get_domain(domain_name)
        all_items = domain_obj.all_items

        filtered = []
        s_lower = search.lower() if search else None
        c_lower = category.lower() if category else None

        for item_id in all_items:
            meta = domain_obj.metadata.get(item_id, {})
            title = meta.get("title", f"Mục #{item_id}")
            sub, sec = self._format_item_subtitles(domain_obj.name, meta)

            if s_lower:
                in_title = s_lower in title.lower()
                in_sub = s_lower in sub.lower()
                if not (in_title or in_sub):
                    continue

            if c_lower:
                genres = [g.lower() for g in meta.get("genres", [])]
                if not any(c_lower in g for g in genres) and c_lower not in sub.lower():
                    continue

            filtered.append(ItemDto(
                id=item_id,
                domain=domain_obj.name,
                title=title,
                subtitle=sub,
                category=sec or sub,
                posterUrl=meta.get("posterUrl"),
                releaseYear=meta.get("release_year"),
                details=meta
            ))

        total = len(filtered)
        start = (page - 1) * limit
        end = start + limit
        paginated = filtered[start:end]

        return ItemListResponseDto(
            domain=domain_obj.name,
            total=total,
            page=page,
            limit=limit,
            data=paginated
        )

recommendation_engine = RecommendationEngine()
