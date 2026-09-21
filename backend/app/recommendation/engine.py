import os
import json
import pickle
import logging
from typing import List, Dict, Set, Any, Optional
import numpy as np
import torch
from sqlalchemy.orm import Session

from app.core.config import settings, BACKEND_DIR
from app.core.neo4j_client import neo4j_client
from app.models.sql_models import Rating, Interaction, Movie
from app.models.schemas import RecommendationItemDto
from app.recommendation.ckan_model import CKAN
from app.recommendation.dynamic_propagation import (
    build_kg_dict,
    generate_user_triple_set,
    get_triple_tensor,
)

logger = logging.getLogger("recommendation_engine")

class ModelArgs:
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, v)

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
        self.model: Optional[CKAN] = None
        self.model_args: Optional[ModelArgs] = None
        self.device = torch.device("cpu")
        self.is_loaded = False
        
        self.user_triple_set: Dict[int, Any] = {}
        self.item_triple_set: Dict[int, Any] = {}
        self.user_history: Dict[int, Set[int]] = {}
        self.all_items: List[int] = []
        self.movie_metadata: Dict[int, Dict[str, Any]] = {}
        self.kg_dict: Dict[int, List[tuple]] = {}
        self.n_entity = 0
        self.n_relation = 0
        self._initialized = True

    def initialize(self):
        if self.is_loaded:
            return

        logger.info("Initializing Recommendation Engine...")
        use_cuda = settings.USE_CUDA and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"Engine device: {self.device}")

        # 1. Load Knowledge Graph dictionary for dynamic on-the-fly propagation
        if os.path.exists(settings.KG_PATH):
            logger.info(f"Loading Knowledge Graph from {settings.KG_PATH}...")
            kg_np = np.load(settings.KG_PATH)
            self.kg_dict = build_kg_dict(kg_np)
            self.n_entity = int(max(np.max(kg_np[:, 0]), np.max(kg_np[:, 2]))) + 1
            self.n_relation = int(np.max(kg_np[:, 1])) + 1
            logger.info(f"Knowledge Graph ready: {len(self.kg_dict)} heads, {self.n_entity} entities, {self.n_relation} relations.")
        else:
            logger.warning(f"KG file not found at {settings.KG_PATH}")

        # 2. Load propagation cache if exists
        if os.path.exists(settings.CACHE_PATH):
            logger.info(f"Loading propagation cache from {settings.CACHE_PATH}...")
            with open(settings.CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            self.user_triple_set = cache.get("user_triple_set", {})
            self.item_triple_set = cache.get("item_triple_set", {})
            self.all_items = sorted([int(k) for k in self.item_triple_set.keys()])
            train_data = cache.get("train_data", [])
            for u, i, r in train_data:
                if r == 1:
                    self.user_history.setdefault(int(u), set()).add(int(i))
            logger.info(f"Cache loaded: {len(self.user_triple_set)} users, {len(self.all_items)} items.")
        else:
            logger.warning(f"Cache file not found at {settings.CACHE_PATH}. Dynamic propagation will be used.")

        # 3. Load model weights if checkpoint exists, otherwise initialize clean model
        model_args_dict = {
            "dim": 64,
            "n_layer": 1,
            "agg": "concat",
            "batch_size": settings.BATCH_SIZE,
            "use_cuda": use_cuda,
        }

        # Check candidate checkpoint paths
        candidate_paths = [
            settings.MODEL_PATH,
            str(BACKEND_DIR / "models" / "ckan_model.pt"),
            str(BACKEND_DIR / "models" / "ckan_movie.pt"),
            "models/ckan_model.pt",
            "models/ckan_movie.pt",
        ]
        chosen_model_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

        if chosen_model_path:
            logger.info(f"Loading CKAN checkpoint from {chosen_model_path}...")
            try:
                checkpoint = torch.load(chosen_model_path, map_location=self.device)
                saved_args = checkpoint.get("args", model_args_dict)
                self.model_args = ModelArgs(saved_args)
                self.n_entity = checkpoint.get("n_entity", self.n_entity)
                self.n_relation = checkpoint.get("n_relation", self.n_relation)
                self.model = CKAN(self.model_args, self.n_entity, self.n_relation)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                best_auc = checkpoint.get("best_auc")
                best_f1 = checkpoint.get("best_f1")
                extra_info = f" (Peak AUC: {best_auc:.4f}, F1: {best_f1:.4f})" if best_auc else ""
                logger.info(f"CKAN checkpoint loaded successfully{extra_info}.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Initializing fresh CKAN model.")
                self.model_args = ModelArgs(model_args_dict)
                self.model = CKAN(self.model_args, max(self.n_entity, 100000), max(self.n_relation, 100))
        else:
            logger.info("No checkpoint found. Initializing CKAN model with Xavier weights.")
            self.model_args = ModelArgs(model_args_dict)
            self.model = CKAN(self.model_args, max(self.n_entity, 100000), max(self.n_relation, 100))

        self.model.to(self.device)
        self.model.eval()

        # 4. Load movie metadata
        if os.path.exists(settings.METADATA_PATH):
            logger.info(f"Loading movie metadata from {settings.METADATA_PATH}...")
            with open(settings.METADATA_PATH, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
            self.movie_metadata = {int(m["ckan_id"]): m for m in meta_list}
            if not self.all_items:
                self.all_items = sorted(list(self.movie_metadata.keys()))
            logger.info(f"Loaded {len(self.movie_metadata)} movie metadata records.")

        self.is_loaded = True
        logger.info("Recommendation Engine initialization complete.")

    def get_user_liked_items(self, user_id: int, db: Session) -> List[int]:
        # 1. Likes from PostgreSQL
        pos_ratings = db.query(Rating.movieId).filter(Rating.userId == user_id, Rating.rating >= 3.5).all()
        pos_interactions = db.query(Interaction.movieId).filter(Interaction.userId == user_id, Interaction.type == "LIKE").all()
        db_likes = set([r[0] for r in pos_ratings] + [i[0] for i in pos_interactions])

        # 2. Historical likes from train cache
        cache_likes = self.user_history.get(user_id, set())
        
        all_likes = list(db_likes.union(cache_likes))
        return all_likes

    def get_user_disliked_items(self, user_id: int, db: Session) -> Set[int]:
        neg_ratings = db.query(Rating.movieId).filter(Rating.userId == user_id, Rating.rating < 3.5).all()
        neg_interactions = db.query(Interaction.movieId).filter(Interaction.userId == user_id, Interaction.type == "DISLIKE").all()
        return set([r[0] for r in neg_ratings] + [i[0] for i in neg_interactions])

    def predict_scores(
        self,
        user_triple_set: List[Any],
        candidate_items: List[int]
    ) -> np.ndarray:
        if not candidate_items:
            return np.array([])

        scores = []
        batch_size = settings.BATCH_SIZE
        n_layer = self.model_args.n_layer
        start = 0

        # Build user tensor once for all candidates
        # format: list of [batch_size, triple_set_size] tensors
        u_h, u_r, u_t = [], [], []
        for l in range(n_layer):
            u_h.append(user_triple_set[l][0])
            u_r.append(user_triple_set[l][1])
            u_t.append(user_triple_set[l][2])

        with torch.no_grad():
            while start < len(candidate_items):
                end = min(start + batch_size, len(candidate_items))
                batch_items = candidate_items[start:end]
                b_size = len(batch_items)

                # Expand user tensor for batch size
                batch_user_triple = []
                b_uh = [torch.LongTensor([u_h[l]] * b_size).to(self.device) for l in range(n_layer)]
                b_ur = [torch.LongTensor([u_r[l]] * b_size).to(self.device) for l in range(n_layer)]
                b_ut = [torch.LongTensor([u_t[l]] * b_size).to(self.device) for l in range(n_layer)]
                batch_user_triple = [b_uh, b_ur, b_ut]

                # Items tensor
                items_tensor = torch.LongTensor(batch_items).to(self.device)

                # Items triple set tensor
                # If item not in cache, fallback to self triple
                i_h_list, i_r_list, i_t_list = [], [], []
                for l in range(n_layer):
                    h_l, r_l, t_l = [], [], []
                    for item_id in batch_items:
                        if item_id in self.item_triple_set:
                            h_l.append(self.item_triple_set[item_id][l][0])
                            r_l.append(self.item_triple_set[item_id][l][1])
                            t_l.append(self.item_triple_set[item_id][l][2])
                        else:
                            # Self-loop fallback
                            h_l.append([item_id] * 64)
                            r_l.append([0] * 64)
                            t_l.append([item_id] * 64)
                    i_h_list.append(torch.LongTensor(h_l).to(self.device))
                    i_r_list.append(torch.LongTensor(r_l).to(self.device))
                    i_t_list.append(torch.LongTensor(t_l).to(self.device))
                batch_item_triple = [i_h_list, i_r_list, i_t_list]

                batch_scores = self.model(items_tensor, batch_user_triple, batch_item_triple)
                scores.extend(batch_scores.cpu().numpy())
                start = end

        return np.array(scores)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        db: Optional[Session] = None
    ) -> List[RecommendationItemDto]:
        if not self.is_loaded:
            self.initialize()

        # 1. Get user interactions
        liked_items = self.get_user_liked_items(user_id, db) if db else []
        disliked_items = self.get_user_disliked_items(user_id, db) if db else set()
        seen_items = set(liked_items).union(disliked_items)

        # 2. Filter candidates (Blacklist Dislike + Seen movies)
        candidates = [it for it in self.all_items if it not in seen_items]
        if not candidates:
            return []

        # 3. Handle Zero-interaction Cold-Start (User has liked 0 movies)
        if not liked_items:
            logger.info(f"User {user_id} has zero interactions. Using popular fallback...")
            return self._popular_fallback(candidates, top_k, db)

        # 4. Generate user triple set dynamically on-the-fly (No retrain!)
        user_ts = generate_user_triple_set(
            liked_items=liked_items,
            kg_dict=self.kg_dict,
            n_layer=self.model_args.n_layer,
            set_size=32
        )

        # 5. Compute pure CKAN prediction scores directly
        ckan_scores = self.predict_scores(user_ts, candidates)

        # 6. Rank Top-K directly by pure CKAN scores
        top_indices = np.argsort(ckan_scores)[::-1][:top_k]

        # 7. Query Neo4j strictly for multi-hop graph explanations
        top_movie_ids = [candidates[i] for i in top_indices]
        explanation_map = self._get_explanations(user_id, top_movie_ids)

        # 8. Format response
        results = []
        for idx in top_indices:
            item_id = candidates[idx]
            score = float(ckan_scores[idx])
            meta = self.movie_metadata.get(item_id, {})

            results.append(RecommendationItemDto(
                movieId=item_id,
                movieLensId=meta.get("movie_id"),
                title=meta.get("title", f"Movie #{item_id}"),
                releaseYear=meta.get("release_year"),
                genres=meta.get("genres", []),
                posterUrl=None,
                score=round(max(0.01, min(0.99, score)), 4),
                totalRatings=0,
                reasons=explanation_map.get(item_id) or [
                    "Strong collaborative match based on your preferences"
                ]
            ))

        return results

    def _popular_fallback(self, candidates: List[int], top_k: int, db: Optional[Session]) -> List[RecommendationItemDto]:
        results = []
        # If DB available, pick movies with highest ratings count
        if db:
            popular_movies = (
                db.query(Movie)
                .filter(Movie.id.in_(candidates[:500]))
                .limit(top_k)
                .all()
            )
            for m in popular_movies:
                results.append(RecommendationItemDto(
                    movieId=m.id,
                    movieLensId=m.movieLensId,
                    title=m.title,
                    releaseYear=m.releaseYear,
                    genres=m.genres or [],
                    posterUrl=m.posterUrl,
                    score=0.88,
                    totalRatings=0,
                    reasons=["Popular trending film recommended for your taste exploration"]
                ))
        
        # If not enough, fill with metadata
        if len(results) < top_k:
            for item_id in candidates[:top_k - len(results)]:
                meta = self.movie_metadata.get(item_id, {})
                results.append(RecommendationItemDto(
                    movieId=item_id,
                    movieLensId=meta.get("movie_id"),
                    title=meta.get("title", f"Movie #{item_id}"),
                    releaseYear=meta.get("release_year"),
                    genres=meta.get("genres", []),
                    score=0.85,
                    reasons=["Highly acclaimed title recommended for new movie lovers"]
                ))
        return results

    def _get_explanations(self, user_id: int, movie_ids: List[int]) -> Dict[int, List[str]]:
        if not movie_ids:
            return {}
            
        cypher = """
        MATCH (u:User {id: $userId})-[l:LIKED]->(m1:Movie)-[r1]->(common)<-[r2]-(m2:Movie)
        WHERE m2.id IN $movieIds AND m1.id <> m2.id AND NOT type(r1) = 'LIKED'
        RETURN 
          m2.id AS recMovieId,
          m1.title AS sourceMovieTitle,
          type(r1) AS relation,
          coalesce(common.name, common.title, common.id) AS entityName
        LIMIT 60
        """
        records = neo4j_client.read(cypher, {"userId": user_id, "movieIds": movie_ids})
        exp_map: Dict[int, List[str]] = {}
        for r in records:
            m_id = int(r["recMovieId"])
            text = f"Linked to \"{r['sourceMovieTitle']}\" via {r['relation']} ({r['entityName']})"
            exp_map.setdefault(m_id, [])
            if len(exp_map[m_id]) < 3 and text not in exp_map[m_id]:
                exp_map[m_id].append(text)
        return exp_map

recommendation_engine = RecommendationEngine()

