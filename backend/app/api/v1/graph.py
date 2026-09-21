import re
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.neo4j_client import neo4j_client
from app.core.entity_resolver import entity_resolver
from app.models.sql_models import Movie, Rating, User
from app.models.schemas import SubgraphResponseDto, GraphNodeDto, GraphEdgeDto
from app.recommendation.engine import recommendation_engine

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])

@router.get("/subgraph/{movieId}", response_model=SubgraphResponseDto)
def get_movie_subgraph(movieId: int, db: Session = Depends(get_db)):
    movie = db.query(Movie).filter(Movie.id == movieId).first()
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie {movieId} not found")

    cypher_meta = """
    MATCH (m:Movie {id: $id})-[r]->(target)
    WHERE NOT target:User
    RETURN 
      m.id AS movieId,
      m.title AS movieTitle,
      type(r) AS relationship,
      labels(target)[0] AS targetType,
      coalesce(target.name, target.title, target.id) AS targetName,
      coalesce(target.id, target.name, target.title) AS targetId
    """
    cypher_users = """
    MATCH (u:User)-[r:LIKED]->(m:Movie {id: $id})
    RETURN 
      m.id AS movieId,
      m.title AS movieTitle,
      type(r) AS relationship,
      'User' AS targetType,
      coalesce(u.name, 'User #' + toString(u.id)) AS targetName,
      u.id AS targetId
    LIMIT 6
    """
    records_meta = neo4j_client.read(cypher_meta, {"id": movieId}) or []
    records_users = neo4j_client.read(cypher_users, {"id": movieId}) or []
    records = records_meta + records_users

    nodes = {}
    edges = []

    # Central movie node
    center_id = f"movie-{movieId}"
    nodes[center_id] = GraphNodeDto(
        id=center_id,
        label=movie.title,
        type="Movie",
        data={
            "id": movie.id,
            "title": movie.title,
            "releaseYear": movie.releaseYear,
            "genres": movie.genres or [],
            "posterUrl": movie.posterUrl
        }
    )

    for idx, r in enumerate(records):
        t_type = r.get("targetType") or "Entity"
        t_id = r.get("targetId")
        raw_name = r.get("targetName") or str(t_id)
        t_name = entity_resolver.resolve(raw_name, t_type)
        rel = r.get("relationship") or "RELATED"

        safe_val = str(t_id if t_id is not None else raw_name)
        safe_id = re.sub(r"[\s/\\#?&]+", "_", safe_val)
        node_key = f"{t_type}-{safe_id}"

        if node_key not in nodes:
            nodes[node_key] = GraphNodeDto(
                id=node_key,
                label=str(t_name),
                type=t_type,
                data={"id": t_id, "name": t_name, "type": t_type}
            )

        edge_id = f"e-{center_id}-{rel}-{node_key}"
        if not any(e.id == edge_id for e in edges):
            edges.append(GraphEdgeDto(
                id=edge_id,
                source=center_id,
                target=node_key,
                type=rel,
                label=rel
            ))

    return SubgraphResponseDto(
        nodes=list(nodes.values()),
        edges=edges
    )

@router.get("/user-subgraph/{userId}", response_model=SubgraphResponseDto)
def get_user_subgraph(userId: int, topK: int = 5, db: Session = Depends(get_db)):
    nodes = {}
    edge_map = {}

    # 1. User Central Node
    u_key = f"user-{userId}"
    user = db.query(User).filter(User.id == userId).first()
    u_name = user.name if user and user.name else f"User #{userId}"
    nodes[u_key] = GraphNodeDto(
        id=u_key,
        label=u_name,
        type="User",
        data={"id": userId, "label": u_name}
    )

    # 2. Fetch multiple liked movies from Postgres (up to 7-8 liked movies)
    liked_records = (
        db.query(Rating, Movie)
        .join(Movie, Movie.id == Rating.movieId)
        .filter(Rating.userId == userId, Rating.rating >= 3.5)
        .order_by(Rating.rating.desc(), Rating.id.desc())
        .limit(7)
        .all()
    )
    if not liked_records:
        liked_records = (
            db.query(Rating, Movie)
            .join(Movie, Movie.id == Rating.movieId)
            .filter(Rating.userId == userId)
            .order_by(Rating.id.desc())
            .limit(7)
            .all()
        )

    liked_ids = []
    for r, m in liked_records:
        liked_ids.append(m.id)
        src_id = f"movie-{m.id}"
        nodes[src_id] = GraphNodeDto(
            id=src_id,
            label=m.title,
            type="MovieLiked",
            data={
                "id": m.id,
                "title": m.title,
                "releaseYear": m.releaseYear,
                "genres": m.genres or [],
                "posterUrl": m.posterUrl,
                "rating": r.rating
            }
        )
        # Direct User -> Liked Movie Edge
        e1_key = f"{u_key}->{src_id}"
        edge_map[e1_key] = GraphEdgeDto(
            id=f"e-{u_key}-{src_id}",
            source=u_key,
            target=src_id,
            type="LIKED",
            label="LIKED"
        )

    # 3. Get top recommendations from CKAN engine
    rec_ids = []
    try:
        recs = recommendation_engine.recommend(user_id=userId, top_k=topK, db=db)
        for r in recs:
            rec_ids.append(r.movieId)
            tgt_id = f"movie-{r.movieId}"
            if tgt_id not in nodes:
                nodes[tgt_id] = GraphNodeDto(
                    id=tgt_id,
                    label=r.title,
                    type="MovieRecommended",
                    data={
                        "id": r.movieId,
                        "title": r.title,
                        "releaseYear": r.releaseYear,
                        "genres": r.genres or [],
                        "posterUrl": r.posterUrl,
                        "score": r.score
                    }
                )
    except Exception as e:
        print(f"Error obtaining recommendations for user subgraph: {e}")

    # 4. Multi-hop KG paths connecting liked movies to recommended movies
    if liked_ids and rec_ids:
        path_cypher = """
        MATCH (m1:Movie)-[r1]->(common)<-[r2]-(m2:Movie)
        WHERE m1.id IN $likedIds AND m2.id IN $recIds AND NOT type(r1) = 'LIKED'
        RETURN 
          m1.id AS sourceId,
          m1.title AS sourceTitle,
          type(r1) AS r1Type,
          labels(common)[0] AS commonType,
          coalesce(common.name, common.title, common.id) AS commonName,
          type(r2) AS r2Type,
          m2.id AS targetId,
          m2.title AS targetTitle
        LIMIT 50
        """
        records = neo4j_client.read(path_cypher, {"likedIds": liked_ids, "recIds": rec_ids}) or []

        for r in records:
            src_id = f"movie-{r['sourceId']}"
            tgt_id = f"movie-{r['targetId']}"
            raw_c_name = str(r.get("commonName", "Entity"))
            c_type = str(r.get("commonType", "Entity"))
            c_name = entity_resolver.resolve(raw_c_name, c_type)
            safe_c_id = re.sub(r"[\s/\\#?&]+", "_", str(raw_c_name))
            c_key = f"{c_type}-{safe_c_id}"

            # Ensure common entity node
            if c_key not in nodes:
                nodes[c_key] = GraphNodeDto(
                    id=c_key,
                    label=c_name,
                    type=c_type,
                    data={"name": c_name, "raw_id": raw_c_name, "type": c_type}
                )

            # m1 -> common
            e2_key = f"{src_id}->{c_key}:{r['r1Type']}"
            if e2_key not in edge_map:
                edge_map[e2_key] = GraphEdgeDto(
                    id=f"e-{src_id}-{c_key}",
                    source=src_id,
                    target=c_key,
                    type=r["r1Type"],
                    label=r["r1Type"]
                )

            # common -> m2
            e3_key = f"{c_key}->{tgt_id}:{r['r2Type']}"
            if e3_key not in edge_map:
                edge_map[e3_key] = GraphEdgeDto(
                    id=f"e-{c_key}-{tgt_id}",
                    source=c_key,
                    target=tgt_id,
                    type=r["r2Type"],
                    label=r["r2Type"]
                )

    # 5. For any liked movies without a multi-hop path, attach their direct genre / director nodes
    if liked_ids:
        ent_cypher = """
        MATCH (m:Movie)-[r]->(common)
        WHERE m.id IN $likedIds AND NOT common:User AND NOT type(r) = 'LIKED'
        RETURN 
          m.id AS sourceId,
          type(r) AS rType,
          labels(common)[0] AS commonType,
          coalesce(common.name, common.title, common.id) AS commonName
        LIMIT 30
        """
        ent_records = neo4j_client.read(ent_cypher, {"likedIds": liked_ids}) or []
        for r in ent_records:
            src_id = f"movie-{r['sourceId']}"
            raw_c_name = str(r.get("commonName", "Entity"))
            c_type = str(r.get("commonType", "Entity"))
            c_name = entity_resolver.resolve(raw_c_name, c_type)
            safe_c_id = re.sub(r"[\s/\\#?&]+", "_", str(raw_c_name))
            c_key = f"{c_type}-{safe_c_id}"

            if c_key not in nodes:
                nodes[c_key] = GraphNodeDto(
                    id=c_key,
                    label=c_name,
                    type=c_type,
                    data={"name": c_name, "raw_id": raw_c_name, "type": c_type}
                )

            e_key = f"{src_id}->{c_key}:{r['rType']}"
            if e_key not in edge_map:
                edge_map[e_key] = GraphEdgeDto(
                    id=f"e-{src_id}-{c_key}",
                    source=src_id,
                    target=c_key,
                    type=r["rType"],
                    label=r["rType"]
                )

    return SubgraphResponseDto(nodes=list(nodes.values()), edges=list(edge_map.values()))

