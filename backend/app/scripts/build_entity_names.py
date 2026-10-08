"""Recover real names for anonymous knowledge-graph entities.

The KG files only carry integer entity ids. Names are recovered by alignment:

* Movies: MovieLens ids are mapped to IMDb ids (links.csv from the MovieLens-20M archive),
  then Wikidata supplies each film's director, cast, writers and so on. An entity that the
  KG attaches to several films through, say, ``film.film.director`` gets the name that
  Wikidata lists as director for most of those films.
* Books: the catalogue metadata already has an author per book, so author entities are
  named by the same vote without any network access.

Only confident matches are written; everything else keeps its "role #id" label.

Run from the backend directory:
    python -m app.scripts.build_entity_names
"""
import collections
import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
CACHE_DIR = BACKEND_DIR / "models" / "cache"
ML20M_ZIP = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "ckan-kg-demo-entity-names/1.0 (university course project)"

# Last segment of the KG relation -> Wikidata properties that describe the same role.
RELATION_PROPS = {
    "director": ["P57"],
    "actor": ["P161", "P725"],
    "star": ["P161", "P725"],
    "performance": ["P161", "P725"],
    "writer": ["P58"],
    "producer": ["P162"],
    "executive_producer": ["P1431"],
    "cinematographer": ["P344"],
    "editor": ["P1040"],
    "music": ["P86"],
    "production_company": ["P272"],
    "country": ["P495"],
    "country_of_origin": ["P495"],
    "language": ["P364"],
    "genre": ["P136"],
    "art_director": ["P3174"],
    "costume_designer": ["P2515"],
    "set_designer": ["P4608"],
    "production_designer": ["P2554"],
    "rating": ["P1657"],
    "series": ["P179"],
    "location": ["P915"],
}
ALL_PROPS = sorted({p for props in RELATION_PROPS.values() for p in props})

MIN_SHARE = 0.6


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, so zipfile can pull one member out of a large archive."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.size = int(resp.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def load_links() -> dict:
    """MovieLens movieId -> IMDb id ('tt0114709')."""
    path = DATA_DIR / "movie" / "links.csv"
    if not path.exists():
        print("Fetching links.csv from the MovieLens-20M archive...", flush=True)
        with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ML20M_ZIP), buffer_size=1 << 16)) as zf:
            path.write_bytes(zf.read("ml-20m/links.csv"))
    links = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("imdbId"):
                links[int(row["movieId"])] = "tt" + row["imdbId"].zfill(7)
    return links


def fetch_credits(imdb_ids: list) -> dict:
    """{imdb_id: {property: [labels]}} from Wikidata, cached on disk between runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "wikidata_credits.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [i for i in imdb_ids if i not in cache]
    batch_size = 150
    props = " ".join(f"wdt:{p}" for p in ALL_PROPS)

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        values = " ".join(f'"{i}"' for i in batch)
        query = (
            "SELECT ?imdb ?prop ?label WHERE { "
            f"VALUES ?imdb {{ {values} }} ?film wdt:P345 ?imdb . "
            f"VALUES ?prop {{ {props} }} ?film ?prop ?value . "
            '?value rdfs:label ?label FILTER(LANG(?label) = "en") }'
        )
        body = urllib.parse.urlencode({"query": query}).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    SPARQL, data=body, headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    rows = json.load(resp)["results"]["bindings"]
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"  Wikidata request failed ({e}); retrying in {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError("Wikidata kept failing; re-run to resume from the cache.")

        for imdb in batch:
            cache[imdb] = {}
        for row in rows:
            prop = row["prop"]["value"].rsplit("/", 1)[-1]
            cache[row["imdb"]["value"]].setdefault(prop, []).append(row["label"]["value"])
        if (start // batch_size) % 10 == 0:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  Wikidata: {min(start + batch_size, len(todo))}/{len(todo)} films", flush=True)
        time.sleep(1.0)

    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def vote(entity_links: dict, candidates_for) -> dict:
    """Name each entity by majority vote over the items it is attached to.

    entity_links: {entity: {(head, role)}}; candidates_for(head, role) -> set of labels or None.
    """
    # How many distinct tails a head has per role; a one-to-one link needs no second witness.
    fanout = collections.Counter()
    for links in entity_links.values():
        for head, role in links:
            fanout[(head, role)] += 1

    names = {}
    for entity, links in entity_links.items():
        counts = collections.Counter()
        witnesses = set()
        sole_candidate = None
        for head, role in links:
            labels = candidates_for(head, role)
            if not labels:
                continue
            witnesses.add(head)
            if len(labels) == 1 and fanout[(head, role)] == 1:
                sole_candidate = next(iter(labels))
        for head in witnesses:
            seen = set()
            for h, role in links:
                if h == head:
                    seen |= candidates_for(h, role) or set()
            counts.update(seen)
        if not counts:
            continue
        label, support = counts.most_common(1)[0]
        runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        if support >= 2 and support / len(witnesses) >= MIN_SHARE and support > runner_up:
            names[entity] = label
        elif len(witnesses) == 1 and sole_candidate:
            names[entity] = sole_candidate
    return names


def load_kg(domain: str):
    kg = np.load(DATA_DIR / domain / "kg_final.npy")
    relations = json.loads((DATA_DIR / domain / "kg_relations.json").read_text(encoding="utf-8"))["index2rel"]
    return kg, {int(k): v.split(".")[-1] for k, v in relations.items()}, relations


def write_names(domain: str, names: dict, kg: np.ndarray, keep: dict = None):
    """Write the voted names plus any previously curated ones in `keep` that the vote left open."""
    path = DATA_DIR / domain / "entity_names.json"
    existing = keep or {}
    merged = {str(k): v for k, v in existing.items()}
    merged.update({str(k): v for k, v in names.items()})
    ordered = dict(sorted(merged.items(), key=lambda kv: int(kv[0])))
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=0), encoding="utf-8")

    tails = kg[:, 2]
    named_edges = int(np.isin(tails, np.fromiter((int(k) for k in ordered), dtype=np.int64)).sum())
    print(
        f"[{domain}] {len(ordered)} of {len(set(tails.tolist()))} entities named "
        f"({len(existing)} kept from the earlier hand-curated file), covering {named_edges / len(kg):.0%} of KG edges",
        flush=True,
    )


def build_movie():
    kg, roles, _ = load_kg("movie")
    meta = json.loads((DATA_DIR / "movie" / "movies_metadata.json").read_text(encoding="utf-8"))
    ml_id = {int(m["ckan_id"]): int(m["movie_id"]) for m in meta}
    links = load_links()
    imdb_of = {item: links[ml] for item, ml in ml_id.items() if ml in links}
    print(f"[movie] {len(imdb_of)} of {len(ml_id)} films have an IMDb id", flush=True)
    wikidata = fetch_credits(sorted(set(imdb_of.values())))

    entity_links = collections.defaultdict(set)
    for head, rel, tail in kg.tolist():
        if head in imdb_of and roles[rel] in RELATION_PROPS:
            entity_links[tail].add((head, roles[rel]))

    def candidates_for(head, role):
        film = wikidata.get(imdb_of[head])
        if not film:
            return None
        labels = set()
        for prop in RELATION_PROPS[role]:
            labels.update(film.get(prop, ()))
        return labels

    voted = vote(entity_links, candidates_for)

    # The earlier hand-curated file is only trusted where the film credits back it up: an
    # entry stays if the vote was silent and its name is credited on most of the entity's films.
    curated_path = DATA_DIR / "movie" / "entity_names.curated.json"
    curated = json.loads(curated_path.read_text(encoding="utf-8")) if curated_path.exists() else {}
    films_of = collections.defaultdict(set)
    for head, _, tail in kg.tolist():
        if head in imdb_of:
            films_of[tail].add(head)
    keep, dropped = {}, 0
    for key, name in curated.items():
        entity = int(key)
        if entity in voted:
            continue
        films = films_of.get(entity, set())
        credited = sum(
            1 for f in films if any(name in labels for labels in wikidata.get(imdb_of[f], {}).values())
        )
        if films and credited / len(films) >= 0.5:
            keep[key] = name
        else:
            dropped += 1
    print(f"[movie] curated file: {len(curated)} entries, {dropped} dropped as unsupported by film credits", flush=True)
    write_names("movie", voted, kg, keep)


def build_book():
    kg, _, relations = load_kg("book")
    meta = json.loads((DATA_DIR / "book" / "books_metadata.json").read_text(encoding="utf-8"))
    author_of = {}
    for m in meta:
        author = (m.get("author") or "").strip()
        if author:
            # The catalogue mixes "RAY BRADBURY" and "Ray Bradbury".
            author_of[int(m["ckan_id"])] = author.title() if author.isupper() or author.islower() else author
    author_rel = {int(k) for k, v in relations.items() if v == "book.written_work.author"}

    entity_links = collections.defaultdict(set)
    for head, rel, tail in kg.tolist():
        if rel in author_rel and head in author_of:
            entity_links[tail].add((head, "author"))

    # Compare case-insensitively, then report the most common spelling.
    spelling = collections.defaultdict(collections.Counter)
    for name in author_of.values():
        spelling[name.casefold()][name] += 1

    names = vote(entity_links, lambda head, role: {author_of[head].casefold()})
    write_names("book", {e: spelling[n].most_common(1)[0][0] for e, n in names.items()}, kg)


def main():
    build_book()
    build_movie()


if __name__ == "__main__":
    sys.exit(main())
