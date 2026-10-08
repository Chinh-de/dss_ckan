"""Fetch artist pictures for the Last.FM domain from the public Deezer search API.

The picture URLs shipped with the HetRec Last.FM dump (userserve-ak.last.fm) no longer
resolve. Deezer's artist search needs no API key, so each artist name is looked up there
and the picture is kept only when the returned name matches exactly after normalisation.

Run from the backend directory:
    python -m app.scripts.fetch_artist_images

Writes data/music/artist_images.json as {ckan_id: picture_url}. Re-running resumes.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "music"
OUT_PATH = DATA_DIR / "artist_images.json"
MISS_PATH = DATA_DIR / "artist_images_missing.json"
API = "https://api.deezer.com/search/artist?limit=5&q="
# Deezer allows 50 requests per 5 seconds; lookup() backs off when the quota error comes back.
WORKERS = 6


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"^the\s+", "", text.casefold())
    return re.sub(r"[^a-z0-9]+", "", text)


def lookup(name: str):
    url = API + urllib.parse.quote(name)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.load(resp)
        except Exception:
            time.sleep(2 * (attempt + 1))
            continue
        if "error" in payload:  # quota exceeded
            time.sleep(5)
            continue
        target = normalise(name)
        for hit in payload.get("data", []):
            picture = hit.get("picture_big") or hit.get("picture_medium") or ""
            # Artists without a photo get a URL with an empty hash segment.
            if normalise(hit.get("name", "")) == target and "/artist//" not in picture and picture:
                return picture
        return None
    raise RuntimeError(f"Deezer lookup kept failing for {name!r}")


def main():
    artists = json.loads((DATA_DIR / "music_metadata.json").read_text(encoding="utf-8"))
    found = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    missing = set(json.loads(MISS_PATH.read_text(encoding="utf-8"))) if MISS_PATH.exists() else set()

    def save():
        OUT_PATH.write_text(json.dumps(found, ensure_ascii=False, indent=0), encoding="utf-8")
        MISS_PATH.write_text(json.dumps(sorted(missing)), encoding="utf-8")

    todo = [a for a in artists if str(a["ckan_id"]) not in found and str(a["ckan_id"]) not in missing]
    print(f"{len(found)} already found, {len(todo)} to look up", flush=True)
    # Lookups are latency-bound, so a few run at once; WORKERS keeps this under Deezer's quota.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(todo), 100):
            chunk = todo[start:start + 100]
            for artist, picture in zip(chunk, pool.map(lambda a: lookup(a["title"]), chunk)):
                key = str(artist["ckan_id"])
                if picture:
                    found[key] = picture
                else:
                    missing.add(key)
            save()
            print(f"{start + len(chunk)}/{len(todo)} looked up, {len(found)} pictures", flush=True)
    save()
    print(f"done: {len(found)} of {len(artists)} artists have a picture", flush=True)


if __name__ == "__main__":
    sys.exit(main())
