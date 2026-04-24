"""
Endee vector store wrapper for CineMatch.
Handles: index creation, movie upsertion, semantic search.
"""

import os
from endee import Endee
from data.movies import MOVIES
from cinematch.embedder import embed_text, embed_batch

INDEX_NAME = "cinematch_movies"
DIMENSION = 384  # all-MiniLM-L6-v2


def get_client() -> Endee:
    token = os.getenv("ENDEE_TOKEN", "")
    if token:
        return Endee(token=token)
    return Endee()  # no-auth local dev mode


def setup_index(force_recreate: bool = False) -> None:
    """Create Endee index and upsert all movies. Skip if already populated."""
    client = get_client()
    existing = client.list_indexes()
    existing_names = [idx.get("name") for idx in (existing or [])]

    if INDEX_NAME in existing_names:
        if not force_recreate:
            print(f"Index '{INDEX_NAME}' already exists. Skipping setup.")
            return
        print(f"Recreating index '{INDEX_NAME}'...")
        try:
            client.delete_index(name=INDEX_NAME)
        except Exception:
            pass

    print(f"Creating index '{INDEX_NAME}' (dim={DIMENSION}, cosine)...")
    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
    )

    index = client.get_index(name=INDEX_NAME)

    print(f"Embedding and upserting {len(MOVIES)} movies...")
    vibe_texts = [m["vibe"] for m in MOVIES]
    vectors = embed_batch(vibe_texts)

    records = []
    for movie, vec in zip(MOVIES, vectors):
        records.append({
            "id": movie["id"],
            "vector": vec,
            "meta": {
                "title": movie["title"],
                "year": movie["year"],
                "genres": ", ".join(movie["genres"]),
                "director": movie["director"],
                "rating": movie["rating"],
                "vibe": movie["vibe"],
                "mood_tags": ", ".join(movie["mood_tags"]),
                "runtime_min": movie["runtime_min"],
            },
            "filter": {
                "genres": movie["genres"][0],  # primary genre for filtering
            }
        })

    index.upsert(records)
    print("Endee index ready.")


def semantic_search(
    query: str,
    top_k: int = 5,
    genre_filter: str = None,
) -> list:
    """
    Search movies by free-text query using vector similarity.
    Optionally filter by primary genre.
    Returns list of movie metadata dicts with scores.
    """
    client = get_client()
    index = client.get_index(name=INDEX_NAME)

    query_vec = embed_text(query)

    search_kwargs = dict(vector=query_vec, top_k=top_k)
    if genre_filter:
        search_kwargs["filter"] = {"genres": {"$eq": genre_filter}}

    results = index.query(**search_kwargs)

    hits = []
    for r in (results or []):
        meta = r.get("meta", {})
        hits.append({
            "id": r.get("id"),
            "score": round(r.get("score", 0), 4),
            "title": meta.get("title"),
            "year": meta.get("year"),
            "genres": meta.get("genres"),
            "director": meta.get("director"),
            "rating": meta.get("rating"),
            "vibe": meta.get("vibe"),
            "mood_tags": meta.get("mood_tags"),
            "runtime_min": meta.get("runtime_min"),
        })
    return hits
