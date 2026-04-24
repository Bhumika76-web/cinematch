"""
CLI demo — test semantic search directly without the UI.
  python scripts/search_demo.py "cozy rainy day film"
  python scripts/search_demo.py "something terrifying" --genre=horror
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from cinematch.vector_store import semantic_search, setup_index

if __name__ == "__main__":
    query = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    genre = None
    for a in sys.argv[1:]:
        if a.startswith("--genre="):
            genre = a.split("=")[1]

    if not query:
        query = "something emotional and beautiful"

    print(f"\nQuery: '{query}' | Genre filter: {genre or 'none'}\n")
    setup_index()
    results = semantic_search(query, top_k=5, genre_filter=genre)

    for r in results:
        print(f"  [{r['score']}] {r['title']} ({r['year']}) — {r['genres']}")
        vibe_preview = r['vibe'][:80] if r['vibe'] else ''
        print(f"          {vibe_preview}...")
        print()
