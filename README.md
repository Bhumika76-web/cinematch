# CineMatch — Mood-Based AI Movie Recommender

> **Find your next film by describing a feeling, not a title.**

CineMatch is an agentic AI system that recommends movies based on mood, vibe, or situation descriptions. You type something like *"something cozy for a rainy Sunday"* or *"I want a film that will destroy me emotionally"* — and Claude + Endee figure out exactly what you need.

---

## What Makes It Unique

Unlike simple search or basic RAG, CineMatch chains **three AI tasks** in one pipeline:

1. **Agentic Intent Parsing** — Claude reads your natural language mood description and extracts structured search parameters (vibe keywords, genre hints, content preferences)
2. **Semantic Vector Search** — Endee retrieves the most vibe-similar movies using dense vector embeddings of film personality descriptions
3. **Personalized Recommendation** — Claude synthesizes the results into a warm, reasoned recommendation with context from your conversation history

---

## System Design

```
User Input (natural language mood)
        |
        v
+-------------------+
|  Claude (Agent)   |  <- Intent parsing: extracts structured vibe query,
|  Intent Parser    |     genre hint, content flags
+--------+----------+
         |  structured intent
         v
+-------------------+
|  Sentence         |  <- Embeds the extracted vibe query into 384-dim
|  Transformers     |     dense vector (all-MiniLM-L6-v2)
+--------+----------+
         |  query vector
         v
+-------------------+
|  Endee Vector DB  |  <- Cosine similarity search over 50 indexed movies
|  Semantic Search  |     with optional payload/genre filtering
+--------+----------+
         |  top-K candidates + metadata
         v
+-------------------+
|  Claude (Agent)   |  <- Synthesizes candidates into personalized
|  Recommender      |     recommendation with conversation history
+--------+----------+
         |  recommendation text
         v
    Streamlit UI / CLI
```

### Component Roles

| Component | Role |
|---|---|
| `data/movies.py` | 50-movie dataset with rich vibe descriptions and mood tags |
| `cinematch/embedder.py` | Sentence-Transformers wrapper (all-MiniLM-L6-v2, 384-dim) |
| `cinematch/vector_store.py` | Endee client: index setup, upsert, semantic search |
| `cinematch/agent.py` | Claude agentic pipeline: intent parse -> search -> recommend |
| `app.py` | Streamlit chat UI with multi-turn conversation |
| `scripts/` | CLI demos and setup utilities |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/cinematch
cd cinematch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add:
#   ANTHROPIC_API_KEY=your_key
#   ENDEE_TOKEN=your_endee_token   (leave blank for local no-auth mode)
```

**Get Endee token:** Sign up at [endee.io](https://endee.io) — free Starter tier available.
**Get Anthropic key:** [console.anthropic.com](https://console.anthropic.com)

### 4. Initialize the vector store

```bash
python scripts/setup_db.py
```

This embeds all 50 movies and upserts them into Endee. One-time setup (~30 seconds).

### 5. Run

**Streamlit UI:**
```bash
streamlit run app.py
```

**CLI demo (no UI):**
```bash
python scripts/agent_demo.py
```

**Pure semantic search test:**
```bash
python scripts/search_demo.py "something cozy for a rainy Sunday"
python scripts/search_demo.py "terrifying and atmospheric" --genre=horror
```

---

## Example Interactions

| User says | Claude recommends |
|---|---|
| "cozy rainy Sunday film" | Paddington 2, Spirited Away, About Time |
| "mind-bending but emotional" | Everything Everywhere, Arrival, Your Name |
| "scared but not gory" | The Witch, The Lighthouse, Hereditary |
| "funny for a bad day" | Hot Fuzz, Hunt for the Wilderpeople, Office Space |
| "epic adventure for me and my 10-year-old" | Castle in the Sky, Princess Mononoke, Coco |

---

## Endee Usage

CineMatch uses the following Endee operations:

```python
# Create index
client.create_index(name="cinematch_movies", dimension=384, space_type="cosine")

# Upsert movies with metadata + filter fields
index.upsert([{
    "id": "m001",
    "vector": [...],   # 384-dim sentence embedding of vibe description
    "meta": {"title": "Interstellar", "year": 2014, ...},
    "filter": {"genres": "sci-fi"}
}])

# Semantic search with optional genre filtering
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"genres": {"$eq": "horror"}}  # optional
)
```

---

## Project Structure

```
cinematch/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── __init__.py
│   └── movies.py           # 50-movie dataset
├── cinematch/
│   ├── __init__.py
│   ├── embedder.py         # Sentence-Transformers wrapper
│   ├── vector_store.py     # Endee client and operations
│   └── agent.py            # Claude agentic pipeline
└── scripts/
    ├── setup_db.py         # One-time index setup
    ├── search_demo.py      # CLI semantic search test
    └── agent_demo.py       # CLI full pipeline demo
```

---

## AI/ML Concepts Demonstrated

- **Semantic search** — embedding-based similarity over natural language vibe descriptions
- **RAG** — retrieval-augmented generation: search then synthesize
- **Agentic AI** — multi-step Claude pipeline with structured output
- **Hybrid filtering** — vector search + metadata payload filtering in Endee
- **Multi-turn conversation** — chat history passed to recommendation step

---

## Possible Extensions

- Add TMDB API integration for posters and real-time movie data
- Expand dataset to 500+ films
- Add hybrid sparse+dense search (Endee supports this natively)
- User history tracking — don't recommend films already seen
- Mood-to-playlist: chain movie -> soundtrack recommendations

---

## License

MIT
