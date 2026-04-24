"""
CineMatch Agentic AI — Claude-powered mood parser + recommendation engine.

Flow:
  1. User types natural language mood/vibe request
  2. Claude (agent) extracts structured intent: mood keywords, genres, runtime prefs
  3. Endee semantic search retrieves top candidates
  4. Claude synthesizes results into a warm, personalized recommendation
"""

import os
import json
import anthropic
from cinematch.vector_store import semantic_search

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

INTENT_SYSTEM_PROMPT = """You are CineMatch's intent parser.

Given a user's movie request (natural language, mood-based, or specific),
extract structured search intent as JSON.

Return ONLY valid JSON, no markdown, no explanation:
{
  "search_query": "<rich natural language vibe query for semantic search>",
  "genre_hint": "<single primary genre or null>",
  "mood_summary": "<1 sentence describing the mood they want>",
  "exclude_dark": <true if user wants to avoid dark/disturbing content>,
  "wants_short": <true if user prefers shorter films>,
  "top_k": <integer 3-8, how many candidates to retrieve>
}

Genre must be one of: action, animation, comedy, crime, drama, fantasy, horror,
musical, mystery, romance, sci-fi, thriller — or null.

Be generous with search_query — expand the user's request into a rich vibe description."""

RECOMMENDATION_SYSTEM_PROMPT = """You are CineMatch, a warm and knowledgeable film companion.

Your personality: enthusiastic but never sycophantic, specific not vague,
you give reasons not just lists, you speak like a friend who watches a lot of films.

Given the user's mood request and a list of candidate movies retrieved from
semantic search, write a personalized recommendation response.

Rules:
- Lead with the BEST match (highest fit), explain WHY it matches their mood
- Mention 2-3 alternatives briefly
- Keep total response under 350 words
- Use the movie's vibe description to tailor your explanation
- If candidates don't match well, say so honestly and suggest adjusting the query
- Format: plain conversational text, no excessive bullets or headers
- End with one fun question to help refine the search further"""


def parse_intent(user_message: str) -> dict:
    """Use Claude to extract structured search intent from natural language."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=INTENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "search_query": user_message,
            "genre_hint": None,
            "mood_summary": user_message,
            "exclude_dark": False,
            "wants_short": False,
            "top_k": 5,
        }


def build_candidate_context(candidates: list) -> str:
    """Format retrieved movies for Claude's recommendation prompt."""
    lines = []
    for i, m in enumerate(candidates, 1):
        lines.append(
            f"{i}. {m['title']} ({m['year']}) — {m['genres']} | "
            f"Rating: {m['rating']} | Runtime: {m['runtime_min']}min\n"
            f"   Director: {m['director']}\n"
            f"   Vibe: {m['vibe']}\n"
            f"   Mood tags: {m['mood_tags']}\n"
            f"   Similarity score: {m['score']}"
        )
    return "\n\n".join(lines)


def recommend(user_message: str, chat_history: list = None) -> dict:
    """
    Full agentic pipeline:
    user message -> intent parse -> Endee search -> Claude recommendation

    Returns:
        {
            "intent": {...},
            "candidates": [...],
            "recommendation": "...",
        }
    """
    # Step 1: Parse intent
    intent = parse_intent(user_message)

    # Step 2: Semantic search in Endee
    candidates = semantic_search(
        query=intent["search_query"],
        top_k=intent.get("top_k", 5),
        genre_filter=intent.get("genre_hint"),
    )

    if not candidates:
        return {
            "intent": intent,
            "candidates": [],
            "recommendation": (
                "Hmm, I couldn't find strong matches in my collection for that. "
                "Try describing a feeling or situation instead — like 'something cozy for a rainy Sunday' "
                "or 'I want a film that will wreck me emotionally'."
            ),
        }

    # Step 3: Filter dark content if requested
    if intent.get("exclude_dark"):
        dark_tags = {"dark", "unsettling", "intense", "horror"}
        candidates = [
            c for c in candidates
            if not dark_tags.intersection(set(c.get("mood_tags", "").split(", ")))
        ][:intent.get("top_k", 5)]

    # Step 4: Sort by runtime if user wants short films
    if intent.get("wants_short"):
        candidates = sorted(candidates, key=lambda x: x.get("runtime_min", 999))

    # Step 5: Build recommendation with Claude
    candidate_context = build_candidate_context(candidates)

    messages = []
    if chat_history:
        for turn in chat_history[-6:]:  # last 3 turns for context
            messages.append(turn)

    messages.append({
        "role": "user",
        "content": (
            f"User's request: {user_message}\n\n"
            f"Their mood (extracted): {intent.get('mood_summary')}\n\n"
            f"Retrieved movie candidates from semantic search:\n\n"
            f"{candidate_context}"
        ),
    })

    rec_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=RECOMMENDATION_SYSTEM_PROMPT,
        messages=messages,
    )

    recommendation = rec_response.content[0].text.strip()

    return {
        "intent": intent,
        "candidates": candidates,
        "recommendation": recommendation,
    }
