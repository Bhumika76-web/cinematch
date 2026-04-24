"""
CineMatch — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from cinematch.vector_store import setup_index
from cinematch.agent import recommend

load_dotenv()

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stChatMessage { border-radius: 12px; }
.stButton > button { border-radius: 8px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── Init vector store (once per session) ──────────────────────────────────
@st.cache_resource(show_spinner="Setting up CineMatch vector store...")
def init():
    setup_index()
    return True

init()

# ─── Session state ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_candidates" not in st.session_state:
    st.session_state.last_candidates = []

# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎬 CineMatch")
    st.markdown("**AI-powered mood-based movie finder**")
    st.divider()
    st.markdown("""
**How it works:**
1. Describe your mood, vibe, or situation
2. Claude parses your intent
3. Endee searches 50 films by semantic similarity
4. Claude recommends with personal reasoning
    """)
    st.divider()
    st.markdown("**Try these prompts:**")

    example_prompts = [
        "Cozy film for a rainy Sunday",
        "Mind-bending sci-fi that will haunt me",
        "Something funny but also emotional",
        "Film for me and my 8-year-old",
        "Romantic but not cheesy",
        "Feeling nihilistic, help me",
        "Intense thriller under 2 hours",
        "Visually stunning, I can zone out",
    ]
    for p in example_prompts:
        if st.button(p, key=p, use_container_width=True):
            st.session_state.pending_prompt = p

    st.divider()
    if st.button("🔄 Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_candidates = []
        st.rerun()

# ─── Main UI ───────────────────────────────────────────────────────────────
st.markdown("## What kind of movie are you in the mood for?")
st.caption("Describe a feeling, vibe, situation, or genre — the more specific, the better.")

# Chat history display
for turn in st.session_state.chat_history:
    role = turn["role"]
    content = turn["content"]
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant", avatar="🎬"):
            st.write(content)

# Last candidates panel
if st.session_state.last_candidates:
    with st.expander("🔍 Movies retrieved from Endee vector search", expanded=False):
        num_cols = min(3, len(st.session_state.last_candidates))
        cols = st.columns(num_cols)
        for i, m in enumerate(st.session_state.last_candidates[:6]):
            with cols[i % num_cols]:
                st.markdown(f"**{m['title']}** ({m['year']})")
                st.markdown(f"⭐ {m['rating']} · {m['genres']}")
                st.markdown(f"🎯 Similarity: `{m['score']}`")
                vibe_preview = m['vibe'][:100] + "..." if m['vibe'] and len(m['vibe']) > 100 else m['vibe']
                st.markdown(f"_{vibe_preview}_")
                st.divider()

# ─── Input handling ────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Type your mood or vibe...") or pending

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    claude_history = [
        t for t in st.session_state.chat_history
        if t["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant", avatar="🎬"):
        with st.spinner("Searching the vibe space..."):
            result = recommend(user_input, chat_history=claude_history)

        rec_text = result["recommendation"]
        st.write(rec_text)

        intent = result["intent"]
        st.caption(
            f"🧠 Parsed intent: _{intent.get('mood_summary')}_ "
            f"| Genre hint: `{intent.get('genre_hint') or 'any'}` "
            f"| Query: _{str(intent.get('search_query', ''))[:60]}..._"
        )

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": rec_text})
    st.session_state.last_candidates = result["candidates"]
    st.rerun()
