"""
CLI demo — full agentic pipeline without Streamlit.
  python scripts/agent_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from cinematch.vector_store import setup_index
from cinematch.agent import recommend

BANNER = """
+==========================================+
|  CineMatch -- CLI Mood Movie Finder      |
|  Type 'quit' to exit                     |
+==========================================+
"""

if __name__ == "__main__":
    print(BANNER)
    setup_index()
    history = []

    while True:
        try:
            user_input = input("\nYour mood/vibe: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not user_input:
            continue

        print("\nSearching...\n")
        result = recommend(user_input, chat_history=history)

        print("-" * 60)
        print(result["recommendation"])
        print("-" * 60)

        intent = result["intent"]
        print(f"\n[debug] mood: {intent.get('mood_summary')}")
        query_preview = str(intent.get('search_query', ''))[:70]
        print(f"[debug] search_query: {query_preview}")
        print(f"[debug] candidates retrieved: {len(result['candidates'])}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result["recommendation"]})
