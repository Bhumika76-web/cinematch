"""
Standalone script to initialize Endee index.
Run once before starting the app:
  python scripts/setup_db.py
  python scripts/setup_db.py --force   <- recreate from scratch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from cinematch.vector_store import setup_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    setup_index(force_recreate=force)
    print("Done!")
