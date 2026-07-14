"""
pipeline.py
─────────────────────────────────────────────────────────────────
Backward-compatible wrapper execution script pointing to the refactored
scripts/run_pipeline.py pipeline runner.
"""

import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import main

if __name__ == "__main__":
    main()