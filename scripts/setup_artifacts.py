"""
scripts/setup_artifacts.py
─────────────────────────────────────────────────────────────────
Creates the artifacts/ directory and copies the trained model files
from the root directory if they exist.
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"


def setup():
    print("Setting up directory structure...")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Created: {ARTIFACTS_DIR}")
    print(f"Created: {DATA_DIR}")

    # Copy files if they exist in the root
    files_to_move = ["model.pkl", "scaler.pkl", "feature_cols.pkl"]
    for file_name in files_to_move:
        src = ROOT / file_name
        dest = ARTIFACTS_DIR / file_name
        if src.exists() and not dest.exists():
            print(f"Copying {file_name} from root to artifacts/ ...")
            shutil.copy2(src, dest)

    # Move creditcard.csv if it exists in root
    csv_src = ROOT / "creditcard.csv"
    csv_dest = DATA_DIR / "creditcard.csv"
    if csv_src.exists() and not csv_dest.exists():
        print("Moving creditcard.csv from root to data/ ...")
        shutil.move(str(csv_src), str(csv_dest))

    print("Setup completed successfully.")


if __name__ == "__main__":
    setup()
