import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

root_model_path = ROOT / 'model.pkl'
art_model_path = ROOT / 'artifacts' / 'model.pkl'
root_scaler_path = ROOT / 'scaler.pkl'
art_scaler_path = ROOT / 'artifacts' / 'scaler.pkl'
root_cols_path = ROOT / 'feature_cols.pkl'
art_cols_path = ROOT / 'artifacts' / 'feature_cols.pkl'

print("--- Root vs Artifacts Files ---")
print(f"Root model size: {root_model_path.stat().st_size if root_model_path.exists() else 'N/A'}")
print(f"Art model size:  {art_model_path.stat().st_size if art_model_path.exists() else 'N/A'}")

with open(root_model_path, 'rb') as f:
    m_root = pickle.load(f)
with open(art_model_path, 'rb') as f:
    m_art = pickle.load(f)

with open(root_scaler_path, 'rb') as f:
    s_root = pickle.load(f)
with open(art_scaler_path, 'rb') as f:
    s_art = pickle.load(f)

with open(root_cols_path, 'rb') as f:
    c_root = pickle.load(f)
with open(art_cols_path, 'rb') as f:
    c_art = pickle.load(f)

print(f"Columns equal? {c_root == c_art}")
print(f"Scaler mean equal? {np.allclose(s_root.mean_, s_art.mean_)}")
print(f"Scaler scale equal? {np.allclose(s_root.scale_, s_art.scale_)}")

print(f"\nRoot model n_estimators: {m_root.n_estimators}, max_depth: {m_root.max_depth}, class_weight: {m_root.class_weight}")
print(f"Art model n_estimators:  {m_art.n_estimators}, max_depth: {m_art.max_depth}, class_weight: {m_art.class_weight}")

# Check feature importances comparison
print(f"Feature importances equal? {np.allclose(m_root.feature_importances_, m_art.feature_importances_)}")
print("Max diff in feature importances:", np.max(np.abs(m_root.feature_importances_ - m_art.feature_importances_)))
