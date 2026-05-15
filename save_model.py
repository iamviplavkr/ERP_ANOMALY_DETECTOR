"""
Run this ONCE before starting api.py.
It trains the model and saves model.pkl, scaler.pkl, feature_cols.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('creditcard.csv')
df = df.rename(columns={
    'Time':   'posting_time',
    'Amount': 'transaction_amount',
    'Class':  'is_fraud'
})

# Feature Engineering
df['log_amount']    = np.log1p(df['transaction_amount'])
df['hour_of_day']   = (df['posting_time'] % 86400) // 3600
df['is_night']      = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
df['amount_zscore'] = (df['transaction_amount'] - df['transaction_amount'].mean()) / df['transaction_amount'].std()

feature_cols = [c for c in df.columns if c.startswith('V')] + \
               ['log_amount', 'hour_of_day', 'is_night', 'amount_zscore']

X = df[feature_cols].values
y = df['is_fraud'].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Save artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("Saved: model.pkl, scaler.pkl, feature_cols.pkl")
print("Now run: uvicorn api:app --reload")