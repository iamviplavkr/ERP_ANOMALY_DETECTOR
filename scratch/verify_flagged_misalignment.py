import pandas as pd
import numpy as np

# Load flagged_transactions.csv
flagged = pd.read_csv('flagged_transactions.csv')
print("Columns in flagged_transactions.csv:", flagged.columns.tolist())
print(f"Total rows in flagged_transactions.csv: {len(flagged)}")

if 'is_fraud' in flagged.columns:
    print("\nActual fraud status ('is_fraud' column from df_test) of rows in flagged_transactions.csv:")
    print(flagged['is_fraud'].value_counts())
    
    fraud_in_flagged = flagged[flagged['is_fraud'] == 1]
    normal_in_flagged = flagged[flagged['is_fraud'] == 0]
    
    print(f"\nNumber of actual non-fraud (Class=0) rows incorrectly saved as flagged: {len(normal_in_flagged)}")
    print(f"Number of actual fraud (Class=1) rows saved as flagged: {len(fraud_in_flagged)}")
    
    print("\nSample of actual non-fraud transactions in flagged_transactions.csv (their assigned anomaly score vs actual is_fraud):")
    print(normal_in_flagged[['transaction_amount', 'anomaly_score', 'is_fraud']].head(10))

# Now let's check what happened to the real fraud transactions in the tail slice df.iloc[len(X_train):]
df_raw = pd.read_csv('data/creditcard.csv')
X_train_len = int(len(df_raw) * 0.8)
df_tail = df_raw.iloc[X_train_len:].reset_index(drop=True)

fraud_tail_indices = df_tail[df_tail['Class'] == 1].index.tolist()
print(f"\nNumber of actual fraud cases in df_tail (df.iloc[len(X_train):]): {len(fraud_tail_indices)}")
