
from ForestDiffusion import ForestDiffusionModel
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

import matplotlib

matplotlib.use("Agg")  # falls irgendwo Plots erzeugt werden



base_dir = Path(__file__).resolve().parent
file_path = base_dir.parent / "data" / "business_dataset.csv"

df = pd.read_csv(file_path)

df.head()

# Sort by account and date first
df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
df = df.sort_values(["bank_account_uuid", "date_post"])

# Calculate rolling features
df['amount_mean_5'] = df.groupby('bank_account_uuid')['amount'] \
                        .transform(lambda x: x.rolling(5, min_periods=1).mean())
df['amount_std_5'] = df.groupby('bank_account_uuid')['amount'] \
                       .transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
df['amount_change'] = df.groupby('bank_account_uuid')['amount'].diff()
df['amount_change'] = df['amount_change'].fillna(0)
# ✅ Correct receiver change flag (no MultiIndex)


df['month'] = df['date_post'].dt.month
df['dayofweek'] = df['date_post'].dt.dayofweek
# Time delta since last transaction
# Abstand berechnen
df['time_since_last_tx'] = (
    df.groupby(['bank_account_uuid', 'ref_iban'])['date_post']
      .diff().dt.days
)
df.drop("date_post", axis=1, inplace=True)
# globalen Mittelwert über alle gültigen Werte berechnen

# NaN durch den Mittelwert ersetzen
df['time_since_last_tx'] = df['time_since_last_tx'].fillna(30)


print(df.columns)
#gib mir die anomalien aus
anomalies = df[df['anomaly_description'].notna()]
#wie viele unique values hat jede spalte
for col in df.columns:
    print(col, df[col].nunique())
