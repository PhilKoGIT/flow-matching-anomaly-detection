
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

df = df.sort_values(["bank_account_uuid", "date_post"])

df['amount_mean_5'] = df.groupby('bank_account_uuid')['amount'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['amount_std_5'] = df.groupby('bank_account_uuid')['amount'].transform(lambda x: x.rolling(5, min_periods=1).std())
df['amount_change'] = df.groupby('bank_account_uuid')['amount'].diff()

 # Beispiel für kategorische Merkmale: Empfängerwechsel
df['receiver_changed'] = (df.groupby('bank_account_uuid')['ref_name']
                             .apply(lambda x: x != x.shift())).astype(int)

# Zeitliche Abstände
df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
df['time_since_last_tx'] = df.groupby('bank_account_uuid')['date_post'].diff().dt.days

print(df.columns)
print(df.info())