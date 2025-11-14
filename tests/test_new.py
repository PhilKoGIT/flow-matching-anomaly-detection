# test.py

from ForestDiffusion import ForestDiffusionModel
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction import FeatureHasher

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

import matplotlib

matplotlib.use("Agg")  # falls irgendwo Plots erzeugt werden

def uniqueness(df):
    df['account_partner_id'] = df['bank_account_uuid'].astype(str) + "_" + df['business_partner_name'].astype(str)
    df = df.drop(columns=['bank_account_uuid', 'business_partner_name'])

    combo_cols = ["ref_name", "ref_iban", "ref_swift", "pay_method", "channel", "currency", "trns_type"]

    for col in range(len(combo_cols)):
        combo_observed = combo_cols[:col+1]
        combo_counts = (
            df.groupby(combo_observed, dropna=False)
            .size()
            .reset_index(name=f'combination_freq_{col+1}')
        )
        df = df.merge(combo_counts, on=combo_observed, how='left')

    k = len(combo_cols)
    freq_cols = [f'combination_freq_{i}' for i in range(1, k + 1)]

    all_equal = (df[freq_cols].nunique(axis=1, dropna=False) == 1)

    at_least_one_one = (df[freq_cols] == 1).any(axis=1)

    df['valid_ref'] = (
        all_equal |
        (~all_equal & ~at_least_one_one)
    ).astype(int)

    #drop combo columns
    df = df.drop(columns=combo_cols + freq_cols)
    df = df.drop(columns = ["ref_bank", "paym_note"])
    print(df.head())
    print(df.columns)
    return df

def create_time_series_features(df):
    # Sort by account and date first
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values(["bank_account_uuid", "date_post"])

    # Calculate rolling features
    mean_rolling = lambda x: x.rolling(5, min_periods=1).mean()
    std_rolling = lambda x: x.rolling(5, min_periods=1).std()
    df['amount_mean_5'] = df.groupby('bank_account_uuid')['amount'].transform(mean_rolling)
    df['amount_std_5'] = df.groupby('bank_account_uuid')['amount'].transform(std_rolling).fillna(0)
    df['amount_change'] = df.groupby('bank_account_uuid')['amount'].diff()
    df['amount_change'] = df['amount_change'].fillna(0)
    df['month'] = df['date_post'].dt.month
    df['dayofweek'] = df['date_post'].dt.dayofweek
    df["year"] = df['date_post'].dt.year
    # Time delta since last transaction
    # Abstand berechnen
    df['time_since_last_tx'] = (
        df.groupby(['bank_account_uuid', 'ref_iban'])['date_post']
        .diff().dt.days
    )

    # NaN durch den Wert 30 ersetzen
    df['time_since_last_tx'] = df['time_since_last_tx'].fillna(30)
    return df

def load_business_dataset():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data" / "business_transactions_big.csv"
    df = pd.read_csv(file_path)

    return df

def main():

    df = load_business_dataset()
    df = create_time_series_features(df)
    print(df.head())
    #nochmal mischen
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sort_values(["date_post"])
    df.drop("date_post", axis=1, inplace=True)
    print(df.head())

    # anomaly_description contains 0 or 1
    target_col = "anomaly_description"
    anomaly_mask = df[target_col].notna()
    df[target_col] = anomaly_mask.astype(int)


    print(df.head())
    print(df.columns)
    print(df.isna().any().any())


    # Train/Test-Split by time (letzte 20% der Daten als Test-Set)
    split_index = int(0.8 * len(df))

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]  

    X_train = uniqueness(X_train)
    X_test = uniqueness(X_test)


# 1. Nur die Features im Training kodieren und die Abbildung speichern
    codes, uniques = pd.factorize(X_train["account_partner_id"])
    X_train["account_partner_id"] = codes

    # 2. Test-Set mit der im Training gelernten Abbildung kodieren
    # Unbekannte Werte werden standardmäßig zu -1 (oder NaN/spez. Code)
    X_test["account_partner_id"] = pd.Categorical(
        X_test["account_partner_id"], categories=uniques
    ).codes
    print(X_train.columns)

    y_train = y_train.values
    y_test = y_test.values

    int_index = [4,5,6,8]
    bin_index = [9]
    cat_index = [7]
    model = ForestDiffusionModel(
        X=X_train,
       # label_y=y_train,
        # Diffusion / Training
        n_t=30,                 # weniger Zeitschritte für schnellere Tests
        duplicate_K=10,          # weniger Duplikate für schnellere Tests
        diffusion_type='flow',  # für Zero-Shot-Klassifikation ok
        n_batch=64,
        seed=666,
        n_jobs=-1,

        # Spaltentypen: alles numerisch
        bin_indexes= bin_index,
        cat_indexes= cat_index,
        int_indexes=int_index,

   
        max_depth=8,
        n_estimators=150,
        gpu_hist=True,          # nur aktiv lassen, wenn du wirklich eine GPU+CUDA hast

        # Sonstiges
        remove_miss=False,
        p_in_one=True,
    )

    # model = ForestDiffusionModel(
    #     X=X_train,
    #     label_y=y_train,
    #     n_t=10,
    #     duplicate_K=5,
    #     diffusion_type='flow',
    #     n_batch=32,
    #     seed=666,
    #     n_jobs=4,
    #     bin_indexes= bin_index,
    #     cat_indexes= cat_index,
    #     int_indexes=int_index,
    #     max_depth=4,
    #     n_estimators=20,
    #     gpu_hist=True,
    #     remove_miss=False,
    #     p_in_one=True,
    # )

    

    # Modell speichern
    dump(model, "forest_diffusion_model.joblib")



    y_pred = model.predict(X_test, n_t=10, n_z=5)

    print("\nKlassifikationsreport (0=normal, 1=Anomalie):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Konfusionsmatrix:")
    print(confusion_matrix(y_test, y_pred))

    n_anomalies_trainingsset = (y_train == 1).sum()
    print(f"\nAnomalien im Trainings-Set: {n_anomalies_trainingsset} von {len(y_train)}")

    n_anomalies_true = (y_test == 1).sum()
    n_anomalies_pred = (y_pred == 1).sum()

    print(f"\nWahre Anomalien im Test-Set: {n_anomalies_true} von {len(y_test)}")
    print(f"Vom Modell als Anomalie (1) vorhergesagt: {n_anomalies_pred} von {len(y_test)}")
    tp = ((y_test == 1) & (y_pred == 1)).sum()
    print(f"Richtig erkannte Anomalien: {tp} von {(y_test == 1).sum()}")

    y_probs = model.predict_proba(X_test, n_t=10, n_z=5)
    y_probs = np.asarray(y_probs)
    print("\nWahrscheinlichkeiten (erste 10):", y_probs[:10])
    print(y_probs)
    print("\nWahrscheinlichkeiten-Array Form:", y_probs.shape)


if __name__ == "__main__":
    main()
