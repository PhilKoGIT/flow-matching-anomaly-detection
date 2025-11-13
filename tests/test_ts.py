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

def hashers(df):
    # Hashing der ref_iban Spalte
    hasher_iban = FeatureHasher(input_type='string', n_features=10)
    hashed_features = hasher_iban.transform(df['ref_iban'].astype(str).values.reshape(-1,1))
    hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'iban_hash_{i}' for i in range(10)])
    df = pd.concat([hashed_df, df], axis=1)
    df = df.drop("ref_iban", axis=1)

    # Hashing der ref_swift Spalte
    hasher_swift = FeatureHasher(input_type='string', n_features=10)
    hashed_features_swift = hasher_swift.transform(df['ref_swift'].astype(str).values.reshape(-1,1))
    hashed_df_swift = pd.DataFrame(hashed_features_swift.toarray(), columns=[f'swift_hash_{i}' for i in range(10)])
    df = pd.concat([hashed_df_swift, df], axis=1)
    df = df.drop("ref_swift", axis=1)

    # Hashing der paym_note Spalte
    hasher_paym_note = FeatureHasher(input_type='string', n_features=10)
    hashed_paym_note = hasher_paym_note.transform(df['paym_note'].astype(str).values.reshape(-1,1))
    hashed_df_payment_note = pd.DataFrame(hashed_paym_note.toarray(), columns=[f'paym_note{i}' for i in range(10)])
    df = pd.concat([hashed_df_payment_note, df], axis=1)
    df = df.drop("paym_note", axis=1)


    return df


def create_time_series_features(df):
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
    file_path = base_dir.parent / "data" / "1311dataset.csv"
    df = pd.read_csv(file_path)

    return df

def main():

    df = load_business_dataset()
    df = create_time_series_features(df)
    print(df.head())
    columns_to_drop = ["bank_account_uuid", "ref_bank", "ref_name", "currency", "trns_type"]
    df = df.drop(columns=columns_to_drop)

    #nochmal mischen
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sort_values(["date_post"])
    df.drop("date_post", axis=1, inplace=True)
    df = hashers(df)
    print(df.head())

    # anomaly_description contains 0 or 1
    target_col = "anomaly_description"
    anomaly_mask = df[target_col].notna()
    df[target_col] = anomaly_mask.astype(int)


    
    print(df.isna().any().any())


    # Train/Test-Split by time (letzte 20% der Daten als Test-Set)
    split_index = int(0.8 * len(df))

    y = df[target_col].astype(int).values
    for c in ["business_partner_name", "pay_method", "channel"]:
        df[c] = pd.factorize(df[c])[0]  # oder .astype("category").cat.codes
    X = df.drop(columns=[target_col]).values

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]  


    int_index = [
    0,1,2,3,4,5,6,7,8,9,       # paym_note*
    10,11,12,13,14,15,16,17,18,19,   # swift_hash_*
    20,21,22,23,24,25,26,27,28,29,   # iban_hash_*
    37,  # month
    38,  # dayofweek
    39   # year
    ]
    cat_index = [
    30,   # business_partner_name
    32,   #pay_method
    33,    # channel
    ]
    bin_index = []

    # model = ForestDiffusionModel(
    #     X=X_train,
    #     label_y=y_train,
    #     # Diffusion / Training
    #     n_t=30,                 # weniger Zeitschritte für schnellere Tests
    #     duplicate_K=10,          # weniger Duplikate für schnellere Tests
    #     diffusion_type='flow',  # für Zero-Shot-Klassifikation ok
    #     n_batch=64,
    #     seed=666,
    #     n_jobs=-1,

    #     # Spaltentypen: alles numerisch
    #     bin_indexes=[],                     # optional könntest du hier echte 0/1-Spalten eintragen
    #     cat_indexes=[],                     # WICHTIG: keine Kategorien mehr, alles schon One-Hot
    #     int_indexes=int_indexes,

   
    #     max_depth=8,
    #     n_estimators=150,
    #     gpu_hist=True,          # nur aktiv lassen, wenn du wirklich eine GPU+CUDA hast

    #     # Sonstiges
    #     remove_miss=False,
    #     p_in_one=True,
    # )

# ZIIEELL ist einziges binär


    model = ForestDiffusionModel(
        X=X_train,
        label_y=y_train,
        n_t=10,
        duplicate_K=5,
        diffusion_type='flow',
        n_batch=32,
        seed=666,
        n_jobs=4,
        bin_indexes= bin_index,
        cat_indexes= cat_index,
        int_indexes=int_index,
        max_depth=4,
        n_estimators=20,
        gpu_hist=True,
        remove_miss=False,
        p_in_one=True,
    )

    

    # Modell speichern
    dump(model, "forest_diffusion_model.joblib")



    y_pred = model.predict(X_test, n_t=10, n_z=5)

    print("\nKlassifikationsreport (0=normal, 1=Anomalie):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Konfusionsmatrix:")
    print(confusion_matrix(y_test, y_pred))


    n_anomalies_true = (y_test == 1).sum()
    n_anomalies_pred = (y_pred == 1).sum()

    print(f"\nWahre Anomalien im Test-Set: {n_anomalies_true} von {len(y_test)}")
    print(f"Vom Modell als Anomalie (1) vorhergesagt: {n_anomalies_pred} von {len(y_test)}")


    y_probs = model.predict_proba(X_test, n_t=10, n_z=5)
    y_probs = np.asarray(y_probs)
    print("\nWahrscheinlichkeiten-Array Form:", y_probs.shape)


if __name__ == "__main__":
    main()
