# test.py

from ForestDiffusion import ForestDiffusionModel
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

import matplotlib

matplotlib.use("Agg")  # falls irgendwo Plots erzeugt werden

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
    # Time delta since last transaction
    # Abstand berechnen
    df['time_since_last_tx'] = (
        df.groupby(['bank_account_uuid', 'ref_iban'])['date_post']
        .diff().dt.days
    )
    df.drop("date_post", axis=1, inplace=True)

    # NaN durch den Wert 30 ersetzen
    df['time_since_last_tx'] = df['time_since_last_tx'].fillna(30)
    return df

def load_business_dataset(max_rows=1000):
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data" / "business_dataset.csv"

    df = pd.read_csv(file_path)

    target_col = "anomaly_description"


    # Anomalie: 1 = es gibt eine Beschreibung, 0 = keine
    anomaly_mask = df[target_col].notna()
    df[target_col] = anomaly_mask.astype(int)

    df = create_time_series_features(df)
    y = df[target_col].astype(int).values
    X_raw = df.drop(columns=[target_col])

    # One-Hot-Encoding aller kategorialen Spalten
    # (numerische Spalten bleiben numerisch)
    X_dummies = pd.get_dummies(X_raw, drop_first=False)

    print(f"Feature-Shape nach One-Hot-Encoding: {X_dummies.shape}")
    print(f"Anzahl Anomalien (y=1): {y.sum()} von {len(y)}")

    return X_dummies, y

def main():

    X_df, y = load_business_dataset(max_rows=700)
    X = X_df.values  # numpy-Array

    # Train/Test-Split (stratifiziert, weil binäres Label)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
   # ============================================================

#achtung hier werden die anomalien ins test set reingemischt


# ============================================================
    df_train = pd.concat([pd.DataFrame(X_train).reset_index(drop=True),
                pd.Series(y_train, name="target")], axis=1)

    df_test = pd.concat([pd.DataFrame(X_test).reset_index(drop=True),
                pd.Series(y_test, name="target")], axis=1)

    df_train_anomalies = df_train[df_train["target"] == 1]

    df_test_augmented = pd.concat([df_test, df_train_anomalies], ignore_index=True)

    df_train_reduced = df_train[df_train["target"] != 1]

    X_train = df_train_reduced.drop(columns=["target"]).to_numpy()
    y_train = df_train_reduced["target"].to_numpy()
    # ============================================================
# ============================================================


    X_test = df_test_augmented.drop(columns=["target"]).to_numpy()
    y_test = df_test_augmented["target"].to_numpy()

    print(f"Train-Shape: {X_train.shape}, Test-Shape: {X_test.shape}")

    n_features = X_train.shape[1]
    int_indexes = list(range(n_features)) 
    not_int = [13,14,15,18] # alle Spalten numerisch
    for i in not_int:
        int_indexes.remove(i)
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
    print("Starte Training des Forest Diffusion Modells...")

    model = ForestDiffusionModel(
        X=X_train,
        label_y=y_train,
        n_t=10,
        duplicate_K=5,
        diffusion_type='flow',
        n_batch=32,
        seed=666,
        n_jobs=4,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=list(range(n_features)),
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
