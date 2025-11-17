# test.py

from xml.parsers.expat import model
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


#edge case: one wrong ref and one correct ref in test set-> both ref_valid = 0
# was wenn die anomalie 2 mal vorkommt?!
def uniqueness(df):
    #df['account_partner_id'] = df['bank_account_uuid'].astype(str) + "_" + df['business_partner_name'].astype(str)
    df = df.drop(columns=['business_partner_name'])
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
    return df

def create_time_series_features(df):
    # Sort by account and date first
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values(["bank_account_uuid", "date_post"])

    #ref name instead of ref_iban -> bc anomalies might have change iban
    #actually also not 100% stable 
    group_cols = ['bank_account_uuid', 'ref_name']

    # Calculate rolling features
    mean_rolling = lambda x: x.rolling(5, min_periods=1).mean()
    std_rolling = lambda x: x.rolling(5, min_periods=1).std()
    df['amount_mean_5'] = df.groupby(group_cols)['amount'].transform(mean_rolling)
    df['amount_std_5']  = df.groupby(group_cols)['amount'].transform(std_rolling)
    df['amount_change'] = df.groupby(group_cols)['amount'].diff().fillna(0)
    df['amount_change'] = df['amount_change'].fillna(0)
    # Time delta since last transaction
    # Abstand berechnen
    df['time_since_last_tx'] = (
        df.groupby(group_cols)['date_post']
        .diff().dt.days
    )
    # NaN durch den Wert 30 ersetzen
    df['time_since_last_tx'] = df['time_since_last_tx'].fillna(30)
    df['day_of_month'] = df['date_post'].dt.day
    df['median_dom_per_series'] = (
        df.groupby(group_cols)['day_of_month']
        .transform('median')
    )
    df['dom_deviation'] = (df['day_of_month'] - df['median_dom_per_series']).abs()

    #for off-payment identification... no anomaly ! 
    #just one column needed, could be anything instead of amount
    df['partner_tx_count'] = (
    df.groupby(['bank_account_uuid', 'ref_name'])['amount']
        .transform('size')
    )
    return df

def load_business_dataset():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data" / "business_transactions_big.csv"
    df = pd.read_csv(file_path)
    return df

def main():

    df = load_business_dataset()
    df = create_time_series_features(df)
    #nochmal mischen
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sort_values(["date_post"])
    df.drop("date_post", axis=1, inplace=True)

    # anomaly_description contains 0 or 1
    target_col = "anomaly_description"
    anomaly_mask = df[target_col].notna()
    df[target_col] = anomaly_mask.astype(int)

    # Train/Test-Split by time (letzte 20% der Daten als Test-Set)
    split_index = int(0.8 * len(df))

    y = df[target_col]
    X = df.drop(columns=[target_col])
    # # 1. Nur die Features im Training kodieren und die Abbildung speichern
    #codes, uniques = pd.factorize(X["bank_account_uuid"])
    #X["bank_account_uuid"] = codes


    X_train, X_test = X[:split_index].copy(), X[split_index:].copy()
    y_train, y_test = y[:split_index].copy(), y[split_index:].copy()

    X_train = uniqueness(X_train)
    X_test = uniqueness(X_test)

    # Spalten, die du one-hot encoden willst
    ohe_cols = ["bank_account_uuid"]   # ggf. weitere Kategorien ergänzen

    X_train_ohe = pd.get_dummies(X_train, columns=ohe_cols)
    X_test_ohe = pd.get_dummies(X_test, columns=ohe_cols)

    # Train und Test auf dieselben Spalten bringen
    X_train_aligned, X_test_aligned = X_train_ohe.align(
        X_test_ohe,
        join="left",    # alle Spalten aus Train behalten
        axis=1,
        fill_value=0    # fehlende Spalten im Test mit 0 auffüllen
    )

    # Für spätere Index-Bestimmung merken:
    feature_cols = X_train_aligned.columns.tolist()

    X_train_np = X_train_aligned.values
    X_test_np = X_test_aligned.values
    y_train_np = y_train.values
    y_test_np = y_test.values

        # ...
    X_train_np = X_train_aligned.to_numpy(dtype=float)
    X_test_np  = X_test_aligned.to_numpy(dtype=float)

    y_train_np = y_train.to_numpy(dtype=float)
    y_test_np  = y_test.to_numpy(dtype=float)


    # Beispiel: deine ursprünglichen numerischen Spalten
    int_cols = ["amount", "amount_mean_5", "amount_std_5",
                "amount_change", "time_since_last_tx"]

    bin_cols = ["valid_ref"]  # bleibt binär

    int_index = [feature_cols.index(c) for c in int_cols if c in feature_cols]
    bin_index = [feature_cols.index(c) for c in bin_cols if c in feature_cols]
    #int_index = []
    #bin_index = []

    cat_index = []  # GANZ WICHTIG: ForestDiffusion soll NICHT mehr dummify() machen

    model = ForestDiffusionModel(
        X=X_train_np,
        #label_y=y_train_np,
        # Diffusion / Training
        n_t=10,                 # weniger Zeitschritte für schnellere Tests
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

    # Modell speichern
    dump(model, "forest_diffusion_model.joblib")


# Neue Samples generieren
    samples = model.generate(batch_size=5000, n_t=10)

    # Als DataFrame zurückwandeln
    df_generated = pd.DataFrame(samples, columns=feature_cols)
    df_generated.head(20)

        # -------------------------
    # bank_account_uuid aus One-Hot zurückholen
    # -------------------------

    # alle Dummy-Spalten für die UUID finden
    uuid_dummy_cols = [c for c in feature_cols if c.startswith("bank_account_uuid_")]

    # Werte dieser Spalten als NumPy holen
    uuid_matrix = df_generated[uuid_dummy_cols].to_numpy()

    # pro Zeile: Index der Spalte mit dem größten Wert
    max_idx = uuid_matrix.argmax(axis=1)

    # zugehörige Spaltennamen
    chosen_cols = [uuid_dummy_cols[i] for i in max_idx]

    # Präfix "bank_account_uuid_" entfernen → originale UUID
    bank_account_uuid_original = [
        col.replace("bank_account_uuid_", "") for col in chosen_cols
    ]

    # neue Spalte mit der „echten“ UUID
    df_generated["bank_account_uuid_original"] = bank_account_uuid_original
    df_generated.drop(columns=uuid_dummy_cols, inplace=True)
    # Wenn du willst, die Dummy-Spalten wieder wegwerfen:
    # df_generated = df_generated.drop(columns=uuid_dummy_cols)
    print(df_generated.head(20))
    print(df_generated.columns)




if __name__ == "__main__":
    main()
