import numpy as np
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# HIER anpassen, falls deine preprocess/EFVFMDataset woanders liegen
from utils_train import preprocess, EFVFMDataset


def main():
    # Basis-Verzeichnis: Ordner dieser Datei
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"

    # Name deiner npz-Datei (so wie in deinem Snippet)
    dataset_name = "29_Pima.npz"   # oder "5_campaign.npz", falls das der genaue Name ist
    file_path = data_dir / dataset_name

    print(f"Lade Datei: {file_path}")
    data = np.load(file_path, allow_pickle=True)

    # Wir gehen davon aus, dass die Keys X und y heißen (wie in deinem Code)
    X, y = data["X"], data["y"].astype(int)

    print("\n=== Originaldaten aus NPZ ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    # ---------------------------
    # Dein ursprünglicher Split:
    # normal (0) vs anomal (1)
    # ---------------------------
    x_normal, X_anomalous = X[y == 0], X[y == 1]
    y_normal, y_anomalous = y[y == 0], y[y == 1]

    X_train_raw, X_test_normal_raw, y_train, y_test_normal = train_test_split(
        x_normal, y_normal, test_size=0.5, random_state=42
    )

    # Testset normal + anomal
    X_test_raw = np.vstack((X_test_normal_raw, X_anomalous))
    y_test = np.concatenate((y_test_normal, y_anomalous))

    print("\n=== Nach deinem Split (noch unskaliert) ===")
    print(f"X_train_raw: {X_train_raw.shape}, y_train len: {len(y_train)} (nur Normal)")
    print(f"X_test_raw:  {X_test_raw.shape}, y_test  len: {len(y_test)}")
    print(f"Test Normal: {np.sum(y_test == 0)}, Test Anomal: {np.sum(y_test == 1)}")

    # Optional: StandardScaler wie in deinem Code (nur zu Vergleichszwecken)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("\n=== Mit StandardScaler (nur Info, EF-VFM nutzt eigene Skalierung) ===")
    print(f"X_train (scaled) shape: {X_train.shape}")
    print(f"X_test  (scaled) shape: {X_test.shape}")

    # ----------------------------------------------------
    # EF-VFM erwartet einen Ordner mit .npy + info.json
    # Wir nehmen die UNskalierten Daten dafür.
    # ----------------------------------------------------
    ef_dir = data_dir / "29_Pima_efvfm"
    ef_dir.mkdir(exist_ok=True)
    print(f"\nSpeichere EF-VFM-Dateien nach: {ef_dir}")

    # Alles als numerische Features behandeln -> X_num
    X_num_train = X_train_raw
    X_num_test = X_test_raw

    np.save(ef_dir / "X_num_train.npy", X_num_train)
    np.save(ef_dir / "X_num_test.npy", X_num_test)

    # Keine kategorialen Features -> keine X_cat-Dateien
    # y speichern
    np.save(ef_dir / "y_train.npy", y_train)
    np.save(ef_dir / "y_test.npy", y_test)

    # info.json bauen
    n_features = X_num_train.shape[1]
    num_idx = list(range(n_features))
    cat_idx = []   # keine Kategorien
    int_idx = []   # falls du weißt, dass manche Spalten integer sind, kannst du die hier aufführen

    info = {
        "task_type": "binclass",             # binäre Klassifikation (0 normal, 1 anomal)
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "int_col_idx": int_idx,
        "int_col_idx_wrt_num": [],           # hier könnten Indizes integer-Spalten relativ zu num stehen
        "n_classes": 2,
    }

    with open(ef_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print("info.json geschrieben.")

    # ----------------------------------------------------
    # Jetzt: EF-VFM preprocess benutzen, um zu sehen,
    # wie das Dataset in EF-Darstellung aussieht
    # ----------------------------------------------------
    print("\n=== Rufe EF-VFM preprocess() auf ===")
    X_num, X_cat, categories, d_numerical, num_inv, int_inv, cat_inv = preprocess(
        dataset_path=str(ef_dir),
        dequant_dist="none",
        int_dequant_factor=0.0,
        task_type="binclass",
        inverse=True,
        cat_encoding=None,
        concat=True,
    )

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    print("\n=== Shapes nach EF-VFM-Preprocessing ===")
    print(f"X_train_num: {X_train_num.shape}")
    print(f"X_test_num:  {X_test_num.shape}")
    print(f"X_train_cat: {X_train_cat.shape}  (sollte (N,0) sein, da keine Kategorien)")
    print(f"X_test_cat:  {X_test_cat.shape}")
    print(f"d_numerical: {d_numerical}")
    print(f"categories:  {categories}")

    print("\nBeispiel: erste 3 Zeilen von X_train_num:")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(X_train_num[:3])

    # ----------------------------------------------------
    # Optional: EFVFMDataset nutzen (wie im Training)
    # ----------------------------------------------------
    print("\n=== EFVFMDataset (Train) ===")
    # info aus Datei laden, damit es genau dem Training entspricht
    with open(ef_dir / "info.json", "r") as f:
        info_loaded = json.load(f)
        info_loaded["task_type"] = "binclass"

    train_dataset = EFVFMDataset(
        dataname="29_Pima_ef_vfm",
        data_dir=str(ef_dir),
        info=info_loaded,
        isTrain=True,
        dequant_dist="none",
        int_dequant_factor=0.0,
    )

    print(f"Anzahl Trainingssamples: {len(train_dataset)}")
    sample = train_dataset[0]
    print(f"Shape eines Samples: {sample.shape}")
    print(f"d_numerical im Dataset: {train_dataset.d_numerical}")
    print(f"Kategorien im Dataset: {train_dataset.categories}")

    x0_num = sample[: train_dataset.d_numerical]
    x0_cat = sample[train_dataset.d_numerical :]

    print("\nBeispiel-Sample (train_dataset[0]):")
    print("Numerischer Teil (erste 10 Werte):")
    print(x0_num[:10])
    print("Kategorialer Teil (sollte leer oder sehr kurz sein):")
    print(x0_cat)

    print("\nFertig – jetzt siehst du genau, wie dein campaign-Datensatz in EF-VFM aussieht.\n")


if __name__ == "__main__":
    main()
