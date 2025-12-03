import random
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from ForestDiffusion import ForestDiffusionModel
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
import seaborn as sns
import json
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import time
from preprocessing_bd_unsupervised import prepare_data_unsupervised
from preprocessing_bd_supervised import prepare_data_supervised
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent
project_root_dir = parent_dir.parent
if str(project_root_dir) not in sys.path:
    sys.path.append(str(project_root_dir))
from TCCM.FlowMatchingAD import TCCM
from TCCM.functions import determine_FMAD_hyperparameters



#copied partly from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py




# HINWEIS: Wir benötigen die preprocess-Funktion und die EFVFMDataset-Klasse NICHT mehr direkt,
# da wir die fertigen, vom Preprocessing erstellten .npy-Dateien laden.
# Es genügt, die .npy und .json Dateien zu laden, die du im letzten Schritt erstellt hast.

def load_dataset(dataset_name, semi_supervised):
    
    # Pfad zur Basisdatenbank
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"

    if "business_dataset" in dataset_name:
        # Dein bestehender Code für Business Dataset...
        cols_to_drop_for_model = ["bank_account_uuid"]
        if semi_supervised:
            X_train_df, X_test_df, y_train, y_test, train_mapping, test_mapping, feature_columns = prepare_data_supervised()
        else: 
            X_train_df, X_test_df, y_train, y_test, train_mapping, test_mapping, feature_columns = prepare_data_unsupervised()
        
        X_train_df_model = X_train_df.drop(columns=cols_to_drop_for_model)
        X_test_df_model = X_test_df.drop(columns=cols_to_drop_for_model)
        X_train = X_train_df_model.to_numpy(dtype=float)
        X_test = X_test_df_model.to_numpy(dtype=float)
        y_train = y_train.to_numpy().astype(int)
        y_test = y_test.to_numpy().astype(int)
        
    elif dataset_name.endswith(".npz"):

        #------ copied!!!!!!! --------
        
        if semi_supervised:
            file_path = data_dir / dataset_name
            data = np.load(file_path, allow_pickle=True)
            X, y = data['X'], data['y'].astype(int)
            x_normal, X_anomalous = X[y == 0], X[y == 1]
            y_normal, y_anomalous = y[y == 0], y[y == 1]

            X_train_raw, X_test_normal_raw, y_train, y_test_normal = train_test_split(
                x_normal, y_normal, test_size = 0.5, random_state = 42
            )
            X_test_raw = np.vstack((X_test_normal_raw, X_anomalous))
            y_test = np.concatenate((y_test_normal, y_anomalous))
            
            # Data standardization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # Print dataset information
            print(" ")
            print(f"Dataset loaded successfully!")
            print(f"Training data: {X_train.shape}, Normal: {len(y_train)}")
            print(f"Test data: {X_test.shape}, Normal: {sum(y_test == 0)}, Anomalies: {sum(y_test == 1)}")
        else:
            # Falls unsupervised, nur die Rohdaten laden (nicht hier implementiert)
            pass
        #-----------------------------------
            
    # NEUE SEKTION: Lade die EF-VFM-vorverarbeiteten Daten
    elif dataset_name.endswith("_efvfm"):
        # Wir erwarten, dass der Ordner (z.B. "campaign_efvfm") existiert
        ef_dir = data_dir / dataset_name
        print(f"\nLade EF-VFM Daten aus Ordner: {ef_dir}")

        # Lade die von ForestDiffusion benötigten Daten
        # HINWEIS: ForestDiffusion benötigt skalierten X_train/X_test
        # Die EF-VFM-Vorverarbeitung liefert X_num und X_cat.
        # Da du im vorherigen Schritt KEINE kategorialen Daten hattest, 
        # laden wir nur X_num.
        
        # X_train
        X_train_num = np.load(ef_dir / "X_num_train.npy")
        X_train_cat = np.load(ef_dir / "X_cat_train.npy") if (ef_dir / "X_cat_train.npy").exists() else np.empty((X_train_num.shape[0], 0))
        # Concatenate X_num und X_cat, da das Modell einen einzigen Feature-Block erwartet
        X_train = np.hstack((X_train_num, X_train_cat))
        
        # X_test
        X_test_num = np.load(ef_dir / "X_num_test.npy")
        X_test_cat = np.load(ef_dir / "X_cat_test.npy") if (ef_dir / "X_cat_test.npy").exists() else np.empty((X_test_num.shape[0], 0))
        X_test = np.hstack((X_test_num, X_test_cat))

        # y_test (y_train wird hier nicht benötigt, da semi_supervised/unsupervised)
        y_test = np.load(ef_dir / "y_test.npy").astype(int)
        
        # Lade info.json, um int_indexes und cat_indexes für ForestDiffusion zu bekommen
        with open(ef_dir / "info.json", "r") as f:
            info = json.load(f)

        # HINWEIS: In diesem Setup werden die kategorialen/integer Indizes dem ForestDiffusionModel
        # im nächsten Schritt (create_trained_model) übergeben, was wichtig ist.
        
        # Print dataset information
        print(f"EF-VFM Dataset geladen: X_train {X_train.shape}, X_test {X_test.shape}")
        print(f"Test Normal: {sum(y_test == 0)}, Test Anomalies: {sum(y_test == 1)}")
        
        # Da die EF-VFM-Daten *bereits* transformiert/skaliert sind,
        # geben wir sie direkt zurück.
        
    else:
        raise ValueError(f"Unbekanntes Dataset-Format: {dataset_name}")
        
    return X_train, X_test, y_test

# ... REST DES SKRIPTS (create_trained_model, calculate_scores, __main__) ...

def create_trained_model(n_t, duplicate_K, seed, X_train, dataset_name, diffusion_type):
    start_time = time.time()
    model = ForestDiffusionModel(
    X=X_train,
    label_y=None,     # unsupervised; wir geben Labels nur für Evaluation
    # Diffusion settings
    n_t=n_t,
    duplicate_K=duplicate_K,
    diffusion_type=diffusion_type,  # wichtig für compute_deviation_score
    eps=1e-3,
    model='xgboost',
    max_depth=7,
    #max_depth=2,
    n_estimators=100,
    #n_estimators = 10,
    eta=0.3,
    gpu_hist=False,   # auf True setzen, wenn GPU verfügbar
    n_batch=1,        # Important: 0 for compute_deviation_score
    seed=seed,
    n_jobs=-1,
    bin_indexes=[],
    cat_indexes=[],
    int_indexes=[],
    remove_miss=False,
    p_in_one=True,    # WICHTIG für compute_deviation_score
    )
    end_time_train = time.time()
    time_train = end_time_train - start_time
    print("✓ Model trained successfully on full training data!")
    joblib.dump(model, f"{dataset_name}_model.joblib")

    return model, time_train


def calculate_scores(X_test, y_test, trained_model, n_t, duplicate_K, diffusion_type):
    # train time, dev time, rec time
    times = []
    model, time_train = trained_model
    times.append(time_train)

    # ---Computation deviation Score ---
    start_time_dev = time.time()
    anomaly_scores_deviation = model.compute_deviation_score(
        test_samples=X_test,
        diffusion_type=diffusion_type,
        n_t=n_t, 
        duplicate_K=duplicate_K
    )
    end_time_dev = time.time()
    time_dev = end_time_dev - start_time_dev
    times.append(time_dev)

    #---Computation reconstruction Score ---
    start_time_rec = time.time()
    anomaly_scores_reconstruction = model.compute_reconstruction_score(
        test_samples=X_test,
        diffusion_type=diffusion_type,
        n_t=n_t, 
        duplicate_K=duplicate_K
    )
    end_time_rec = time.time()
    time_reconstruction = end_time_rec - start_time_rec
    times.append(time_reconstruction)

    # ---calculate the auroc & auprc ---
    auroc_deviation = roc_auc_score(y_test, anomaly_scores_deviation)
    auprc_deviation = average_precision_score(y_test, anomaly_scores_deviation)
    print(f"\nAUROC deviation: {auroc_deviation:.4f}")
    print(f"AUPRC deviation (Average Precision): {auprc_deviation:.4f}")

    auroc_reconstruction = roc_auc_score(y_test, anomaly_scores_reconstruction)
    auprc_reconstruction = average_precision_score(y_test, anomaly_scores_reconstruction)
    print(f"\nAUROC reconstruction: {auroc_reconstruction:.4f}")
    print(f"AUPRC reconstruction (Average Precision): {auprc_reconstruction:.4f}")

    return auroc_deviation, auprc_deviation, auroc_reconstruction, auprc_reconstruction, times

#noch mit binary, float etc. machen, als parameter übergeben!!!


if __name__ == "__main__":
    dataset_names = [#"5_campaign.npz"]
                     # "13_fraud.npz"]
                     #"campaign_efvfm"]
                    "29_Pima.npz"]
                    #"44_Wilt.npz"]
                    #"31_satimage-2.npz"]
                     # "business_dataset_3011.csv"]
    #schleife bauen
    # supervised = [True]
    # n_t = 15  #not 1!
    # duplicate_K = 10
    # duplicate_K_scoring = 3
    # number_of_runs = 5

    n_t = 80  #not 1!
    duplicate_K = 30
    duplicate_K_scoring = 30
    number_of_runs = 1
    diffusion_type = "vp"

    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)

        dev_auroc_list = []
        dev_auprc_list = []
        rec_auroc_list = []
        rec_auprc_list = []

        time_train_list = []
        time_dev_inf_list = []
        time_rec_inf_list = []
        #compute seperately total times
        time_dev_total_list = [] 
        time_rec_total_list = []

        #always the same split for comparability, no seed change
        X_train, X_test, y_test = load_dataset(dataset_name, semi_supervised=True)

        # ---runs ---
        for i in range(number_of_runs):
            print(f"\n--- Iteration {i+1}/{number_of_runs} (Seed: {i}) ---")
            trained_model = create_trained_model(
                n_t=n_t,
                duplicate_K=duplicate_K,
                seed=i,
                X_train=X_train, 
                dataset_name=dataset_name,
                diffusion_type=diffusion_type
            )
            a_dev, ap_dev, a_rec, ap_rec, times = calculate_scores(X_test, y_test, trained_model, n_t, duplicate_K_scoring, diffusion_type=diffusion_type)
            print(f"Iteration {i+1} results: AUROC Deviation: {a_dev:.4f}, AUPRC Deviation: {ap_dev:.4f}, AUROC Reconstruction: {a_rec:.4f}, AUPRC Reconstruction: {ap_rec:.4f}")
            dev_auroc_list.append(a_dev)
            dev_auprc_list.append(ap_dev)
            rec_auroc_list.append(a_rec)
            rec_auprc_list.append(ap_rec)

            #add times to time lists
            time_train_list.append(times[0])
            time_dev_inf_list.append(times[1])
            time_rec_inf_list.append(times[2])

            #compute seperately total times
            time_dev_total_list.append(times[0] + times[1])  
            time_rec_total_list.append(times[0] + times[2])  

        # ---merge results over different runs ---
        dev_auroc_mean = np.mean(dev_auroc_list)
        dev_auroc_std = np.std(dev_auroc_list)
        dev_auprc_mean = np.mean(dev_auprc_list)
        dev_auprc_std = np.std(dev_auprc_list)

        rec_auroc_mean = np.mean(rec_auroc_list)
        rec_auroc_std = np.std(rec_auroc_list)
        rec_auprc_mean = np.mean(rec_auprc_list)
        rec_auprc_std = np.std(rec_auprc_list)

        #compute the time means and stds
        time_train_mean = np.mean(time_train_list)
        time_train_std = np.std(time_train_list)
        time_dev_inf_mean = np.mean(time_dev_inf_list)
        time_dev_inf_std = np.std(time_dev_inf_list)
        time_rec_inf_mean = np.mean(time_rec_inf_list)
        time_rec_inf_std = np.std(time_rec_inf_list)

        time_dev_total_mean = np.mean(time_dev_total_list)
        time_dev_total_std = np.std(time_dev_total_list)
        time_rec_total_mean = np.mean(time_rec_total_list)
        time_rec_total_std = np.std(time_rec_total_list)


        print("\n" + "=" * 40 + " FINAL RESULTS " + "=" * 40)
        print(f"Dataset: {dataset_name}")
        print("dev_auroc_list: "+str(dev_auroc_list))
        print("dev_auprc_list: "+str(dev_auprc_list))
        print("rec_auroc_list: "+str(rec_auroc_list))
        print("rec_auprc_list: "+str(rec_auprc_list))
        print("-" * 87)
        print(f"Metric | Deviation Score (Mean ± Std) | Reconstruction Score (Mean ± Std)")
        print("-" * 87)
        print(f"AUROC | {dev_auroc_mean:.3f} ± {dev_auroc_std:.3f} | {rec_auroc_mean:.3f} ± {rec_auroc_std:.3f}")
        print(f"AUPRC | {dev_auprc_mean:.3f} ± {dev_auprc_std:.3f} | {rec_auprc_mean:.3f} ± {rec_auprc_std:.3f}")
        print("=" * 87)

        print("\n" + "=" * 40 + " TIME RESULTS " + "=" * 40)
        print(f"Dataset: {dataset_name}")
        print("-" * 87)
        print(f"Time Metric | Training Time (Mean ± Std) | Dev Inference Time (Mean ± Std) | Rec Inference Time (Mean ± Std)")
        print("-" * 87)
        print(f"Time (s) | {time_train_mean:.3f} ± {time_train_std:.3f} | {time_dev_inf_mean:.3f} ± {time_dev_inf_std:.3f} | {time_rec_inf_mean:.3f} ± {time_rec_inf_std:.3f}")
        print(f"Total Time Dev (s) | {time_dev_total_mean:.3f} ± {time_dev_total_std:.3f}")
        print(f"Total Time Rec (s) | {time_rec_total_mean:.3f} ± {time_rec_total_std:.3f}")
        print("=" * 87) 



