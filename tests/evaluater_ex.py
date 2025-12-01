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


#copied partly from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py




def load_dataset(dataset_name, semi_supervised):
    if "business_dataset" in dataset_name:
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
    else: 
        if semi_supervised:
            #copied partly from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py

            base_dir = Path(__file__).resolve().parent
            #file_path = base_dir.parent / "data" / "5_campaign.npz"
            file_path = base_dir.parent / "data" / dataset_name
            data = np.load(file_path, allow_pickle=True)
            X, y = data['X'], data['y'].astype(int)
            x_normal, X_anomalous = X[y == 0], X[y == 1]
            y_normal, y_anomalous = y[y == 0], y[y == 1]

            X_train, X_test_normal, y_train, y_test_normal = train_test_split(
                x_normal, y_normal, test_size = 0.5, random_state = 42
            )
            # Test set contains both normal and abnormal data
            X_test = np.vstack((X_test_normal, X_anomalous))
            y_test = np.concatenate((y_test_normal, y_anomalous))
            # Data standardization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Print dataset information
            print(" ")
            print(f"Dataset loaded successfully!")
            print(f"Training data: {X_train.shape}, Normal: {len(y_train)}")
            print(f"Test data: {X_test.shape}, Normal: {sum(y_test == 0)}, Anomalies: {sum(y_test == 1)}")
        else:
            pass
    return X_train, X_test, y_test


def create_trained_model(n_t, duplicate_K, seed, X_train, dataset_name):
    start_time = time.time()
    model = ForestDiffusionModel(
    X=X_train,
    label_y=None,     # unsupervised; wir geben Labels nur für Evaluation
    # Diffusion settings
    n_t=n_t,
    duplicate_K=duplicate_K,
    diffusion_type='flow',  # wichtig für compute_deviation_score
    eps=1e-3,
    model='xgboost',
    #max_depth=7,
    max_depth=2,
    #n_estimators=100,
    n_estimators = 10,
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


def calculate_scores(X_test, y_test, trained_model, n_t, duplicate_K):
    # train time, dev time, rec time
    times = []
    model, time_train = trained_model
    times.append(time_train)

    # ---Computation deviation Score ---
    start_time_dev = time.time()
    anomaly_scores_deviation = model.compute_deviation_score(
        test_samples=X_test,
        n_t=n_t, 
        duplicate_K=duplicate_K
    )
    end_time_dev = time.time()
    time_dev = end_time_dev - start_time_dev
    times.append(time_dev)

    # ---Computation reconstruction Score ---
    start_time_rec = time.time()
    anomaly_scores_reconstruction = model.compute_reconstruction_score(
        test_samples=X_test,
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
    dataset_names = [
"business_dataset_3011.csv"]
    #schleife bauen
    supervised = [True]
    n_t = 15  #not 1!
    duplicate_K = 10
    duplicate_K_scoring = 3
    number_of_runs = 5

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
                dataset_name=dataset_name
            )
            a_dev, ap_dev, a_rec, ap_rec, times = calculate_scores(X_test, y_test, trained_model, n_t, duplicate_K_scoring)
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



