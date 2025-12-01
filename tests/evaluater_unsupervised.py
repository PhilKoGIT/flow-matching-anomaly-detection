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



#copied partly from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py


n_t = 15  #not 1!
#duplicate_K = 10
duplicate_K = 10
number_of_runs = 5

# # ============================================================================
# # adapted/copied from https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py
# # ============================================================================

def load_adbench_npz(dataset_name, test_size=0.5, random_state=42):
    #part copied from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py

    base_dir = Path(__file__).resolve().parent
    #file_path = base_dir.parent / "data" / "5_campaign.npz"
    file_path = base_dir.parent / "data_contamination" / dataset_name
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

    return X_train, y_train, X_test, y_test

# # ============================================================================
# # 
# # ============================================================================



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


def calculate_scores_deviation(X_test, y_test, trained_model, duplicate_K, n_t):

    model, _ = trained_model
    anomaly_scores_deviation = model.compute_deviation_score(
        test_samples=X_test,
        n_t=n_t, 
        duplicate_K=duplicate_K   # same amount of noise as training
    )

    return anomaly_scores_deviation


def calculate_scores_reconstruction(X_test, y_test, trained_model, duplicate_K, n_t):
    model, _ = trained_model
    # ---Computation reconstruction Score ---
    anomaly_scores_reconstruction = model.compute_reconstruction_score(
        test_samples=X_test,
        n_t=n_t, 
        duplicate_K=duplicate_K 
    )
    return anomaly_scores_reconstruction



#noch mit binary, float etc. machen, als parameter übergeben!!!

# # ============================================================================
# # adapted from https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/AblationStudies.py
# # ============================================================================


# Dataset configuration



seed_list = [0, 1, 2, 3, 4]

#!!!!!!!!
#seed_list = [0,1]
def run_training_contamination_ablation_dynamic_fixed_split(score):
    all_results = {}
    all_contam_levels = {}
    dataset_names = [
        #"17_InternetAds.npz",
        #"18_Ionosphere.npz",
        #"5_campaign.npz",
        #"13_fraud.npz",
       # "22_magic.gamma.npz",
       # "23_mammography.npz",
       # "25_musk.npz",
        "29_Pima.npz",
       # "31_satimage-2.npz"
       # "44_Wilt.npz"
    ]
    for dataset_name in dataset_names:       
        print(f"\n Dataset: {dataset_name}")
        auroc_all = []
        auprc_all = []

        # ============================
        # Split normal / anomaly using fixed random seed 42
        # ============================
        X_train_full, y_train_full, X_test_full, y_test_full = load_adbench_npz(dataset_name, test_size=0.5, random_state=42)
        X_all = np.vstack([X_train_full, X_test_full])
        y_all = np.concatenate([y_train_full, y_test_full])

        X_normal = X_all[y_all == 0]
        X_abnormal = X_all[y_all == 1]

        from sklearn.model_selection import train_test_split
        
        X_train_normal, X_test_normal = train_test_split(X_normal, test_size=0.5, random_state=42, stratify=None)
        X_train_abnormal_full, X_test_abnormal = train_test_split(X_abnormal, test_size=0.5, random_state=42, stratify=None)

        X_test = np.vstack([X_test_normal, X_test_abnormal])
        y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_abnormal))])

        n_train_normal = len(X_train_normal)
        n_train_abnormal_max = len(X_train_abnormal_full)
        max_abnormal_ratio = n_train_abnormal_max / (n_train_abnormal_max + n_train_normal)
        contamination_levels = np.linspace(0.001, max_abnormal_ratio, 10)

        all_contam_levels[dataset_name] = contamination_levels

        for contam_ratio in contamination_levels:
            aucs, prs = [], []

            for seed in seed_list:
                np.random.seed(seed)
                random.seed(seed)

                n_ab = int(contam_ratio * n_train_normal / (1 - contam_ratio))
                n_ab = min(n_ab, n_train_abnormal_max)

                selected_ab = X_train_abnormal_full[:n_ab]  # Can be replaced by random sampling if needed
                X_train = np.vstack([X_train_normal, selected_ab])
                y_train = np.zeros(len(X_train))  # label unused

                model = create_trained_model(n_t, duplicate_K, seed, X_train, dataset_name)
                if score == "deviation":
                    scores = calculate_scores_deviation(X_test, y_test, model, n_t=n_t, duplicate_K=duplicate_K)
                else:
                    scores = calculate_scores_reconstruction(X_test, y_test, model, n_t=n_t, duplicate_K=duplicate_K)

                auc = roc_auc_score(y_test, scores)
                pr = average_precision_score(y_test, scores)
                aucs.append(auc)
                prs.append(pr)

            auroc_all.append((np.mean(aucs), np.std(aucs)))
            auprc_all.append((np.mean(prs), np.std(prs)))

        all_results[dataset_name] = {
            "auroc": auroc_all,
            "auprc": auprc_all,
        }
        print(all_results)

    return all_results, all_contam_levels


# Create the line plots
def plot_training_contamination_ablation_dynamic(results, contamination_levels_dict, score):
    fig, axs = plt.subplots(2, 4, figsize=(28, 10), sharey=True)
    dataset_list = list(results.keys())

    for i, dataset_name in enumerate(dataset_list):
        row, col = divmod(i, 4)
        ax = axs[row][col]

        auroc = np.array(results[dataset_name]["auroc"])
        auprc = np.array(results[dataset_name]["auprc"])
        contam_levels = contamination_levels_dict[dataset_name]

        ax.plot(contam_levels, auroc[:, 0], label="AUROC", color="blue", marker='o')
        ax.fill_between(contam_levels,
                        auroc[:, 0] - auroc[:, 1],
                        auroc[:, 0] + auroc[:, 1],
                        color="blue", alpha=0.2)

        ax.plot(contam_levels, auprc[:, 0], label="AUPRC", color="red", marker='s')
        ax.fill_between(contam_levels,
                        auprc[:, 0] - auprc[:, 1],
                        auprc[:, 0] + auprc[:, 1],
                        color="red", alpha=0.2)

        ax.set_title(dataset_name)
        ax.set_xlabel("Abnormal Ratio in Training Set")
        ax.set_ylim(0, 1.05)
        ax.grid(True)

        if col == 0:
            ax.set_ylabel("Accuracy Score")

    axs[0][3].legend(loc="upper center", bbox_to_anchor=(1.15, 1.1), fontsize="large")
    plt.tight_layout()
    os.makedirs("./results_ablation/", exist_ok=True)
    plt.savefig(f"./results_ablation/ForestDiffusion_{score}.pdf")
    plt.show()




# # ============================================================================
# # 
# # ============================================================================

if __name__ == "__main__":
    results, contamination_levels = run_training_contamination_ablation_dynamic_fixed_split(score="deviation")
    plot_training_contamination_ablation_dynamic(results, contamination_levels, score="deviation")
    results_reconstruction, contamination_levels_reconstruction = run_training_contamination_ablation_dynamic_fixed_split(score="reconstruction")
    plot_training_contamination_ablation_dynamic(results_reconstruction, contamination_levels_reconstruction, score="reconstruction")




