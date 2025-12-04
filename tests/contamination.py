import random
import os
#from ForestDiffusion_neu.ForestDiffusion.tests.evaluater_unsupervised import calculate_scores_deviation, calculate_scores_reconstruction
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from ForestDiffusion import ForestDiffusionModel
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
import seaborn as sns
import json
import joblib
from sklearn.metrics import average_precision_score
import time
from preprocessing_bd_unsupervised import prepare_data_unsupervised
from preprocessing_bd_supervised import prepare_data_supervised
import sys
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent
project_root_dir = parent_dir.parent
if str(project_root_dir) not in sys.path:
    sys.path.append(str(project_root_dir))
from TCCM.FlowMatchingAD import TCCM
from TCCM.functions import determine_FMAD_hyperparameters


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




def create_ForestDiffusionModel(n_t, duplicate_K, seed, X_train, dataset_name, diffusion_type):
    start_time_train = time.time()
    model = ForestDiffusionModel(
        X=X_train,
        label_y=None,     
        n_t=n_t,
        duplicate_K=duplicate_K,
        diffusion_type=diffusion_type, 
        eps=1e-3,
        model='xgboost',
        max_depth=7,
        #max_depth=2,
        n_estimators=100,
        #n_estimators = 10,
        eta=0.3,
        gpu_hist=False,  
        n_batch=1,      
        seed=seed,
        n_jobs=-1,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=[],
        remove_miss=False,
        p_in_one=True,      
    )
    end_time_train = time.time()
    time_train = end_time_train - start_time_train
    print("✓ Model trained successfully on full training data!")
    joblib.dump(model, f"{dataset_name}_model.joblib")

    return model, time_train


def calculate_scores_ForestDiffusionModel(X_test, y_test, model, n_t, duplicate_K_test, diffusion_type, score):

    if score == "deviation":
        if diffusion_type == "flow":
            anomaly_scores_deviation = model.compute_deviation_score(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test
            )
        elif diffusion_type == "vp":
            anomaly_scores_deviation = model.compute_deviation_score(
                X_test, 
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test, 
                diffusion_type=diffusion_type
            )
        return anomaly_scores_deviation

    elif score == "reconstruction":
        if diffusion_type == "flow":
            anomaly_scores_reconstruction = model.compute_reconstruction_score(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test
            )
        else: 
            raise ValueError("Reconstruction score is only implemented for diffusion.")
        return anomaly_scores_reconstruction
    elif score == "decision":
        if diffusion_type == "flow":
            anomaly_scores_decision = model.compute_decision_score(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test
            )
        elif diffusion_type == "vp":
            anomaly_scores_decision = model.compute_decision_score_vp(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test
            )
        return anomaly_scores_decision
    
    else: 
        raise ValueError(f"Unknown score type: {score}")


#noch mit binary, float etc. machen, als parameter übergeben!!!

def create_trained_tccm_model(X_train, dataset_name, seed):
    """
    Erstellt und trainiert ein TCCM-Modell
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Hole die optimalen Hyperparameter für das Dataset
    hyperparams = determine_FMAD_hyperparameters(dataset_name)
    
    print(f"\nTraining TCCM with hyperparameters:")
    print(f"  Epochs: {hyperparams['epochs']}")
    print(f"  Learning Rate: {hyperparams['learning_rate']}")
    print(f"  Batch Size: {hyperparams['batch_size']}")
    
    start_time = time.time()
    
    # Initialisiere TCCM mit den Hyperparametern
    model = TCCM(
        n_features=X_train.shape[1],
        epochs=hyperparams['epochs'],
        learning_rate=hyperparams['learning_rate'],
        batch_size=hyperparams['batch_size']
    )
    
    # Trainiere das Modell
    model.fit(X_train)
    
    end_time_train = time.time()
    time_train = end_time_train - start_time
    
    print(f"✓ TCCM Model trained successfully in {time_train:.2f} seconds!")
    
    # Speichere das Modell
    model_path = f"{dataset_name}_tccm_model_seed{seed}.joblib"
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")
    
    return model, time_train



def calculate_tccm_scores(X_test, y_test, model, n_t, score):
    # --- Score 1: Decision Function  ---
    if score == "decision":
        anomaly_scores_decision = model.decision_function(X_test)
        return anomaly_scores_decision

    # --- Score 2: Deviation Score  ---
    elif score == "deviation":
        anomaly_scores_deviation = model.compute_deviation_score(X_test, n_t=n_t)
        return anomaly_scores_deviation
        
    # --- Score 3: Reconstruction Score  ---
    elif score == "reconstruction":
        anomaly_scores_reconstruction = model.compute_reconstruction_score(X_test, n_t = n_t)
        return anomaly_scores_reconstruction
    
    else: 
        raise ValueError(f"Unknown score type: {score}")


# # ============================================================================
# # adapted (then extended) from https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/AblationStudies.py
# # ============================================================================

def run_training_contamination_ablation_dynamic_fixed_split(score, dataset_names, model):
    #seed_list = [0, 1, 2, 3, 4]
    seed_list = [1]
    all_results = {}
    all_contam_levels = {}
    model_name = model[0]
    model_cnf = model[1]
    for dataset_name in dataset_names:       
        print(f"\n Dataset: {dataset_name} with model: {model_name} and score: {score}")
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

                p = model_cnf["params"]
                if model_cnf["type"] == "forest":
                    model, _ = create_ForestDiffusionModel(
                        n_t=p["n_t"],
                        duplicate_K=p["duplicate_K"],
                        seed=seed,
                        X_train=X_train,
                        dataset_name=f"{dataset_name}_{model_name}",
                        diffusion_type=p["diffusion_type"]
                    )
                    scores = calculate_scores_ForestDiffusionModel(X_test, y_test, model, n_t=p["n_t"], duplicate_K_test=p["duplicate_K_test"], diffusion_type=p["diffusion_type"], score=score)
 
                if model_cnf["type"] == "tccm":
                    model, _ = create_trained_tccm_model(
                        X_train=X_train,
                        dataset_name=f"{dataset_name}_{model_name}",
                        seed=seed
                    )
                    scores = calculate_tccm_scores(X_test, y_test, model, n_t=p["n_t"], score=score)


                auc = roc_auc_score(y_test, scores)
                pr = average_precision_score(y_test, scores)
                aucs.append(auc)
                prs.append(pr)

            auroc_all.append((np.mean(aucs), np.std(aucs)))
            auprc_all.append((np.mean(prs), np.std(prs)))

        all_results[dataset_name] = {
            "score": score,
            "model_name": model_name,
            "auroc": auroc_all,
            "auprc": auprc_all,
            "contamination_levels": contamination_levels 
        }

    #all results contains a dictionary for each dataset with auroc and auprc values (with mean and std) for each contamination level
    #added score and model_name contamination for grouping and creating plots
    #all_contam_levels contains the contamination levels used for each dataset
    return all_results


# Create the line plots
def plot_training_contamination_ablation_dynamic(results, contamination_levels_dict, model_name, score):
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
    plt.savefig(f"./results_ablation/{model_name}_{score}.pdf")
    plt.show()

# # ============================================================================
# # 
# # ============================================================================



def plot_model_scores_comparison(all_results, model_name, metric, dataset_names):
    """
    Zeigt für EIN Modell alle 3 Scores auf allen Datasets
    metric: "auroc" oder "auprc"
    """
    # Diese Zeile fehlte!
    fig, axs = plt.subplots(1, len(dataset_names), figsize=(14, 5))
    
    first_dataset = dataset_names[0]
    scores = list(all_results[first_dataset][model_name].keys())
    
    # Definiere Farben für alle möglichen Scores
    colors = {
        "deviation": "blue", 
        "reconstruction": "green", 
        "decision": "red"
    }

    for idx, dataset_name in enumerate(dataset_names):
        ax = axs[idx] if len(dataset_names) > 1 else axs
        
        for score in scores:
            data = all_results[dataset_name][model_name][score]
            contam_levels = data["contamination_levels"]
            values = np.array(data[metric])
            
            ax.plot(contam_levels, values[:, 0], label=score, 
                   color=colors[score], marker='o')
            ax.fill_between(contam_levels,
                          values[:, 0] - values[:, 1],
                          values[:, 0] + values[:, 1],
                          color=colors[score], alpha=0.2)
        
        ax.set_title(f"{dataset_name} - {model_name}")
        ax.set_xlabel("Contamination Level")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"./results_ablation/{model_name}_all_scores_{metric}.pdf")
    plt.show()


def plot_score_models_comparison(all_results, score, metric, dataset_names):
    """
    Zeigt für EINEN Score alle Modelle auf allen Datasets
    metric: "auroc" oder "auprc"
    """
    fig, axs = plt.subplots(1, len(dataset_names), figsize=(14, 5))
    model_names = list(all_results[dataset_names[0]].keys())
    colors = {"ForestFlow_1": "blue", "ForestFlow_2": "green", "TCCM": "red"}
    
    for idx, dataset_name in enumerate(dataset_names):
        ax = axs[idx] if len(dataset_names) > 1 else axs
        
        for model_name in model_names:
            # Prüfe ob dieser Score für dieses Modell existiert
            if score not in all_results[dataset_name][model_name]:
                print(f"Skipping {model_name} for score '{score}' (not available)")
                continue
            data = all_results[dataset_name][model_name][score]
            contam_levels = data["contamination_levels"]
            values = np.array(data[metric])
            
            ax.plot(contam_levels, values[:, 0], label=model_name,
                   color=colors.get(model_name, "black"), marker='o')
            ax.fill_between(contam_levels,
                          values[:, 0] - values[:, 1],
                          values[:, 0] + values[:, 1],
                          color=colors.get(model_name, "black"), alpha=0.2)
        
        ax.set_title(f"{dataset_name} - {score}")
        ax.set_xlabel("Contamination Level")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"./results_ablation/all_models_{score}_{metric}.pdf")
    plt.show()



if __name__ == "__main__":
    dataset_names = ["29_Pima.npz"]
                    #["5_campaign.npz"]
                     #"18_Ionosphere.npz"]


    #attention with the names!
    models_to_run = {
        "ForestFlow_1": {
            "type": "forest",
            "params": {
                "n_t": 20,
                "duplicate_K": 20,
                "duplicate_K_test": 20,
                "diffusion_type": "flow"
            }
        },
        "ForestFlow_2": {
            "type": "forest",
            "params": {
                "n_t": 10,
                "duplicate_K": 30,
                "duplicate_K_test": 30,
                "diffusion_type": "flow"
            }
        },
        "TCCM": {
            "type": "tccm",
            "params": {
                "n_t": 400   # only used for scoring functions
            }
        }
    }   
    # for model in models_to_run.items():
    #     for score in ["deviation", "reconstruction", "decision"]:
    #         results, contamination_levels = run_training_contamination_ablation_dynamic_fixed_split(score, dataset_names, model)
    #         results
    #         plot_training_contamination_ablation_dynamic(results, contamination_levels, model_name=model[0], score=score)
    all_results_combined = {}

    for model_name, model_config in models_to_run.items():
        model_type = model_config["type"]
        params = model_config["params"]       
        if model_type == "forest":
            if params["diffusion_type"] == "vp":
                score_list = ["deviation", "decision"]
            elif params["diffusion_type"] == "flow":
                score_list = ["deviation", "reconstruction", "decision"]
            else:
                score_list = ["deviation", "reconstruction", "decision"]  
        elif model_type == "tccm":
            score_list = ["deviation", "reconstruction", "decision"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        for score in score_list:
            results = run_training_contamination_ablation_dynamic_fixed_split(score, dataset_names, (model_name, model_config) )
            for dataset_name, data in results.items():
                if dataset_name not in all_results_combined:
                    all_results_combined[dataset_name] = {}
                if model_name not in all_results_combined[dataset_name]:
                    all_results_combined[dataset_name][model_name] = {}
                all_results_combined[dataset_name][model_name][score] = data
    for model_name in models_to_run.keys():
        plot_model_scores_comparison(all_results_combined, model_name, metric="auroc", dataset_names=dataset_names)
        plot_model_scores_comparison(all_results_combined, model_name, metric="auprc", dataset_names=dataset_names)

    for score in ["deviation", "reconstruction", "decision"]:
        plot_score_models_comparison(all_results_combined, score, metric="auroc", dataset_names=dataset_names)
        plot_score_models_comparison(all_results_combined, score, metric="auprc", dataset_names=dataset_names)




