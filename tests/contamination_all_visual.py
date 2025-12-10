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
from ForestDiffusion.tests.preprocessing_bd_unsupervised_old import prepare_data_unsupervised
from ForestDiffusion.tests.preprocessing_bd_supervised_old import prepare_data_supervised
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

#extended to safing the results for the extreme cases (0% and max contamination)

def run_training_contamination_ablation_dynamic_fixed_split(score, dataset_names, model):
    #seed_list = [0, 1, 2, 3, 4]
    seed_list = [1,2,3]
    all_results = {}
    all_contam_levels = {}

    #safe for all maximum contamination levels for each dataset and for zero contamination
    extreme_cases_data = {}


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

        # extreme cases
        extreme_contamination_levels = [0.0, max_abnormal_ratio]

        all_contam_levels[dataset_name] = contamination_levels

        extreme_cases_data[dataset_name] = {
            "no_contamination": {
                "scores_per_seed": [],
                "y_test": y_test,
                "threshold_metrics_per_seed": []
            },
            "full_contamination": {
                "scores_per_seed": [],
                "y_test": y_test,
                "threshold_metrics_per_seed": []
            }
        }
        for contam_idx, contam_ratio in enumerate(contamination_levels):
            aucs, prs = [], []
            ####---------------------added 

            is_no_contam = (contam_idx == 0)  # Erster Level ≈ 0
            is_full_contam = (contam_idx == len(contamination_levels) - 1)  # Letzter Level
            ####---------------------added end
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

                ####---------------------added 
                #Safes extreme cases
                if is_no_contam or is_full_contam:
                    case_key = "no_contamination" if is_no_contam else "full_contamination"
                    extreme_cases_data[dataset_name][case_key]["scores_per_seed"].append({
                        "seed": seed,
                        "anomaly_scores": scores.copy(),
                        "auc": auc,
                        "auprc": pr
                    })
                    
                    # Berechne Threshold-Metriken
                    threshold_metrics = compute_threshold_metrics(scores, y_test)
                    threshold_metrics["seed"] = seed
                    extreme_cases_data[dataset_name][case_key]["threshold_metrics_per_seed"].append(threshold_metrics)
                ####---------------------added end
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
    return all_results, all_contam_levels, extreme_cases_data


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
# #  oben drüber wichtig!!!!!!!
# # ============================================================================

#needed for theshold metrics safing for extreme cases
def compute_threshold_metrics(anomaly_scores, y_test):
    """Berechnet Metriken für verschiedene Threshold-Percentiles"""
    thresholds_percentiles = [70, 80, 90, 95, 97.5, 99]
    results = {
        "percentile_metrics": {},
        "best_f1": 0,
        "best_threshold": None,
        "best_percentile": None
    }
    
    for percentile in thresholds_percentiles:
        threshold = np.percentile(anomaly_scores, percentile)
        preds = (anomaly_scores > threshold).astype(int)

        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_test)

        results["percentile_metrics"][percentile] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }
        
        if f1 > results["best_f1"]:
            results["best_f1"] = f1
            results["best_threshold"] = threshold
            results["best_percentile"] = percentile
    
    return results









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


def aggregate_extreme_cases(extreme_cases_data):
    """Aggregiert die Ergebnisse über alle Seeds"""
    aggregated = {}

    for dataset_name, cases in extreme_cases_data.items():
        aggregated[dataset_name] = {}
        
        for case_key in ["no_contamination", "full_contamination"]:
            case_data = cases[case_key]
            scores_list = case_data["scores_per_seed"]
            threshold_list = case_data["threshold_metrics_per_seed"]
            
            if not scores_list:
                continue
            
            # Aggregiere AUC/AUPRC
            aucs = [s["auc"] for s in scores_list]
            auprcs = [s["auprc"] for s in scores_list]
            
            # Aggregiere Threshold-Metriken
            aggregated_thresholds = {}
            for percentile in [70, 80, 90, 95, 97.5, 99]:
                f1s = [t["percentile_metrics"][percentile]["f1"] for t in threshold_list]
                precisions = [t["percentile_metrics"][percentile]["precision"] for t in threshold_list]
                recalls = [t["percentile_metrics"][percentile]["recall"] for t in threshold_list]
                
                aggregated_thresholds[percentile] = {
                    "f1_mean": np.mean(f1s),
                    "f1_std": np.std(f1s),
                    "precision_mean": np.mean(precisions),
                    "precision_std": np.std(precisions),
                    "recall_mean": np.mean(recalls),
                    "recall_std": np.std(recalls)
                }
            
            aggregated[dataset_name][case_key] = {
                "auc_mean": np.mean(aucs),
                "auc_std": np.std(aucs),
                "auprc_mean": np.mean(auprcs),
                "auprc_std": np.std(auprcs),
                "threshold_metrics": aggregated_thresholds,
                "raw_scores_per_seed": scores_list,  # Falls du die Rohdaten brauchst
                "raw_threshold_metrics_per_seed": threshold_list
            }

    return aggregated



def plot_score_models_comparison(all_results, score, metric, dataset_names):
    """
    Zeigt für EINEN Score alle Modelle auf allen Datasets
    metric: "auroc" oder "auprc"
    """
    fig, axs = plt.subplots(1, len(dataset_names), figsize=(14, 5))
    model_names = list(all_results[dataset_names[0]].keys())
    colors = {"ForestFlow_1": "blue", "ForestDiffusion_1": "green", "TCCM": "red"}
    
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

def plot_extreme_cases_comparison(all_extreme_cases, dataset_name, metric="f1"):
    """Vergleicht no_contamination vs full_contamination für alle Scores"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    percentiles = [70, 80, 90, 95, 97.5, 99]
    
    for idx, (model_name, scores_data) in enumerate(all_extreme_cases.items()):
        for score_idx, (score, datasets) in enumerate(scores_data.items()):
            ax = axes[score_idx]
            
            metrics_no = datasets[dataset_name]["no_contamination"]["threshold_metrics"]
            metrics_full = datasets[dataset_name]["full_contamination"]["threshold_metrics"]
            
            values_no = [metrics_no[p][f"{metric}_mean"] for p in percentiles]
            stds_no = [metrics_no[p][f"{metric}_std"] for p in percentiles]
            
            values_full = [metrics_full[p][f"{metric}_mean"] for p in percentiles]
            stds_full = [metrics_full[p][f"{metric}_std"] for p in percentiles]
            
            x = np.arange(len(percentiles))
            width = 0.35
            
            ax.bar(x - width/2, values_no, width, label='No Contamination', 
                   yerr=stds_no, capsize=3, color='steelblue')
            ax.bar(x + width/2, values_full, width, label='Full Contamination', 
                   yerr=stds_full, capsize=3, color='coral')
            
            ax.set_xlabel('Percentile')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{score.capitalize()} Score')
            ax.set_xticks(x)
            ax.set_xticklabels(percentiles)
            ax.legend()
            ax.set_ylim(0, 0.6)
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - {metric.upper()} by Percentile Threshold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./results_ablation/extreme_comparison_{metric}.pdf")
    plt.show()

def plot_precision_recall_tradeoff(all_extreme_cases, dataset_name):
    """Zeigt Precision vs Recall für verschiedene Thresholds"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for model_name, scores_data in all_extreme_cases.items():
        for case_idx, case_key in enumerate(["no_contamination", "full_contamination"]):
            ax = axes[case_idx]
            
            for score, datasets in scores_data.items():
                metrics = datasets[dataset_name][case_key]["threshold_metrics"]
                
                precisions = [metrics[p]["precision_mean"] for p in [70, 80, 90, 95, 97.5, 99]]
                recalls = [metrics[p]["recall_mean"] for p in [70, 80, 90, 95, 97.5, 99]]
                
                ax.plot(recalls, precisions, marker='o', label=score, markersize=8)
                
                # Annotate mit Percentilen
                for i, p in enumerate([70, 80, 90, 95, 97.5, 99]):
                    ax.annotate(f'{p}', (recalls[i], precisions[i]), 
                               textcoords="offset points", xytext=(5,5), fontsize=8)
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{case_key.replace("_", " ").title()}')
            ax.legend()
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0.2, 0.7)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - Precision-Recall Tradeoff', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./results_ablation/precision_recall_tradeoff.pdf")
    plt.show()


def plot_score_comparison(all_extreme_cases, dataset_name):
    """Vergleicht alle 3 Scores nebeneinander"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for model_name, scores_data in all_extreme_cases.items():
        for case_idx, case_key in enumerate(["no_contamination", "full_contamination"]):
            ax = axes[case_idx]
            
            scores = list(scores_data.keys())
            x = np.arange(3)  # AUC, AUPRC, Best F1
            width = 0.25
            
            for i, score in enumerate(scores):
                data = scores_data[score][dataset_name][case_key]
                
                values = [
                    data["auc_mean"],
                    data["auprc_mean"],
                    max(data["threshold_metrics"].items(), key=lambda x: x[1]["f1_mean"])[1]["f1_mean"]
                ]
                stds = [
                    data["auc_std"],
                    data["auprc_std"],
                    max(data["threshold_metrics"].items(), key=lambda x: x[1]["f1_mean"])[1]["f1_std"]
                ]
                
                ax.bar(x + i*width, values, width, label=score, yerr=stds, capsize=3)
            
            ax.set_ylabel('Score')
            ax.set_title(f'{case_key.replace("_", " ").title()}')
            ax.set_xticks(x + width)
            ax.set_xticklabels(['AUC', 'AUPRC', 'Best F1'])
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - Score Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./results_ablation/score_comparison.pdf")
    plt.show()


def plot_f1_heatmap(all_extreme_cases, dataset_name):
    """Heatmap: Scores x Percentiles für beide Cases"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    percentiles = [90, 95, 97.5, 99, 99.5, 99.9]
    
    for model_name, scores_data in all_extreme_cases.items():
        for case_idx, case_key in enumerate(["no_contamination", "full_contamination"]):
            ax = axes[case_idx]
            
            scores = list(scores_data.keys())
            data_matrix = []
            
            for score in scores:
                metrics = scores_data[score][dataset_name][case_key]["threshold_metrics"]
                row = [metrics[p]["f1_mean"] for p in percentiles]
                data_matrix.append(row)
            
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.5)
            
            ax.set_xticks(range(len(percentiles)))
            ax.set_xticklabels(percentiles)
            ax.set_yticks(range(len(scores)))
            ax.set_yticklabels(scores)
            ax.set_xlabel('Percentile')
            ax.set_ylabel('Score Type')
            ax.set_title(f'{case_key.replace("_", " ").title()}')
            
            # Werte in Zellen
            for i in range(len(scores)):
                for j in range(len(percentiles)):
                    ax.text(j, i, f'{data_matrix[i][j]:.3f}', ha='center', va='center', fontsize=9)
            
            plt.colorbar(im, ax=ax, label='F1 Score')
    
    plt.suptitle(f'{dataset_name} - F1 Score Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./results_ablation/f1_heatmap.pdf")
    plt.show()


if __name__ == "__main__":
    dataset_names = ["29_Pima.npz"]

    models_to_run = {
        "TCCM": {
            "type": "tccm",
            "params": {
                "n_t": 200
            }
        }
    }
    
    all_results_combined = {}
    all_extreme_cases = {}

    for model_name, model_config in models_to_run.items():
        model_type = model_config["type"]
        params = model_config["params"]
        
        if model_type == "forest":
            if params["diffusion_type"] == "vp":
                score_list = ["deviation", "decision"]
            else:
                score_list = ["deviation", "reconstruction", "decision"]
        elif model_type == "tccm":
            score_list = ["deviation", "reconstruction", "decision"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        for score in score_list:
            results, contam_levels, extreme_cases = run_training_contamination_ablation_dynamic_fixed_split(
                score, dataset_names, (model_name, model_config)
            )
            
            for dataset_name, data in results.items():
                if dataset_name not in all_results_combined:
                    all_results_combined[dataset_name] = {}
                if model_name not in all_results_combined[dataset_name]:
                    all_results_combined[dataset_name][model_name] = {}
                all_results_combined[dataset_name][model_name][score] = data
            
            # Sammle Extremfälle (aggregiert)
            aggregated = aggregate_extreme_cases(extreme_cases)
            
            if model_name not in all_extreme_cases:
                all_extreme_cases[model_name] = {}
            all_extreme_cases[model_name][score] = aggregated

    # Plots
    for model_name in models_to_run.keys():
        plot_model_scores_comparison(all_results_combined, model_name, metric="auroc", dataset_names=dataset_names)
        plot_model_scores_comparison(all_results_combined, model_name, metric="auprc", dataset_names=dataset_names)

    for score in ["deviation", "reconstruction", "decision"]:
        plot_score_models_comparison(all_results_combined, score, metric="auroc", dataset_names=dataset_names)
        plot_score_models_comparison(all_results_combined, score, metric="auprc", dataset_names=dataset_names)

    # Ausgabe der Extremfälle
    print("\n" + "="*80)
    print("EXTREME CASES SUMMARY")
    print("="*80)
    for model_name, scores_data in all_extreme_cases.items():
        for score, datasets in scores_data.items():
            print(f"\n{model_name} - {score}:")
            for dataset_name, cases in datasets.items():
                for case_key, metrics in cases.items():
                    print(f"  {dataset_name} [{case_key}]:")
                    print(f"    AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
                    print(f"    AUPRC: {metrics['auprc_mean']:.4f} ± {metrics['auprc_std']:.4f}")

                    print(f"\n    Threshold Percentile Metrics:")
                    print(f"    {'Percentile':<12} {'Precision':<20} {'Recall':<20} {'F1':<20}")
                    print(f"    {'-'*72}")
                    
                    for percentile, thresh_metrics in metrics['threshold_metrics'].items():
                        prec = f"{thresh_metrics['precision_mean']:.4f} ± {thresh_metrics['precision_std']:.4f}"
                        rec = f"{thresh_metrics['recall_mean']:.4f} ± {thresh_metrics['recall_std']:.4f}"
                        f1 = f"{thresh_metrics['f1_mean']:.4f} ± {thresh_metrics['f1_std']:.4f}"
                        print(f"    {percentile:<12} {prec:<20} {rec:<20} {f1:<20}")
                    
                    # Bestes Percentil nach F1
                    best_p = max(metrics['threshold_metrics'].items(), 
                                key=lambda x: x[1]['f1_mean'])
                    print(f"\n    ✓ Best F1: {best_p[1]['f1_mean']:.4f} @ {best_p[0]}th percentile")
    plot_extreme_cases_comparison(all_extreme_cases, dataset_names[0], metric="f1")
    plot_extreme_cases_comparison(all_extreme_cases, dataset_names[0], metric="precision")
    plot_extreme_cases_comparison(all_extreme_cases, dataset_names[0], metric="recall")
    plot_precision_recall_tradeoff(all_extreme_cases, dataset_names[0])
    plot_score_comparison(all_extreme_cases, dataset_names[0])
    plot_f1_heatmap(all_extreme_cases, dataset_names[0])







# if __name__ == "__main__":
#     dataset_names = ["29_Pima.npz"]
#                     #["5_campaign.npz"]
#                      #"18_Ionosphere.npz"]


#     #attention with the names! not 2 times the same name
#     #update the names and colors in the plot functions 

#     models_to_run = {
#         # "ForestFlow_1": {
#         #     "type": "forest",
#         #     "params": {
#         #         "n_t": 20,
#         #         "duplicate_K": 20,
#         #         "duplicate_K_test": 20,
#         #         "diffusion_type": "flow"
#         #     }
#         # },
#         # "ForestDiffusion_1": {
#         #     "type": "forest",
#         #     "params": {
#         #         "n_t": 100,
#         #         "duplicate_K": 20,
#         #         "duplicate_K_test": 20,
#         #         "diffusion_type": "vp"
#         #     }
#         # },
#         "TCCM": {
#             "type": "tccm",
#             "params": {
#                 "n_t": 400   # only used for scoring functions
#             }
#         }
#     }   
#     # for model in models_to_run.items():
#     #     for score in ["deviation", "reconstruction", "decision"]:
#     #         results, contamination_levels = run_training_contamination_ablation_dynamic_fixed_split(score, dataset_names, model)
#     #         results
#     #         plot_training_contamination_ablation_dynamic(results, contamination_levels, model_name=model[0], score=score)
#     all_results_combined = {}

#     for model_name, model_config in models_to_run.items():
#         model_type = model_config["type"]
#         params = model_config["params"]       
#         if model_type == "forest":
#             if params["diffusion_type"] == "vp":
#                 score_list = ["deviation", "decision"]
#             elif params["diffusion_type"] == "flow":
#                 score_list = ["deviation", "reconstruction", "decision"]
#             else:
#                 score_list = ["deviation", "reconstruction", "decision"]  
#         elif model_type == "tccm":
#             score_list = ["deviation", "reconstruction", "decision"]
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")

#         for score in score_list:
#             results = run_training_contamination_ablation_dynamic_fixed_split(score, dataset_names, (model_name, model_config) )
#             for dataset_name, data in results.items():
#                 if dataset_name not in all_results_combined:
#                     all_results_combined[dataset_name] = {}
#                 if model_name not in all_results_combined[dataset_name]:
#                     all_results_combined[dataset_name][model_name] = {}
#                 all_results_combined[dataset_name][model_name][score] = data
#     for model_name in models_to_run.keys():
#         plot_model_scores_comparison(all_results_combined, model_name, metric="auroc", dataset_names=dataset_names)
#         plot_model_scores_comparison(all_results_combined, model_name, metric="auprc", dataset_names=dataset_names)

#     for score in ["deviation", "reconstruction", "decision"]:
#         plot_score_models_comparison(all_results_combined, score, metric="auroc", dataset_names=dataset_names)
#         plot_score_models_comparison(all_results_combined, score, metric="auprc", dataset_names=dataset_names)




