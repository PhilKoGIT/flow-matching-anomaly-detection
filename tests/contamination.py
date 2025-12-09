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

percentiles = [20, 30, 40, 50, 70, 80, 90, 95, 97.5, 99]


"""
File runs contamination studies for the given models and datasets.

"""

# # ============================================================================
# # adapted/copied from https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py
# # ============================================================================

def load_adbench_npz(dataset_name, test_size=0.5, random_state=42):
    #part copied from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py

    """
    loads a npz dataset and does the printed split

    """

    base_dir = Path(__file__).resolve().parent
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
    """

    creates a ForestDiffusionModel with given parameters

    """
    model = ForestDiffusionModel(
        X=X_train,
        label_y=None,     
        n_t=n_t,
        duplicate_K=duplicate_K,
        diffusion_type=diffusion_type, 
        eps=1e-3,
        model='xgboost',
        max_depth=7,  #use default hyperparameters for xgboost
        n_estimators=100,
        eta=0.3,
        gpu_hist=False,  
        n_batch=1,      
        seed=seed,
        n_jobs=-1,
        bin_indexes=[],      #no categorical, binary or integer features in the datasets used here, only floats
        cat_indexes=[],       
        int_indexes=[],
        remove_miss=False,
        p_in_one=True,      
    )
    print(f"\nTraining ForestDiffusionModel ({diffusion_type}) on dataset: {dataset_name} with n_t={n_t}, duplicate_K={duplicate_K}, seed={seed}")
    return model





def calculate_scores_ForestDiffusionModel(X_test, model, n_t, duplicate_K_test, diffusion_type, score):
    """
    Calculates the  right score type for the given ForestDiffusionModel (flow type or vp type)
    Selects the right funcition to call

    """

    if score == "deviation":
        if diffusion_type == "flow":
            anomaly_scores_deviation = model.compute_deviation_score(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test
            )
        elif diffusion_type == "vp":
            anomaly_scores_deviation = model.compute_deviation_score_vp(
                X_test, 
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test, 
                diffusion_type=diffusion_type
            )
        else: 
            raise ValueError(f"Unknown diffusion type: {diffusion_type}")
        return anomaly_scores_deviation

    elif score == "reconstruction":
        if diffusion_type == "flow":
            anomaly_scores_reconstruction = model.compute_reconstruction_score(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test
            )
        elif diffusion_type == "vp": 
            anomaly_scores_reconstruction = model.compute_reconstruction_score_vp(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=duplicate_K_test
            )
        else:
            raise ValueError(f"Unknown diffusion type: {diffusion_type}")
        return anomaly_scores_reconstruction
    elif score == "decision":
        if diffusion_type == "flow":
            anomaly_scores_decision = model.compute_decision_score(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=2*duplicate_K_test
            )
        elif diffusion_type == "vp":
            anomaly_scores_decision = model.compute_decision_score_vp(
                test_samples=X_test,
                diffusion_type=diffusion_type,
                n_t=n_t, 
                duplicate_K_test=2*duplicate_K_test
            )
        else: 
            raise ValueError(f"Unknown diffusion type: {diffusion_type}")
        return anomaly_scores_decision
    else: 
        raise ValueError(f"Unknown score type: {score}")


def create_trained_tccm_model(X_train, dataset_name, seed):
    """
    
    Creates and trains a TCCM model with the hyperparameters for the specific dataset from this model https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/FMAD/functions.py

    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # take the hyperparamters for the specific dataset
    hyperparams = determine_FMAD_hyperparameters(dataset_name)
    
    print(f"\nTraining TCCM with hyperparameters:")
    print(f"  Epochs: {hyperparams['epochs']}")
    print(f"  Learning Rate: {hyperparams['learning_rate']}")
    print(f"  Batch Size: {hyperparams['batch_size']}")

    # Initialize TCCM with the hyperparameters
    model = TCCM(
        n_features=X_train.shape[1],
        epochs=hyperparams['epochs'],
        learning_rate=hyperparams['learning_rate'],
        batch_size=hyperparams['batch_size']
    )
    
    # Train the model
    model.fit(X_train)
    print(f"TCCM model trained on dataset: {dataset_name} with seed: {seed}")

    return model


def calculate_tccm_scores(X_test, model, n_t, score):
    """
    Calculates the right score type for the TCCM model

    """

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
# # based on https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/AblationStudies.py
# # ============================================================================
#for saving which the models that were already trained (bc scoring function is independet of model training)
tccm = False
forestdiffusion = False
forestflow = False


def run_training_contamination_ablation_dynamic_fixed_split(score, dataset_names, model):

    """
    Runs training and evaluation for different contamination levels on fixed train/test split. A model is trained for each seed and contamination level.
    
    extended to save extreme cases data for (almost) no contamination and full contamination

    """
    seed_list = [0,1,2,3,4]
    all_results = {}
    all_contam_levels = {}

    #safe for all maximum contamination levels for each dataset and for (almost) zero contamination
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

            is_no_contam = (contam_idx == 0)  
            is_full_contam = (contam_idx == len(contamination_levels) - 1)  

            for seed in seed_list:
                np.random.seed(seed)
                random.seed(seed)

                n_ab = int(contam_ratio * n_train_normal / (1 - contam_ratio))
                n_ab = min(n_ab, n_train_abnormal_max)

                #adds the abnormal data to the training set based on the current contamination ratio
                selected_ab = X_train_abnormal_full[:n_ab]  
                X_train = np.vstack([X_train_normal, selected_ab])

                p = model_cnf["params"]
                if model_cnf["type"] == "forest":
                    model = create_ForestDiffusionModel(
                        n_t=p["n_t"],
                        duplicate_K=p["duplicate_K"],
                        seed=seed,
                        X_train=X_train,
                        dataset_name=f"{dataset_name}_{model_name}",
                        diffusion_type=p["diffusion_type"]
                    )
                    scores = calculate_scores_ForestDiffusionModel(X_test, model, n_t=p["n_t"], duplicate_K_test=p["duplicate_K_test"], diffusion_type=p["diffusion_type"], score=score)
                elif model_cnf["type"] == "tccm":
                    model = create_trained_tccm_model(
                        X_train=X_train,
                        dataset_name=f"{dataset_name}_{model_name}",
                        seed=seed
                    )
                    scores = calculate_tccm_scores(X_test, model, n_t=p["n_t"], score=score)
                else:
                    raise ValueError(f"Unknown model type: {model_cnf['type']}")

                auc = roc_auc_score(y_test, scores)
                pr = average_precision_score(y_test, scores)
                aucs.append(auc)
                prs.append(pr)

                #Safes extreme cases
                if is_no_contam or is_full_contam:
                    case_key = "no_contamination" if is_no_contam else "full_contamination"
                    extreme_cases_data[dataset_name][case_key]["scores_per_seed"].append({
                        "seed": seed,
                        "anomaly_scores": scores.copy(),
                        "auc": auc,
                        "auprc": pr
                    })   
                    # Calculate and save threshold metrics
                    threshold_metrics = compute_threshold_metrics(scores, y_test)
                    threshold_metrics["seed"] = seed
                    extreme_cases_data[dataset_name][case_key]["threshold_metrics_per_seed"].append(threshold_metrics)
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


# # ============================================================================
# #  
# # ============================================================================

#needed for theshold metrics saving for extreme cases
def compute_threshold_metrics(anomaly_scores, y_test):
    """Calculates metrics for different threshold percentiles"""
    
    thresholds_percentiles = percentiles
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



def aggregate_extreme_cases(extreme_cases_data):
    """Evaluaates the extreme cases data and aggregates the results across seeds"""
    aggregated = {}

    for dataset_name, cases in extreme_cases_data.items():
        aggregated[dataset_name] = {}
        
        for case_key in ["no_contamination", "full_contamination"]:
            case_data = cases[case_key]
            scores_list = case_data["scores_per_seed"]
            threshold_list = case_data["threshold_metrics_per_seed"]
            
            if not scores_list:
                continue
            
            # Aggregate AUC and AUPRC
            aucs = [s["auc"] for s in scores_list]
            auprcs = [s["auprc"] for s in scores_list]
            
            # Aggregate Threshold Metrics
            aggregated_thresholds = {}
            for percentile in percentiles:
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



def plot_score_models_comparison(all_results, score, metric, dataset_names, model_names):
    """
    Shows for one score all models comparison plots for all datasets
    metric: "auroc" or "auprc"
    
    """
    fig, axs = plt.subplots(1, len(dataset_names), figsize=(14, 5))
    model_names = list(all_results[dataset_names[0]].keys())

    # For each model name a color
    colors = {"ForestFlow_n20_k10": "blue", "ForestDiffusion_n50_k10": "green", "TCCM_n100": "red"}
    #colors = {"TCCM_n_t_20": "red", "TCCM_n_t_300": "green"}
    for idx, dataset_name in enumerate(dataset_names):
        ax = axs[idx] if len(dataset_names) > 1 else axs
        
        for model_name in model_names:

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


def plot_model_scores_comparison(all_results, model_name, metric, dataset_names):
    """
    Shows for one model all 3 scores on all datasets
    metric: "auroc" or "auprc"

    """
    fig, axs = plt.subplots(1, len(dataset_names), figsize=(14, 5))
    
    first_dataset = dataset_names[0]
    scores = list(all_results[first_dataset][model_name].keys())
    
    # Define colors for all possible scores
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



if __name__ == "__main__":

    """"Letze score funktion noch implementieren!!!!!!"""

    #dataset_names = ["29_Pima.npz"]
    #dataset_names = ["29_Pima.npz"]
    dataset_names = ["5_campaign.npz"]

    #MAX three models!

    #Change names in plot_score_models_comparison!!

    models_to_run = {

        "ForestFlow_n20_k10": {
            "type": "forest",
            "params": {
                "n_t": 20,
                "duplicate_K": 10,
                "duplicate_K_test": 10,
                "diffusion_type": "flow"
            },
        },
        "ForestDiffusion_n50_k10": {
            "type": "forest",
            "params": {
                "n_t": 50,
                "duplicate_K": 10,
                "duplicate_K_test": 10,
                "diffusion_type": "vp"
            },
        },
        "TCCM_n100": {
            "type": "tccm",
            "params": {
                "n_t": 100
            }
        }


        # "ForestFlow_n5_k5": {
        #     "type": "forest",
        #     "params": {
        #         "n_t": 20,
        #         "duplicate_K": 10,
        #         "duplicate_K_test": 10,
        #         "diffusion_type": "flow"
        #     },
        # },
        # "ForestDiffusion_n5_k5": {
        #     "type": "forest",
        #     "params": {
        #         "n_t": 50,
        #         "duplicate_K": 10,
        #         "duplicate_K_test": 10,
        #         "diffusion_type": "vp"
        #     },
        # },
        # "TCCM_n200": {
        #     "type": "tccm",
        #     "params": {
        #         "n_t": 50
        #     }
        # }
     }
    
    all_results_combined = {}
    all_extreme_cases = {}

    for model_name, model_config in models_to_run.items():
        model_type = model_config["type"]
        params = model_config["params"]
        #duplicate_K and duplicate_K_test should be the same for contamination studies
        assert params.get("duplicate_K") == params.get("duplicate_K_test"), "duplicate_K and duplicate_K_test must be the same, for simplicity"
        if model_type == "forest":
            score_list =["deviation", "reconstruction", "decision"]            
        elif model_type == "tccm":
            score_list = ["deviation","reconstruction", "decision"]
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
            
            # Aggregate extreme cases
            aggregated = aggregate_extreme_cases(extreme_cases)
            
            if model_name not in all_extreme_cases:
                all_extreme_cases[model_name] = {}
            all_extreme_cases[model_name][score] = aggregated

    # Plots
    for model_name in models_to_run.keys():
        plot_model_scores_comparison(all_results_combined, model_name, metric="auroc", dataset_names=dataset_names)
        plot_model_scores_comparison(all_results_combined, model_name, metric="auprc", dataset_names=dataset_names)

    for score in ["deviation", "reconstruction", "decision"]:
        plot_score_models_comparison(all_results_combined,  score, metric="auroc", dataset_names=dataset_names, model_names=list(models_to_run.keys()))
        plot_score_models_comparison(all_results_combined, score, metric="auprc", dataset_names=dataset_names, model_names=list(models_to_run.keys()))

    # Printing extreme cases summary
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


