import random
import os
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



# HINWEIS: Wir benötigen die preprocess-Funktion und die EFVFMDataset-Klasse NICHT mehr direkt,
# da wir die fertigen, vom Preprocessing erstellten .npy-Dateien laden.
# Es genügt, die .npy und .json Dateien zu laden, die du im letzten Schritt erstellt hast.

def load_dataset(dataset_name, semi_supervised):

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"

    if "business_dataset" in dataset_name:
        #load business datasets, preprocessed supervised or unsupervised
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

        print(f"  Train: {np.sum(y_train==0)} normal, {np.sum(y_train==1)} anomalies ({np.sum(y_train==1)/len(y_train)*100}%)")
        print(f"  Test:  {np.sum(y_test==0)} normal, {np.sum(y_test==1)} anomalies ({np.sum(y_test==1)/len(y_test)*100:}%)")
        
    elif dataset_name.endswith(".npz"):

        #------ copied!!!!!!! --------
        #copied partly from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py
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
            #case in the contamination study 
            pass
        #-----------------------------------
            
    # NEUE SEKTION: Lade die EF-VFM-vorverarbeiteten Daten
    elif dataset_name.endswith("_efvfm"):
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

def create_ForestDiffusionModel(n_t, duplicate_K, seed, X_train, dataset_name, diffusion_type):
    """
    Create and train a ForestDiffusionModel with given hyperparameters.

    """
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
        n_estimators=100,
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

    return model, time_train


def calculate_scores_ForestDiffusionModel(X_test, y_test, model, n_t, duplicate_K_test, diffusion_type):
    """
    Calculate all three ForestDiffusion anomaly scores
    1. Decision Function Score
    2. Deviation Score
    3. Reconstruction Score
    """

    # ---Computation deviation Score ---
    if diffusion_type == "flow":
        start_time_deviation = time.time()
        anomaly_scores_deviation = model.compute_deviation_score(
            test_samples=X_test,
            diffusion_type=diffusion_type,
            n_t=n_t, 
            duplicate_K_test=duplicate_K_test
        )
        end_time_deviation = time.time()
        time_deviation = end_time_deviation - start_time_deviation

        auroc_deviation = roc_auc_score(y_test, anomaly_scores_deviation)
        auprc_deviation = average_precision_score(y_test, anomaly_scores_deviation)
  

        #---Computation reconstruction Score ---
        start_time_reconstruction = time.time()
        anomaly_scores_reconstruction = model.compute_reconstruction_score(
            test_samples=X_test,
            diffusion_type=diffusion_type,
            n_t=n_t, 
            duplicate_K_test=duplicate_K_test
        )
        end_time_reconstruction = time.time()
        time_reconstruction = end_time_reconstruction - start_time_reconstruction
        
        auroc_reconstruction = roc_auc_score(y_test, anomaly_scores_reconstruction)
        auprc_reconstruction = average_precision_score(y_test, anomaly_scores_reconstruction)


        #---Computation decision Score ---
        start_time_decision = time.time()
        anomaly_scores_decision = model.compute_decision_score(
            test_samples=X_test,
            diffusion_type=diffusion_type,
            n_t=n_t, 
            duplicate_K_test=duplicate_K_test
        )
        end_time_decision = time.time()
        time_decision = end_time_decision - start_time_decision
        
        auroc_decision = roc_auc_score(y_test, anomaly_scores_decision)
        auprc_decision = average_precision_score(y_test, anomaly_scores_decision)
  
    elif diffusion_type == "vp":
        # ---Computation deviation Score ---
        start_time_deviation = time.time()
        anomaly_scores_deviation = model.compute_deviation_score_vp(
            test_samples=X_test,
            diffusion_type=diffusion_type,
            n_t=n_t, 
            duplicate_K_test=duplicate_K_test
        )
        end_time_deviation = time.time()
        time_deviation = end_time_deviation - start_time_deviation

        auroc_deviation = roc_auc_score(y_test, anomaly_scores_deviation)
        auprc_deviation = average_precision_score(y_test, anomaly_scores_deviation)
        
        #---Computation decision Score ---
        #no reconstruction score for vp diffusion implemented

        auroc_reconstruction = 0
        auprc_reconstruction = 0
        time_reconstruction = 0
        anomaly_scores_reconstruction = 0
        

        #---Computation decision Score ---
        start_time_decision = time.time()
        anomaly_scores_decision = model.compute_decision_score_vp(
            test_samples=X_test,
            diffusion_type=diffusion_type,
            n_t=n_t, 
            duplicate_K_test=duplicate_K_test
        )
        end_time_decision = time.time()
        time_decision = end_time_decision - start_time_decision
        
        auroc_decision = roc_auc_score(y_test, anomaly_scores_decision)
        auprc_decision = average_precision_score(y_test, anomaly_scores_decision)
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")

    return {
        'decision': {
            'auroc': auroc_decision,
            'auprc': auprc_decision,
            'time': time_decision,
            'scores': anomaly_scores_decision
        },
        'deviation': {
            'auroc': auroc_deviation,
            'auprc': auprc_deviation,
            'time': time_deviation,
            'scores': anomaly_scores_deviation
        },
        'reconstruction': {
            'auroc': auroc_reconstruction,
            'auprc': auprc_reconstruction,
            'time': time_reconstruction,
            'scores': anomaly_scores_reconstruction
        }
    }


def create_trained_tccm_model(X_train, dataset_name, seed):
    """
    Creates and trains a TCCM model with the hyperparameters for the specific dataset from this model https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/FMAD/functions.py

    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Get the optimal hyperparameters for the dataset
    hyperparams = determine_FMAD_hyperparameters(dataset_name)
    start_time = time.time()
    
    # Initialize TCCM with the hyperparameters
    model = TCCM(
        n_features=X_train.shape[1],
        epochs=hyperparams['epochs'],
        learning_rate=hyperparams['learning_rate'],
        batch_size=hyperparams['batch_size']
    )
    
    model.fit(X_train)
    
    end_time_train = time.time()
    time_train = end_time_train - start_time
    
    
    return model, time_train



def calculate_tccm_scores(X_test, y_test, model, n_t):
    """
    Calculates the three anomaly scores for a tccm model::
    1. Decision Function Score  
    2. Deviation Score
    3. Reconstruction Score
    """

    # --- Score 1: Decision Function  ---
    start_time_decision = time.time()
    anomaly_scores_decision = model.decision_function(X_test)
    end_time_decision = time.time()
    time_decision = end_time_decision - start_time_decision
    
    auroc_decision = roc_auc_score(y_test, anomaly_scores_decision)
    auprc_decision = average_precision_score(y_test, anomaly_scores_decision)
    
    # --- Score 2: Deviation Score  ---
    start_time_deviation = time.time()
    anomaly_scores_deviation = model.compute_deviation_score(X_test, n_t=n_t)
    end_time_deviation = time.time()
    time_deviation = end_time_deviation - start_time_deviation
    
    auroc_deviation = roc_auc_score(y_test, anomaly_scores_deviation)
    auprc_deviation = average_precision_score(y_test, anomaly_scores_deviation)

    # --- Score 3: Reconstruction Score  ---

    start_time_reconstruction = time.time()
    anomaly_scores_reconstruction = model.compute_reconstruction_score(X_test, n_t = n_t)
    end_time_reconstruction = time.time()
    time_reconstruction = end_time_reconstruction - start_time_reconstruction
    
    auroc_reconstruction = roc_auc_score(y_test, anomaly_scores_reconstruction)
    auprc_reconstruction = average_precision_score(y_test, anomaly_scores_reconstruction)

    
    return {
        'decision': {
            'auroc': auroc_decision,
            'auprc': auprc_decision,
            'time': time_decision,
            'scores': anomaly_scores_decision
        },
        'deviation': {
            'auroc': auroc_deviation,
            'auprc': auprc_deviation,
            'time': time_deviation,
            'scores': anomaly_scores_deviation
        },
        'reconstruction': {
            'auroc': auroc_reconstruction,
            'auprc': auprc_reconstruction,
            'time': time_reconstruction,
            'scores': anomaly_scores_reconstruction
        }
    }


def evaluate_thresholds(anomaly_scores, y_test, score_name="Score"):
    """
    Calculates Precision, Recall, and F1 for various percentile thresholds
    """
    percentiles = [60, 70, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9]

    print(f"\n{'='*90}")
    print(f"THRESHOLD ANALYSIS: {score_name}")
    print(f"{'='*90}")
    print(f"{'Percentile':<12} {'Threshold':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print(f"{'-'*90}")

    best_f1 = 0
    best_percentile = None

    for p in percentiles:
        threshold = np.percentile(anomaly_scores, p)
        preds = (anomaly_scores > threshold).astype(int)
        
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Mark best F1
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_percentile = p
        
        print(f"{p:<12} {threshold:>12.4f} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f}")

    print(f"{'-'*90}")
    print(f"✓ Best F1: {best_f1:.4f} @ {best_percentile}th percentile")
    print(f"{'='*90}")

    return best_percentile, best_f1

def compute_threshold_metrics(anomaly_scores, y_test, score_name):
    """Calculates metrics for different threshold percentiles"""
    percentiles = [60, 70, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9]
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


if __name__ == "__main__":

    dataset_names = {
        "Campaign":{
            "file": "5_campaign.npz",
            "semi_supervised": True,
        }
        # "business_dataset_semi":{
        #     "file": "business_dataset.csv",  
        #     "semi_supervised": True,    
        # },
        # "business_dataset_un":{
        #     "file": "business_dataset.csv",  
        #     "semi_supervised": False,    
        # }
        #"Fraud":{
        #    "file": "13_fraud.npz",   
        #    "semi_supervised": True,
        #    "type": "efvfm"
    }

    models_to_run = {

        "ForestFlow_n20_k20": {
            "type": "forest",
            "params": {
                "n_t": 20,
                "duplicate_K": 20,
                "duplicate_K_test": 20,
                "diffusion_type": "flow"
            },
        },
        "ForestDiffusion_n50_k20": {
            "type": "forest",
            "params": {
                "n_t": 50,
                "duplicate_K": 20,
                "duplicate_K_test": 20,
                "diffusion_type": "vp"
            },
        },
        "TCCM_n200": {
            "type": "tccm",
            "params": {
                "n_t": 200
            }
        }
    }   

    for dataset_name, dataset_info in dataset_names.items():
        X_train, X_test, y_test = load_dataset(dataset_info["file"], semi_supervised=dataset_info["semi_supervised"])
        dataset_results = {}

        for model_name, cfg in models_to_run.items():
            print("\n" + "#" * 80)
            print(f"Running model: {model_name} on dataset: {dataset_name}")
            print("#" * 80)
            assert cfg["params"].get("duplicate_K") == cfg["params"].get("duplicate_K_test"), "duplicate_K and duplicate_K_test must be the same for simplicity reason"

            decision_auroc_list = []
            decision_auprc_list = []
            deviation_auroc_list = []
            deviation_auprc_list = []
            reconstruction_auroc_list = []
            reconstruction_auprc_list = []

            time_train_list = []
            time_decision_list = []
            time_deviation_list = []
            time_reconstruction_list = []

            for i in range(5):
                #create the correct model and calculate scores
                if cfg["type"] == "forest":
                    p = cfg["params"]
                    model, time_train = create_ForestDiffusionModel(
                        n_t=p["n_t"],
                        duplicate_K=p["duplicate_K"],
                        seed=i,
                        X_train=X_train,
                        dataset_name=f"{dataset_name}_{model_name}",
                        diffusion_type=p["diffusion_type"]
                    )

                    results = calculate_scores_ForestDiffusionModel(
                        X_test, y_test, model,
                        p["n_t"],
                        p["duplicate_K_test"],
                        p["diffusion_type"]
                    )

                elif cfg["type"] == "tccm":
                    p = cfg["params"]
                    model, time_train = create_trained_tccm_model(
                        X_train=X_train,
                        dataset_name=f"{dataset_name}_{model_name}",
                        seed=i
                    )

                    results = calculate_tccm_scores(
                        X_test, y_test, model, n_t=p["n_t"]
                    )
                if i == 4:  # Only print once at the end
                    print(f"\n{'#'*80}")
                    print(f"THRESHOLD ANALYSIS FOR {model_name}")
                    print(f"{'#'*80}")
                    compute_threshold_metrics(results['decision']['scores'], y_test, f"{model_name} - Decision")
                    compute_threshold_metrics(results['deviation']['scores'], y_test, f"{model_name} - Deviation")
                    if cfg["params"].get("diffusion_type") != "vp":
                        compute_threshold_metrics(results['reconstruction']['scores'], y_test, f"{model_name} - Reconstruction")

                # Save results
                time_train_list.append(time_train)

                decision_auroc_list.append(results['decision']['auroc'])
                decision_auprc_list.append(results['decision']['auprc'])
                time_decision_list.append(results['decision']['time'])

                deviation_auroc_list.append(results['deviation']['auroc'])
                deviation_auprc_list.append(results['deviation']['auprc'])
                time_deviation_list.append(results['deviation']['time'])

                reconstruction_auroc_list.append(results['reconstruction']['auroc'])
                reconstruction_auprc_list.append(results['reconstruction']['auprc'])
                time_reconstruction_list.append(results['reconstruction']['time'])

            #calculate means and stds
            
            dataset_results[model_name] = {
                "dec_auroc_mean": np.mean(decision_auroc_list),
                "dec_auroc_std":  np.std(decision_auroc_list),
                "dec_auprc_mean": np.mean(decision_auprc_list),
                "dec_auprc_std":  np.std(decision_auprc_list),

                "dev_auroc_mean": np.mean(deviation_auroc_list),
                "dev_auroc_std":  np.std(deviation_auroc_list),
                "dev_auprc_mean": np.mean(deviation_auprc_list),
                "dev_auprc_std":  np.std(deviation_auprc_list),

                "rec_auroc_mean": np.mean(reconstruction_auroc_list),
                "rec_auroc_std":  np.std(reconstruction_auroc_list),
                "rec_auprc_mean": np.mean(reconstruction_auprc_list),
                "rec_auprc_std":  np.std(reconstruction_auprc_list),

                "train_time_mean": np.mean(time_train_list),
                "train_time_std":  np.std(time_train_list),

                "dec_time_mean": np.mean(time_decision_list),
                "dec_time_std":  np.std(time_decision_list),
                "dev_time_mean": np.mean(time_deviation_list),
                "dev_time_std":  np.std(time_deviation_list),
                "rec_time_mean": np.mean(time_reconstruction_list),
                "rec_time_std":  np.std(time_reconstruction_list),
            }

        print("\n" + "=" * 140)
        print(f"COMPARISON FOR DATASET: {dataset_name}")
        print("=" * 140)

        # Header for metrics
        header1 = (
            f"{'Model':<18} | "
            f"{'DecAUROC':<15} | {'DecAUPRC':<15} | "
            f"{'DevAUROC':<15} | {'DevAUPRC':<15} | "
            f"{'RecAUROC':<15} | {'RecAUPRC':<15}"
        )
        print(header1)
        print("-" * len(header1))

        for model_name, stats in dataset_results.items():
            line = (
                f"{model_name:<18} | "
                f"{stats['dec_auroc_mean']:.4f}±{stats['dec_auroc_std']:.4f} | "
                f"{stats['dec_auprc_mean']:.4f}±{stats['dec_auprc_std']:.4f} | "
                f"{stats['dev_auroc_mean']:.4f}±{stats['dev_auroc_std']:.4f} | "
                f"{stats['dev_auprc_mean']:.4f}±{stats['dev_auprc_std']:.4f} | "
                f"{stats['rec_auroc_mean']:.4f}±{stats['rec_auroc_std']:.4f} | "
                f"{stats['rec_auprc_mean']:.4f}±{stats['rec_auprc_std']:.4f}"
            )
            print(line)

        # Header for timing
        print("\n" + "-" * 100)
        print("TIMING (seconds):")
        print("-" * 100)
        header2 = (
            f"{'Model':<18} | "
            f"{'TrainT':<15} | {'DecT':<15} | {'DevT':<15} | {'RecT':<15} | {'TotalT':>10}"
        )
        print(header2)
        print("-" * len(header2))

        for model_name, stats in dataset_results.items():
            total_time = stats['train_time_mean'] + stats['dec_time_mean'] + stats['dev_time_mean'] + stats['rec_time_mean']
            line = (
                f"{model_name:<18} | "
                f"{stats['train_time_mean']:.2f}  | "
                f"{stats['dec_time_mean']:.4f}| "
                f"{stats['dev_time_mean']:.4f} | "
                f"{stats['rec_time_mean']:.4f}| "
                f"{total_time:>10.2f}"
            )
            print(line)
