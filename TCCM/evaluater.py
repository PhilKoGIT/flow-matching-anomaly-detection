import random
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
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
from FlowMatchingAD import TCCM
from functions import determine_FMAD_hyperparameters


def load_dataset(dataset_name, semi_supervised):
    """
    Lädt das Dataset (aktuell nur .npz-Dateien im semi-supervised Modus)
    """
    # Pfad zur Basisdatenbank
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"

    if dataset_name.endswith(".npz"):
        if semi_supervised:
            file_path = data_dir / dataset_name
            data = np.load(file_path, allow_pickle=True)
            X, y = data['X'], data['y'].astype(int)
            x_normal, X_anomalous = X[y == 0], X[y == 1]
            y_normal, y_anomalous = y[y == 0], y[y == 1]

            X_train_raw, X_test_normal_raw, y_train, y_test_normal = train_test_split(
                x_normal, y_normal, test_size=0.5, random_state=42
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
            
            return X_train, X_test, y_test
        else:
            raise NotImplementedError("Unsupervised mode not implemented yet")
    else:
        raise ValueError(f"Unbekanntes Dataset-Format: {dataset_name}")


def create_trained_tccm_model(X_train, dataset_name, seed=42):
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


def calculate_tccm_scores(X_test, y_test, model, n_t=10):
    """
    Berechnet alle drei TCCM Anomaly Scores:
    1. Decision Function Score (Equation 5)
    2. Deviation Score (über mehrere Zeitpunkte)
    3. Reconstruction Score (implizit in decision_function)
    """
    times = []
    
    # --- Score 1: Decision Function (Standard Anomaly Score) ---
    print("\n--- Computing Decision Function Score ---")
    start_time_decision = time.time()
    anomaly_scores_decision = model.decision_function(X_test)
    end_time_decision = time.time()
    time_decision = end_time_decision - start_time_decision
    times.append(time_decision)
    
    auroc_decision = roc_auc_score(y_test, anomaly_scores_decision)
    auprc_decision = average_precision_score(y_test, anomaly_scores_decision)
    print(f"Decision Function Score computed in {time_decision:.2f} seconds")
    print(f"  AUROC: {auroc_decision:.4f}")
    print(f"  AUPRC: {auprc_decision:.4f}")
    
    # --- Score 2: Deviation Score (über mehrere Zeitpunkte) ---
    print(f"\n--- Computing Deviation Score (n_t={n_t}) ---")
    start_time_deviation = time.time()
    anomaly_scores_deviation = model.compute_deviation_score(X_test, n_t=n_t)
    end_time_deviation = time.time()
    time_deviation = end_time_deviation - start_time_deviation
    times.append(time_deviation)
    
    auroc_deviation = roc_auc_score(y_test, anomaly_scores_deviation)
    auprc_deviation = average_precision_score(y_test, anomaly_scores_deviation)
    print(f"Deviation Score computed in {time_deviation:.2f} seconds")
    print(f"  AUROC: {auroc_deviation:.4f}")
    print(f"  AUPRC: {auprc_deviation:.4f}")
    
    # --- Score 3: Reconstruction Score (über Decision Function mit t=1) ---
    # Dies ist technisch das gleiche wie decision_function, aber wir nennen es
    # explizit "Reconstruction Score" für Konsistenz mit dem Framework
    print("\n--- Computing Reconstruction Score ---")
    start_time_reconstruction = time.time()
    anomaly_scores_reconstruction = model.compute_reconstruction_score(X_test, n_t = n_t)
    end_time_reconstruction = time.time()
    time_reconstruction = end_time_reconstruction - start_time_reconstruction
    times.append(time_reconstruction)
    
    auroc_reconstruction = roc_auc_score(y_test, anomaly_scores_reconstruction)
    auprc_reconstruction = average_precision_score(y_test, anomaly_scores_reconstruction)
    print(f"Reconstruction Score computed in {time_reconstruction:.2f} seconds")
    print(f"  AUROC: {auroc_reconstruction:.4f}")
    print(f"  AUPRC: {auprc_reconstruction:.4f}")
    
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


if __name__ == "__main__":
    # Konfiguration
    dataset_names = ["29_Pima.npz"]
    number_of_runs = 5
    n_t_deviation = 1000  # Anzahl der Zeitpunkte für Deviation Score
    
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name} - TCCM EVALUATION")
        print("=" * 80)
        
        # Listen für Ergebnisse über alle Runs
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
        
        # Lade Dataset einmalig (immer der gleiche Split für Vergleichbarkeit)
        X_train, X_test, y_test = load_dataset(dataset_name, semi_supervised=True)
        
        # --- Runs mit unterschiedlichen Seeds ---
        for i in range(number_of_runs):
            print(f"\n{'='*80}")
            print(f"--- Run {i+1}/{number_of_runs} (Seed: {i}) ---")
            print(f"{'='*80}")
            
            # Trainiere Modell
            model, time_train = create_trained_tccm_model(
                X_train=X_train,
                dataset_name=dataset_name,
                seed=i
            )
            time_train_list.append(time_train)
            
            # Berechne alle Scores
            results = calculate_tccm_scores(X_test, y_test, model, n_t=n_t_deviation)
            
            # Speichere Ergebnisse
            decision_auroc_list.append(results['decision']['auroc'])
            decision_auprc_list.append(results['decision']['auprc'])
            time_decision_list.append(results['decision']['time'])
            
            deviation_auroc_list.append(results['deviation']['auroc'])
            deviation_auprc_list.append(results['deviation']['auprc'])
            time_deviation_list.append(results['deviation']['time'])
            
            reconstruction_auroc_list.append(results['reconstruction']['auroc'])
            reconstruction_auprc_list.append(results['reconstruction']['auprc'])
            time_reconstruction_list.append(results['reconstruction']['time'])
            
            print(f"\nRun {i+1} Summary:")
            print(f"  Decision:       AUROC={results['decision']['auroc']:.4f}, AUPRC={results['decision']['auprc']:.4f}")
            print(f"  Deviation:      AUROC={results['deviation']['auroc']:.4f}, AUPRC={results['deviation']['auprc']:.4f}")
            print(f"  Reconstruction: AUROC={results['reconstruction']['auroc']:.4f}, AUPRC={results['reconstruction']['auprc']:.4f}")
        
        # --- Berechne Statistiken über alle Runs ---
        print("\n\n" + "=" * 90)
        print(" " * 30 + "FINAL RESULTS - TCCM")
        print("=" * 90)
        print(f"Dataset: {dataset_name}")
        print(f"Number of runs: {number_of_runs}")
        print(f"Deviation Score computed with n_t={n_t_deviation} time steps")
        print("-" * 90)
        
        # Decision Function Results
        print("\n--- DECISION FUNCTION SCORE ---")
        print(f"AUROC: {np.mean(decision_auroc_list):.4f} ± {np.std(decision_auroc_list):.4f}")
        print(f"AUPRC: {np.mean(decision_auprc_list):.4f} ± {np.std(decision_auprc_list):.4f}")
        print(f"All AUROC values: {[f'{x:.4f}' for x in decision_auroc_list]}")
        print(f"All AUPRC values: {[f'{x:.4f}' for x in decision_auprc_list]}")
        
        # Deviation Score Results
        print("\n--- DEVIATION SCORE ---")
        print(f"AUROC: {np.mean(deviation_auroc_list):.4f} ± {np.std(deviation_auroc_list):.4f}")
        print(f"AUPRC: {np.mean(deviation_auprc_list):.4f} ± {np.std(deviation_auprc_list):.4f}")
        print(f"All AUROC values: {[f'{x:.4f}' for x in deviation_auroc_list]}")
        print(f"All AUPRC values: {[f'{x:.4f}' for x in deviation_auprc_list]}")
        
        # Reconstruction Score Results
        print("\n--- RECONSTRUCTION SCORE ---")
        print(f"AUROC: {np.mean(reconstruction_auroc_list):.4f} ± {np.std(reconstruction_auroc_list):.4f}")
        print(f"AUPRC: {np.mean(reconstruction_auprc_list):.4f} ± {np.std(reconstruction_auprc_list):.4f}")
        print(f"All AUROC values: {[f'{x:.4f}' for x in reconstruction_auroc_list]}")
        print(f"All AUPRC values: {[f'{x:.4f}' for x in reconstruction_auprc_list]}")
        
        # Time Statistics
        print("\n" + "=" * 90)
        print(" " * 30 + "TIME RESULTS - TCCM")
        print("=" * 90)
        print(f"Training Time:       {np.mean(time_train_list):.3f} ± {np.std(time_train_list):.3f} seconds")
        print(f"Decision Inf. Time:  {np.mean(time_decision_list):.3f} ± {np.std(time_decision_list):.3f} seconds")
        print(f"Deviation Inf. Time: {np.mean(time_deviation_list):.3f} ± {np.std(time_deviation_list):.3f} seconds")
        print(f"Recon. Inf. Time:    {np.mean(time_reconstruction_list):.3f} ± {np.std(time_reconstruction_list):.3f} seconds")
        print(f"\nTotal Time (Training + Decision):      {np.mean(time_train_list) + np.mean(time_decision_list):.3f} seconds")
        print(f"Total Time (Training + Deviation):     {np.mean(time_train_list) + np.mean(time_deviation_list):.3f} seconds")
        print(f"Total Time (Training + Reconstruction): {np.mean(time_train_list) + np.mean(time_reconstruction_list):.3f} seconds")
        print("=" * 90)
        
        # Vergleichstabelle im ForestDiffusion-Stil
        print("\n" + "=" * 90)
        print(" " * 25 + "COMPARISON TABLE (ForestDiffusion Style)")
        print("=" * 90)
        print(f"{'Metric':<20} | {'Decision Score':<25} | {'Deviation Score':<25} | {'Reconstruction Score':<25}")
        print("-" * 90)
        print(f"{'AUROC':<20} | {np.mean(decision_auroc_list):.3f} ± {np.std(decision_auroc_list):.3f}              | "
              f"{np.mean(deviation_auroc_list):.3f} ± {np.std(deviation_auroc_list):.3f}              | "
              f"{np.mean(reconstruction_auroc_list):.3f} ± {np.std(reconstruction_auroc_list):.3f}")
        print(f"{'AUPRC':<20} | {np.mean(decision_auprc_list):.3f} ± {np.std(decision_auprc_list):.3f}              | "
              f"{np.mean(deviation_auprc_list):.3f} ± {np.std(deviation_auprc_list):.3f}              | "
              f"{np.mean(reconstruction_auprc_list):.3f} ± {np.std(reconstruction_auprc_list):.3f}")
        print("=" * 90)
        
        print("\n✓ TCCM evaluation completed successfully!")