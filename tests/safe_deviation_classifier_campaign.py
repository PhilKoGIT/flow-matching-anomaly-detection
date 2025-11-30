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

pd.set_option("display.max_rows", None)      # Alle Zeilen anzeigen
pd.set_option("display.max_columns", None)   # Alle Spalten anzeigen
pd.set_option("display.max_colwidth", None)  # Full content in each cell


#copied partly from by utils.py of https://github.com/ZhongLIFR/TCCM-NIPS/blob/main/utils.py

base_dir = Path(__file__).resolve().parent
file_path = base_dir.parent / "data" / "5_campaign.npz"
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

#n_t = 50
#duplicate_K = 10
n_t = 5
duplicate_K = 4

model = ForestDiffusionModel(
    X=X_train,
    label_y=None,     # unsupervised; wir geben Labels nur für Evaluation
    # Diffusion settings
    n_t=n_t,
    duplicate_K=duplicate_K,
    diffusion_type='flow',  # wichtig für compute_deviation_score
    eps=1e-3,

    # Model settings
    model='xgboost',
    max_depth=7,

   n_estimators=100,
   # n_estimators=2,
    eta=0.3,
    gpu_hist=False,   # auf True setzen, wenn GPU verfügbar

    # Data settings
    n_batch=25,        # Important: 0 for compute_deviation_score
    seed=666,
    n_jobs=-1,

    # Feature types: alles numerisch behandelt
    bin_indexes=[],
    cat_indexes=[],
    int_indexes=[],

    # Other settings
    remove_miss=False,
    p_in_one=True,    # WICHTIG für compute_deviation_score

)

print("✓ Model trained successfully on full training data!")
joblib.dump(model, "campaign_model.joblib")


#model = joblib.load("campaign_model.joblib")
# ============================================================================
# ANOMALY SCORES BERECHNEN
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING ANOMALY SCORES AUF TESTSET")
print("=" * 80)

anomaly_scores_deviation = model.compute_deviation_score(
    test_samples=X_test,
    n_t=n_t   # same amount of noise as training
)

anomaly_scores_reconstruction = model.compute_reconstruction_score(
    test_samples=X_test,
    n_t=n_t
)

# Kennzahlen berechnen
auroc = roc_auc_score(y_test, anomaly_scores_deviation)
auprc = average_precision_score(y_test, anomaly_scores_deviation)
print(f"\nAUROC deviation: {auroc:.4f}")
print(f"AUPRC deviation (Average Precision): {auprc:.4f}")

auroc = roc_auc_score(y_test, anomaly_scores_reconstruction)
auprc = average_precision_score(y_test, anomaly_scores_reconstruction)
print(f"\nAUROC reconstruction: {auroc:.4f}")
print(f"AUPRC reconstruction (Average Precision): {auprc:.4f}")



print(f"✓ Anomaly scores computed: {len(anomaly_scores_deviation)}")
print(f"  Score range: [{anomaly_scores_deviation.min():.4f}, {anomaly_scores_deviation.max():.4f}]")
print(f"  Score mean : {anomaly_scores_deviation.mean():.4f}")
print(f"  Score std  : {anomaly_scores_deviation.std():.4f}")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION METRICS")
print("=" * 80)

# ROC-AUC
try:
    auc_score = roc_auc_score(y_test, anomaly_scores_deviation)
    print(f"\n✓ ROC-AUC Score: {auc_score:.4f}")
except Exception as e:
    print(f"✗ Could not compute ROC-AUC: {e}")
    auc_score = None

# Threshold-Suche über Percentiles
thresholds_percentiles = [70, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9]
print("\nPerformance at different threshold percentiles:")
print("-" * 80)

best_f1 = 0
best_threshold = anomaly_scores_deviation.max()
best_percentile = 100

for percentile in thresholds_percentiles:
    threshold = np.percentile(anomaly_scores_deviation, percentile)
    preds = (anomaly_scores_deviation > threshold).astype(int)

    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    tn = np.sum((preds == 0) & (y_test == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_test)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_percentile = percentile

    print(f"\n{percentile}th Percentile (threshold={threshold:.4f}):")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

print(f"\n✓ Best F1-Score: {best_f1:.4f} at {best_percentile}th percentile (threshold={best_threshold:.4f})")

# Finale Vorhersagen beim besten Threshold
final_preds = (anomaly_scores_deviation > best_threshold).astype(int)
cm = confusion_matrix(y_test, final_preds)

print("\nFinal confusion matrix at best F1 threshold:")
print(cm)
print(f"  Normal correctly classified: {np.sum((y_test == 0) & (final_preds == 0))}/{np.sum(y_test == 0)}")
print(f"  Anomalies detected          : {np.sum((y_test == 1) & (final_preds == 1))}/{np.sum(y_test == 1)}")
