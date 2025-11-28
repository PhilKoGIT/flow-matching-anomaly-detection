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
print(f"Dataset loaded successfully!")
print(f"Training data: {X_train.shape}, Normal: {len(y_train)}")
print(f"Test data: {X_test.shape}, Normal: {sum(y_test == 0)}, Anomalies: {sum(y_test == 1)}")

n_t = 5
duplicate_K = 3

# model = ForestDiffusionModel(
#     X=X_train,
#     label_y=None,     # unsupervised; wir geben Labels nur für Evaluation
#     # Diffusion settings
#     n_t=n_t,
#     duplicate_K=duplicate_K,
#     diffusion_type='flow',  # wichtig für compute_deviation_score
#     eps=1e-3,

#     # Model settings
#     model='xgboost',
#     max_depth=7,

#     n_estimators=100,
#     eta=0.3,
#     gpu_hist=True,   # auf True setzen, wenn GPU verfügbar

#     # Data settings
#     n_batch=0,        # Important: 0 for compute_deviation_score
#     seed=666,
#     n_jobs=-1,

#     # Feature types: alles numerisch behandelt
#     bin_indexes=[],
#     cat_indexes=[],
#     int_indexes=[],

#     # Other settings
#     remove_miss=False,
#     p_in_one=True,    # WICHTIG für compute_deviation_score

# )

# print("✓ Model trained successfully on full training data!")
# joblib.dump(model, "campaign_model.joblib")
# model = joblib.load("campaign_model.joblib")

model = joblib.load("business_model.joblib")
# ============================================================================
# ANOMALY SCORES BERECHNEN
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING ANOMALY SCORES AUF TESTSET")
print("=" * 80)

anomaly_scores = model.compute_reconstruction_score(
    test_samples=X_test,
    n_t=3   # same amount of noise as training
)

# Kennzahlen berechnen
auroc = roc_auc_score(y_test, anomaly_scores)
auprc = average_precision_score(y_test, anomaly_scores)
print(f"\nAUROC: {auroc:.4f}")
print(f"AUPRC (Average Precision): {auprc:.4f}")

print(f"✓ Anomaly scores computed: {len(anomaly_scores)}")
print(f"  Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
print(f"  Score mean : {anomaly_scores.mean():.4f}")
print(f"  Score std  : {anomaly_scores.std():.4f}")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION METRICS")
print("=" * 80)

# ROC-AUC
try:
    auc_score = roc_auc_score(y_test, anomaly_scores)
    print(f"\n✓ ROC-AUC Score: {auc_score:.4f}")
except Exception as e:
    print(f"✗ Could not compute ROC-AUC: {e}")
    auc_score = None

# Threshold-Suche über Percentiles
thresholds_percentiles = [70, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9]
print("\nPerformance at different threshold percentiles:")
print("-" * 80)

best_f1 = 0
best_threshold = anomaly_scores.max()
best_percentile = 100

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

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_percentile = percentile

    print(f"\n{percentile}th Percentile (threshold={threshold:.4f}):")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

print(f"\n✓ Best F1-Score: {best_f1:.4f} at {best_percentile}th percentile (threshold={best_threshold:.4f})")

# Finale Vorhersagen beim besten Threshold
final_preds = (anomaly_scores > best_threshold).astype(int)
cm = confusion_matrix(y_test, final_preds)

print("\nFinal confusion matrix at best F1 threshold:")
print(cm)
print(f"  Normal correctly classified: {np.sum((y_test == 0) & (final_preds == 0))}/{np.sum(y_test == 0)}")
print(f"  Anomalies detected          : {np.sum((y_test == 1) & (final_preds == 1))}/{np.sum(y_test == 1)}")

# ============================================================================
# VISUALISIERUNGEN
# ============================================================================

# print("\n" + "=" * 80)
# print("CREATING VISUALIZATIONS")
# print("=" * 80)

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # 1) Score-Histogramm nach Label
# ax = axes[0, 0]
# ax.hist(anomaly_scores[y_test == 0], bins=40, alpha=0.6, label="Normal", color='blue')
# ax.hist(anomaly_scores[y_test == 1], bins=40, alpha=0.6, label="Anomaly", color='red')
# ax.axvline(best_threshold, color='black', linestyle='--', linewidth=2,
#            label=f'Threshold ({best_percentile}th %ile)')
# ax.set_title("Anomaly Score Distribution (Test Set)")
# ax.set_xlabel("Anomaly Score")
# ax.set_ylabel("Frequency")
# ax.legend()
# ax.grid(alpha=0.3)

# # 2) ROC Curve
# ax = axes[0, 1]
# if auc_score is not None:
#     fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
#     ax.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.4f})", linewidth=2)
#     ax.plot([0, 1], [0, 1], 'k--', label="Random")
#     ax.set_title("ROC Curve")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.legend()
#     ax.grid(alpha=0.3)

# # 3) Precision-Recall Curve
# ax = axes[1, 0]
# precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
# ax.plot(recall, precision, linewidth=2, color='darkgreen')
# ax.set_title("Precision-Recall Curve")
# ax.set_xlabel("Recall")
# ax.set_ylabel("Precision")
# ax.grid(alpha=0.3)

# # 4) Confusion Matrix Heatmap
# ax = axes[1, 1]
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
# ax.set_title(f"Confusion Matrix\nThreshold at {best_percentile}th percentile")
# ax.set_xlabel("Predicted")
# ax.set_ylabel("True")
# ax.set_xticklabels(['Normal', 'Anomaly'])
# ax.set_yticklabels(['Normal', 'Anomaly'])

# plt.tight_layout()
# plot_path = results_dir / "business_anomaly_detection_results.png"
# plt.savefig(plot_path, dpi=300, bbox_inches="tight")
# print(f"✓ Saved plot to: {plot_path}")

# # ============================================================================
# # SAVE RESULTS (MIT MAPPING ZU ORIGINALDATEN)
# # ============================================================================

# print("\n" + "=" * 80)
# print("SAVING RESULTS")
# print("=" * 80)

# # Da das Skript in tests/ liegt und prepare_data in tests/data/ speichert:
# original_data_path = CURRENT_FILE.parent / "data" / "original_data.csv"
# df_original = pd.read_csv(original_data_path, index_col=0)

# # Mapping von transformed_index (0..len(X_test)-1) zu original_index
# mapping_test = test_mapping.set_index('transformed_index').loc[range(len(X_test))]
# original_indices = mapping_test['original_index'].values

# # Subset der Originaldaten für Testset
# df_test_original = df_original.loc[original_indices].reset_index(drop=False)
# df_test_original.rename(columns={'index': 'original_index'}, inplace=True)

# # Ergebnis-DataFrame
# results_df = pd.DataFrame({
#     'test_index': range(len(X_test)),
#     'original_index': original_indices,
#     'is_anomaly': y_test,
#     'anomaly_score': anomaly_scores,
#     'predicted_anomaly': final_preds,
#     'correct_prediction': (y_test == final_preds).astype(int),
# })

# # anomaly_description aus Originaldaten anhängen (falls vorhanden)
# if 'anomaly_description' in df_test_original.columns:
#     results_df['anomaly_description'] = df_test_original['anomaly_description'].values

# # Einige Kontextfelder (optional)
# for col in ['bank_account_uuid', 'date_post', 'amount', 'ref_name']:
#     if col in df_test_original.columns and col not in results_df.columns:
#         results_df[col] = df_test_original[col].values

# # Top-Anomalien ausgeben
# print("\nTop highest anomaly scores:")
# print(results_df.sort_values('anomaly_score', ascending=False).head(400)[
#     ['test_index', 'original_index', 'is_anomaly', 'anomaly_score', 'predicted_anomaly', 'anomaly_description']
# ])

# # Speichern
# results_path = results_dir / "business_anomaly_scores.csv"
# results_df.to_csv(results_path, index=False)
# print(f"\n✓ Results saved to: {results_path}")

# # Summary JSON
# summary = {
#     'model_config': {
#         'n_t': 30,
#         'duplicate_K': 10,
#         'diffusion_type': 'flow',
#         'max_depth': 8,
#         'n_estimators': 150,
#         'eta': 0.3,
#     },
#     'data': {
#         'n_train': int(X_train.shape[0]),
#         'n_test': int(X_test.shape[0]),
#         'n_features': int(X_train.shape[1]),
#         'n_anomalies_train': int(y_train.sum()),
#         'n_anomalies_test': int(y_test.sum()),
#         'anomaly_ratio_test': float(y_test.mean()),
#     },
#     'performance': {
#         'roc_auc': float(auc_score) if auc_score is not None else None,
#         'best_f1_score': float(best_f1),
#         'best_threshold': float(best_threshold),
#         'best_percentile': int(best_percentile),
#     },
#     'scores': {
#         'min': float(anomaly_scores.min()),
#         'max': float(anomaly_scores.max()),
#         'mean': float(anomaly_scores.mean()),
#         'std': float(anomaly_scores.std()),
#     },
# }

# summary_path = results_dir / "business_anomaly_summary.json"
# with open(summary_path, 'w') as f:
#     json.dump(summary, f, indent=2)
# print(f"✓ Summary saved to: {summary_path}")

# print("\n" + "=" * 80)
# print("BUSINESS ANOMALY DETECTION EXPERIMENT COMPLETED!")
# print("=" * 80)
# if auc_score is not None:
#     print(f"ROC-AUC: {auc_score:.4f}")
# print(f"Best F1: {best_f1:.4f}")
# print(f"Normal correctly classified: {np.sum((y_test == 0) & (final_preds == 0))}/{np.sum(y_test == 0)}")
# print(f"Anomalies detected         : {np.sum((y_test == 1) & (final_preds == 1))}/{np.sum(y_test == 1)}")

# # ============================================================================
# # AUROC / AUPRC + ROC- & PR-Kurven
# # ============================================================================

# # Kennzahlen berechnen
# auroc = roc_auc_score(y_test, anomaly_scores)
# auprc = average_precision_score(y_test, anomaly_scores)

# print(f"\nAUROC: {auroc:.4f}")
# print(f"AUPRC (Average Precision): {auprc:.4f}")

# # Kurven berechnen
# fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
# precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)

# # Plots
# plt.figure(figsize=(12, 5))

# # ROC-Kurve
# plt.subplot(1, 2, 1)
# plt.plot(fpr, tpr, label=f"ROC (AUC = {auroc:.3f})")
# plt.plot([0, 1], [0, 1], "--", label="Random")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid(alpha=0.3)

# # Precision-Recall-Kurve
# plt.subplot(1, 2, 2)
# plt.plot(recall, precision, label=f"PR (AUPRC = {auprc:.3f})")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig("roc_pr_curves.png", dpi=300, bbox_inches="tight")
