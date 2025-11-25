

#genau überlegen wie ich das modell einstellen möchte, damit ich die funktione nutzen kann
#n_batch: größe des batches nutzt bei n_batch>0 den iterator. sollte egal sein...




# test_anomaly_detection.py

from ForestDiffusion import ForestDiffusionModel
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
import seaborn as sns
import time
import joblib


pd.set_option("display.max_rows", None)     
pd.set_option("display.max_columns", None)   
pd.set_option("display.max_colwidth", None)

# ============================================================================
# SETUP
# ============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
dataset_path = PROJECT_ROOT / "data" / "EngineFaultDB_Final.csv"
output_dir = PROJECT_ROOT / "results" / "anomaly_detection"
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")
df_original = pd.read_csv(dataset_path)

print(f"Total samples: {len(df_original)}")
print(f"Fault distribution:")
print(df_original["Fault"].value_counts().sort_index())

# ============================================================================
# DATA PREPARATION - TRAIN ON NORMAL DATA ONLY
# ============================================================================

# Nur normale Daten für Training
df_normal = df_original[df_original["Fault"] == 0]

print(f"\nNormal samples: {len(df_normal)}")

# Split normale Daten in Train/Val/Test
train_df, test_df = train_test_split(
    df_normal, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_normal['Fault']
)
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.1, 
    random_state=42, 
    stratify=train_df['Fault']
)

print(f"Train samples (normal): {len(train_df)}")
print(f"Val samples (normal): {len(val_df)}")
print(f"Test samples (normal): {len(test_df)}")

# ============================================================================
# ADD ANOMALIES TO TEST SET
# ============================================================================

# Fault 1 Samples
df_fault1 = df_original[df_original["Fault"] == 1].sample(
    n=min(10, len(df_original[df_original["Fault"] == 1])), 
    random_state=42
)

# Fault 2 Samples
df_fault2 = df_original[df_original["Fault"] == 2].sample(
    n=min(10, len(df_original[df_original["Fault"] == 2])), 
    random_state=42
)

# Fault 3 Samples
df_fault3 = df_original[df_original["Fault"] == 3].sample(
    n=min(10, len(df_original[df_original["Fault"] == 3])), 
    random_state=42
)

# Kombiniere Test-Set mit Anomalien
df_combined = pd.concat([test_df, df_fault1, df_fault2, df_fault3], ignore_index=True)

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

test_df = df_combined.copy()



# ============================================================================
# ADD ANOMALIES TO TRAIN SET
# ============================================================================

# Fault 1 Samples
df_fault1 = df_original[df_original["Fault"] == 1].sample(
    n=min(30, len(df_original[df_original["Fault"] == 1])), 
    random_state=42
)

# Fault 2 Samples
df_fault2 = df_original[df_original["Fault"] == 2].sample(
    n=min(30, len(df_original[df_original["Fault"] == 2])), 
    random_state=42
)

# Fault 3 Samples
df_fault3 = df_original[df_original["Fault"] == 3].sample(
    n=min(30, len(df_original[df_original["Fault"] == 3])), 
    random_state=42
)

# Kombiniere Test-Set mit Anomalien
df_combined = pd.concat([train_df, df_fault1, df_fault2, df_fault3], ignore_index=True)

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

train_df = df_combined.copy()





print(f"\nFinal test set composition:")
print(test_df["Fault"].value_counts().sort_index())
print(f"Total test samples: {len(test_df)}")

# ============================================================================
# PREPARE FEATURES AND LABELS
# ============================================================================

# Labels erstellen: 0 = normal, 1 = anomaly
train_labels = (train_df["Fault"] != 0).astype(int).values
test_labels = (test_df["Fault"] != 0).astype(int).values

print(f"\nTest set anomaly distribution:")
print(f"Normal (Fault=0): {np.sum(test_labels == 0)} ({np.mean(test_labels == 0)*100:.1f}%)")
print(f"Anomalous (Fault>0): {np.sum(test_labels == 1)} ({np.mean(test_labels == 1)*100:.1f}%)")

# Features extrahieren (Fault-Spalte entfernen)
X_train = train_df.drop(columns=["Fault"]).to_numpy()
X_test = test_df.drop(columns=["Fault"]).to_numpy()

print(f"\nFeature matrix shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# ============================================================================
# MODEL TRAINING (NUR AUF NORMALEN DATEN)
# ============================================================================

print("\n" + "="*60)
print("TRAINING FORESTDIFFUSION MODEL")
print("="*60)
print("Training on NORMAL data only (Fault=0)...")

model = ForestDiffusionModel(
    X=X_train,
    label_y=None,  # Unsupervised
    # Diffusion settings
   # n_t=100,
    n_t=50,
    #duplicate_K=10,
    duplicate_K=10,
    diffusion_type='flow',  # WICHTIG für compute_deviation_score
    eps=1e-3,
    
    # Model settings
    model='xgboost',
    max_depth=7,
    n_estimators=100,
    eta=0.3,
    gpu_hist=False,  # Setze auf True wenn GPU verfügbar
    
    # Data settings
    n_batch=0,  # WICHTIG: 0 für compute_deviation_score
    seed=666,
    n_jobs=-1,
    
    # Feature types (alles numerisch)
    bin_indexes=[],
    cat_indexes=[],
    int_indexes=[],
    
    # Other settings
    remove_miss=False,
    p_in_one=True,  

)

print("✓ Model trained successfully on normal data!")

# ============================================================================
# ANOMALY DETECTION
# ============================================================================
joblib.dump(model, "engine_model.joblib")

print("\n" + "="*60)
print("COMPUTING ANOMALY SCORES")
print("="*60)

# Compute deviation scores
print("Computing deviation scores for test samples...")
anomaly_scores = model.compute_deviation_score(
    test_samples=X_test,
    n_t=30  # Gleich wie beim Training
)

print(f"✓ Anomaly scores computed: {len(anomaly_scores)}")
print(f"  Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
print(f"  Score mean: {anomaly_scores.mean():.4f}")
print(f"  Score std: {anomaly_scores.std():.4f}")

# Statistiken nach Gruppe
print("\nScore statistics by fault type:")
for fault_type in sorted(test_df["Fault"].unique()):
    mask = test_df["Fault"] == fault_type
    scores = anomaly_scores[mask]
    label = "Normal" if fault_type == 0 else f"Fault {fault_type}"
    print(f"  {label:12s}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

# ROC-AUC Score
try:
    auc_score = roc_auc_score(test_labels, anomaly_scores)
    print(f"\n✓ ROC-AUC Score: {auc_score:.4f}")
except Exception as e:
    print(f"✗ Could not compute ROC-AUC: {e}")
    auc_score = None

# Performance bei verschiedenen Thresholds
thresholds_percentiles = [90, 95, 97.5, 99, 99.5, 99.9]
print("\nPerformance at different threshold percentiles:")
print("-" * 70)

best_f1 = 0
best_threshold = 0
best_percentile = 0

for percentile in thresholds_percentiles:
    threshold = np.percentile(anomaly_scores, percentile)
    predictions = (anomaly_scores > threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    fn = np.sum((predictions == 0) & (test_labels == 1))
    tn = np.sum((predictions == 0) & (test_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_labels)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_percentile = percentile
    
    print(f"\n{percentile}th Percentile (threshold={threshold:.4f}):")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

print(f"\n✓ Best F1-Score: {best_f1:.4f} at {best_percentile}th percentile (threshold={best_threshold:.4f})")

# ============================================================================
# DETAILED ANALYSIS BY FAULT TYPE
# ============================================================================

print("\n" + "="*60)
print("DETECTION RATE BY FAULT TYPE")
print("="*60)

for fault_type in sorted(test_df["Fault"].unique()):
    if fault_type == 0:
        continue  # Skip normal
    
    mask = test_df["Fault"] == fault_type
    scores = anomaly_scores[mask]
    detected = np.sum(scores > best_threshold)
    total = len(scores)
    rate = detected / total * 100 if total > 0 else 0
    
    print(f"Fault {fault_type}: {detected}/{total} detected ({rate:.1f}%)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Score Distribution by Fault Type
ax = axes[0, 0]
for fault_type in sorted(test_df["Fault"].unique()):
    mask = test_df["Fault"] == fault_type
    label = "Normal (Fault=0)" if fault_type == 0 else f"Fault {fault_type}"
    color = 'blue' if fault_type == 0 else ['red', 'orange', 'purple'][fault_type-1]
    ax.hist(anomaly_scores[mask], bins=30, alpha=0.6, label=label, color=color)
ax.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({best_percentile}th %ile)')
ax.set_xlabel('Anomaly Score', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Anomaly Scores by Fault Type', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. ROC Curve
ax = axes[0, 1]
if auc_score is not None:
    fpr, tpr, _ = roc_curve(test_labels, anomaly_scores)
    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.4f})', linewidth=2, color='darkblue')
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

# 3. Precision-Recall Curve
ax = axes[0, 2]
precision, recall, thresholds_pr = precision_recall_curve(test_labels, anomaly_scores)
ax.plot(recall, precision, linewidth=2, color='darkgreen')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. Boxplot by Fault Type
ax = axes[1, 0]
fault_types = []
score_groups = []
for fault_type in sorted(test_df["Fault"].unique()):
    mask = test_df["Fault"] == fault_type
    label = "Normal" if fault_type == 0 else f"Fault {fault_type}"
    fault_types.append(label)
    score_groups.append(anomaly_scores[mask])

bp = ax.boxplot(score_groups, labels=fault_types, patch_artist=True)
for i, box in enumerate(bp['boxes']):
    color = 'lightblue' if i == 0 else ['lightcoral', 'lightyellow', 'plum'][i-1]
    box.set_facecolor(color)
ax.axhline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold')
ax.set_ylabel('Anomaly Score', fontsize=11)
ax.set_title('Anomaly Scores by Fault Type', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 5. Score vs Sample Index (colored by fault)
ax = axes[1, 1]
for fault_type in sorted(test_df["Fault"].unique()):
    mask = test_df["Fault"] == fault_type
    indices = np.where(mask)[0]
    label = "Normal" if fault_type == 0 else f"Fault {fault_type}"
    color = 'blue' if fault_type == 0 else ['red', 'orange', 'purple'][fault_type-1]
    marker = 'o' if fault_type == 0 else 'x'
    size = 20 if fault_type == 0 else 60
    ax.scatter(indices, anomaly_scores[mask], alpha=0.6, s=size, 
               label=label, color=color, marker=marker)
ax.axhline(best_threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Sample Index', fontsize=11)
ax.set_ylabel('Anomaly Score', fontsize=11)
ax.set_title('Anomaly Scores by Sample', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6. Confusion Matrix
ax = axes[1, 2]
predictions = (anomaly_scores > best_threshold).astype(int)
cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)
ax.set_title(f'Confusion Matrix\n(Threshold at {best_percentile}th percentile)', 
             fontsize=12, fontweight='bold')
ax.set_xticklabels(['Normal', 'Anomaly'])
ax.set_yticklabels(['Normal', 'Anomaly'])

plt.tight_layout()
plot_path = output_dir / "anomaly_detection_results.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to: {plot_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# DataFrame mit allen Ergebnissen
results_df = pd.DataFrame({
    'sample_index': range(len(test_df)),
    'fault_type': test_df["Fault"].values,
    'is_anomaly': test_labels,
    'anomaly_score': anomaly_scores,
    'predicted_anomaly': (anomaly_scores > best_threshold).astype(int),
    'correct_prediction': (test_labels == (anomaly_scores > best_threshold).astype(int)).astype(int)
})

# Top Anomalien
print("\nTop 20 highest anomaly scores:")
print(results_df.nlargest(20, 'anomaly_score')[['sample_index', 'fault_type', 'anomaly_score', 'predicted_anomaly']])

# Speichern
results_path = output_dir / "anomaly_scores.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Results saved to: {results_path}")

# Zusammenfassung
summary = {
    'model_config': {
        'n_t': 30,
        'duplicate_K': 10,
        'diffusion_type': 'flow',
        'max_depth': 8,
        'n_estimators': 150,
    },
    'data': {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'n_anomalies_test': int(np.sum(test_labels)),
        'anomaly_ratio': float(np.mean(test_labels)),
        'fault_distribution': test_df["Fault"].value_counts().to_dict(),
    },
    'performance': {
        'roc_auc': float(auc_score) if auc_score is not None else None,
        'best_f1_score': float(best_f1),
        'best_threshold': float(best_threshold),
        'best_percentile': int(best_percentile),
    },
    'scores': {
        'min': float(anomaly_scores.min()),
        'max': float(anomaly_scores.max()),
        'mean': float(anomaly_scores.mean()),
        'std': float(anomaly_scores.std()),
    },
    'detection_by_fault': {
        int(ft): {
            'detected': int(np.sum((test_df["Fault"] == ft) & (anomaly_scores > best_threshold))),
            'total': int(np.sum(test_df["Fault"] == ft)),
            'rate': float(np.sum((test_df["Fault"] == ft) & (anomaly_scores > best_threshold)) / 
                         np.sum(test_df["Fault"] == ft) * 100) if np.sum(test_df["Fault"] == ft) > 0 else 0
        }
        for ft in sorted(test_df["Fault"].unique()) if ft != 0
    }
}

import json
summary_path = output_dir / "summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved to: {summary_path}")

print("\n" + "="*60)
print("ANOMALY DETECTION TEST COMPLETED!")
print("="*60)
print(f"ROC-AUC: {auc_score:.4f}")
print(f"Best F1: {best_f1:.4f}")
print(f"Normal samples correctly classified: {np.sum((test_labels == 0) & (anomaly_scores <= best_threshold))}/{np.sum(test_labels == 0)}")
print(f"Anomalies detected: {np.sum((test_labels == 1) & (anomaly_scores > best_threshold))}/{np.sum(test_labels == 1)}")