"""
Lädt gespeicherte Extreme-Cases Ergebnisse und erstellt Plots.
"""

import joblib
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(filepath):
    """Lädt gespeicherte Ergebnisse"""
    data = joblib.load(filepath)
    print(f"Geladen: {filepath}")
    print(f"  Datasets: {data['dataset_names']}")
    print(f"  Models: {list(data['models_to_run'].keys())}")
    return data


def plot_percentile_curves(all_extreme_cases, dataset_names, output_dir="./results_"):
    """
    Plottet für jedes Modell, jeden Score und jeden Extremfall (no_contam/full_contam)
    eine Kurve mit Precision, Recall und F1 über die Percentile.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for model_name, scores_data in all_extreme_cases.items():
        for score, datasets in scores_data.items():
            for dataset_name in dataset_names:
                if dataset_name not in datasets:
                    continue
                    
                cases = datasets[dataset_name]
                
                # Ein Plot mit 2 Subplots: no_contamination und full_contamination
                fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                
                for idx, case_key in enumerate(["no_contamination", "full_contamination"]):
                    if case_key not in cases:
                        continue
                    
                    ax = axs[idx]
                    metrics = cases[case_key]["threshold_metrics"]
                    
                    # Daten extrahieren
                    percs = sorted(metrics.keys())
                    precision_means = [metrics[p]["precision_mean"] for p in percs]
                    precision_stds = [metrics[p]["precision_std"] for p in percs]
                    recall_means = [metrics[p]["recall_mean"] for p in percs]
                    recall_stds = [metrics[p]["recall_std"] for p in percs]
                    f1_means = [metrics[p]["f1_mean"] for p in percs]
                    f1_stds = [metrics[p]["f1_std"] for p in percs]
                    
                    # Precision
                    ax.plot(percs, precision_means, 'b-o', label='Precision', linewidth=2)
                    ax.fill_between(percs, 
                                   np.array(precision_means) - np.array(precision_stds),
                                   np.array(precision_means) + np.array(precision_stds),
                                   color='blue', alpha=0.2)
                    
                    # Recall
                    ax.plot(percs, recall_means, 'g-s', label='Recall', linewidth=2)
                    ax.fill_between(percs,
                                   np.array(recall_means) - np.array(recall_stds),
                                   np.array(recall_means) + np.array(recall_stds),
                                   color='green', alpha=0.2)
                    
                    # F1
                    ax.plot(percs, f1_means, 'r-^', label='F1', linewidth=2)
                    ax.fill_between(percs,
                                   np.array(f1_means) - np.array(f1_stds),
                                   np.array(f1_means) + np.array(f1_stds),
                                   color='red', alpha=0.2)
                    
                    # Bestes F1 markieren
                    best_idx = np.argmax(f1_means)
                    ax.axvline(x=percs[best_idx], color='red', linestyle='--', alpha=0.5)
                    ax.scatter([percs[best_idx]], [f1_means[best_idx]], 
                              color='red', s=150, zorder=5, marker='*')
                    
                    # AUC/AUPRC als Text
                    auc = cases[case_key]["auc_mean"]
                    auprc = cases[case_key]["auprc_mean"]
                    ax.text(0.02, 0.98, f'AUC: {auc:.4f}\nAUPRC: {auprc:.4f}',
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    title_suffix = "No Contamination" if case_key == "no_contamination" else "Full Contamination"
                    ax.set_title(f"{title_suffix}")
                    ax.set_xlabel("Threshold Percentile")
                    ax.set_ylabel("Score")
                    ax.set_ylim(0, 1.05)
                    ax.set_xlim(min(percs) - 2, max(percs) + 2)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='lower left')
                
                fig.suptitle(f"{model_name} - {score} - {dataset_name}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                filename = f"{output_dir}/percentile_curve_{model_name}_{score}_{dataset_name.replace('.npz', '')}.pdf"
                plt.savefig(filename)
                plt.close()
                print(f"Saved: {filename}")


def plot_all_scores_percentile_comparison(all_extreme_cases, dataset_names, model_name, 
                                          case_key="no_contamination", output_dir="./results_"):
    """
    Vergleicht alle 3 Scores (deviation, reconstruction, decision) für ein Modell
    in einem Plot - nur F1 für bessere Übersichtlichkeit.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    colors = {"deviation": "blue", "reconstruction": "green", "decision": "red"}
    
    for dataset_name in dataset_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for score in ["deviation", "reconstruction", "decision"]:
            if score not in all_extreme_cases.get(model_name, {}):
                continue
            if dataset_name not in all_extreme_cases[model_name][score]:
                continue
            if case_key not in all_extreme_cases[model_name][score][dataset_name]:
                continue
                
            metrics = all_extreme_cases[model_name][score][dataset_name][case_key]["threshold_metrics"]
            
            percs = sorted(metrics.keys())
            f1_means = [metrics[p]["f1_mean"] for p in percs]
            f1_stds = [metrics[p]["f1_std"] for p in percs]
            
            ax.plot(percs, f1_means, '-o', label=f'{score}', 
                   color=colors[score], linewidth=2)
            ax.fill_between(percs,
                           np.array(f1_means) - np.array(f1_stds),
                           np.array(f1_means) + np.array(f1_stds),
                           color=colors[score], alpha=0.2)
        
        case_title = "No Contamination" if case_key == "no_contamination" else "Full Contamination"
        ax.set_title(f"{model_name} - {dataset_name} - {case_title}\nF1 Score vs Threshold Percentile")
        ax.set_xlabel("Threshold Percentile")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        filename = f"{output_dir}/f1_comparison_{model_name}_{dataset_name.replace('.npz', '')}_{case_key}.pdf"
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")


def print_summary(all_extreme_cases):
    """Gibt eine Zusammenfassung der Ergebnisse aus"""
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
                    
                    # Bestes Percentil nach F1
                    best_p = max(metrics['threshold_metrics'].items(), 
                                key=lambda x: x[1]['f1_mean'])
                    print(f"    Best F1: {best_p[1]['f1_mean']:.4f} @ {best_p[0]}th percentile")


if __name__ == "__main__":
    # ============================================================================
    # HIER DEINEN DATEIPFAD EINTRAGEN
    # ============================================================================
    folder = "results_"
    results_path = Path(f"./{folder}/results_data/extreme_cases_29_Pima_20251212_164701.joblib")
    # Oder: den neuesten File automatisch finden
    # results_dir = Path("./results_data")
    # results_path = sorted(results_dir.glob("extreme_cases_*.joblib"))[-1]
    
    # Laden
    data = load_results(results_path)
    
    all_extreme_cases = data["all_extreme_cases"]
    dataset_names = data["dataset_names"]
    models_to_run = data["models_to_run"]
    
    # Output-Verzeichnis
    output_dir = "./results_plots"
    
    # ============================================================================
    # PLOTS ERSTELLEN
    # ============================================================================
    
    # 1. Percentile-Kurven (Precision, Recall, F1) für jeden Score/Modell
    print("\nErstelle Percentile-Kurven...")
    plot_percentile_curves(all_extreme_cases, dataset_names, output_dir=output_dir)
    
    # 2. F1-Vergleich aller Scores pro Modell
    print("\nErstelle F1-Vergleichsplots...")
    for model_name in models_to_run.keys():
        plot_all_scores_percentile_comparison(
            all_extreme_cases, dataset_names, model_name, 
            case_key="no_contamination", output_dir=output_dir
        )
        plot_all_scores_percentile_comparison(
            all_extreme_cases, dataset_names, model_name, 
            case_key="full_contamination", output_dir=output_dir
        )
    
    # 3. Summary ausgeben
    print_summary(all_extreme_cases)
    
    print(f"\n{'='*60}")
    print(f"Alle Plots gespeichert in: {output_dir}/")
    print(f"{'='*60}")