# # plot_results.py

# import joblib
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from pathlib import Path

# output_dir = Path("./plots_combined")
# output_dir.mkdir(parents=True, exist_ok=True)

# COLORS = {
#     "ForestDiffusion_nt50_dk10": "blue",
#     "ForestFlow_nt20_dk10": "green",
#     "TCCM_nt50": "red",
# }


# def load_and_merge_results(result_paths):
#     """Lädt mehrere Result-Dateien und merged sie."""
#     merged = {"all_extreme_cases": {}}
    
#     for path in result_paths:
#         path = Path(path)
#         if not path.exists():
#             print(f"[!] Nicht gefunden: {path}")
#             continue
        
#         print(f"[✓] Lade: {path.name}")
#         data = joblib.load(path)
        
#         for model, scores in data.get("all_extreme_cases", {}).items():
#             if model not in merged["all_extreme_cases"]:
#                 merged["all_extreme_cases"][model] = {}
#             for score, datasets in scores.items():
#                 if score not in merged["all_extreme_cases"][model]:
#                     merged["all_extreme_cases"][model][score] = {}
#                 merged["all_extreme_cases"][model][score].update(datasets)
    
#     return merged


# def get_all_datasets(all_extreme_cases):
#     """Findet alle Datasets in den Daten."""
#     datasets = set()
#     for model in all_extreme_cases.values():
#         for score in model.values():
#             datasets.update(score.keys())
#     return sorted(datasets)


# def plot_percentile_comparison(all_extreme_cases, dataset_name, score_type, metric,
#                                 case_key, output_dir):
#     """
#     X-Achse: Threshold Percentile
#     Y-Achse: F1, Precision, oder Recall
#     Legende: Modelle
#     """
#     fig, ax = plt.subplots(figsize=(10, 6))
#     has_data = False
    
#     for model_name in all_extreme_cases.keys():
#         if score_type not in all_extreme_cases[model_name]:
#             continue
#         if dataset_name not in all_extreme_cases[model_name][score_type]:
#             continue
#         if case_key not in all_extreme_cases[model_name][score_type][dataset_name]:
#             continue
        
#         metrics_data = all_extreme_cases[model_name][score_type][dataset_name][case_key]["threshold_metrics"]
        
#         percs = sorted(metrics_data.keys())
#         means = [metrics_data[p][f"{metric}_mean"] for p in percs]
#         stds = [metrics_data[p][f"{metric}_std"] for p in percs]
        
#         color = COLORS.get(model_name, None)
#         ax.plot(percs, means, '-o',
#                 label=model_name, color=color, linewidth=2, markersize=6)
#         ax.fill_between(percs,
#                         np.array(means) - np.array(stds),
#                         np.array(means) + np.array(stds),
#                         alpha=0.2, color=color)
#         has_data = True
    
#     if not has_data:
#         plt.close()
#         return
    
#     case_title = "No Contamination" if case_key == "no_contamination" else "Full Contamination"
#     ax.set_xlabel("Threshold Percentile", fontsize=12)
#     ax.set_ylabel(metric.capitalize(), fontsize=12)
#     ax.set_title(f"{dataset_name} - {score_type} - {case_title}\n{metric.capitalize()}", fontsize=14)
#     ax.set_ylim(0, 1.05)
#     ax.grid(True, alpha=0.3)
#     ax.legend()
    
#     plt.tight_layout()
#     filename = output_dir / f"percentile_{metric}_{dataset_name.replace('.npz', '')}_{score_type}_{case_key}.pdf"
#     plt.savefig(filename)
#     plt.close()
#     print(f"  Saved: {filename.name}")


# def create_all_plots(merged, output_dir):
#     """Erstellt alle Percentile-Plots."""
#     all_extreme = merged["all_extreme_cases"]
#     datasets = get_all_datasets(all_extreme)
    
#     print("\n" + "="*60)
#     print("ERSTELLE PERCENTILE PLOTS")
#     print("="*60)
#     print(f"Datasets: {datasets}")
#     print(f"Modelle: {list(all_extreme.keys())}")
    
#     for dataset_name in datasets:
#         print(f"\n--- {dataset_name} ---")
#         for score_type in ["deviation", "reconstruction", "decision"]:
#             for case_key in ["no_contamination", "full_contamination"]:
#                 for metric in ["f1", "precision", "recall"]:
#                     plot_percentile_comparison(
#                         all_extreme, dataset_name, score_type, metric, case_key, output_dir
#                     )


# if __name__ == "__main__":
    
#     # DATEIEN HIER EINTRAGEN
#     result_files = [
#         Path("./results_quick/results_data/extreme_cases_29_Pima_20251213_143308.joblib"),
#         # Weitere Dateien...
#     ]
    
#     output_dir = Path("./plots_combined")
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     merged = load_and_merge_results(result_files)
#     create_all_plots(merged, output_dir)
    
#     print("\n" + "="*60)
#     print(f"FERTIG! Plots in: {output_dir}")
#     print("="*60)

# plot_results.py - Vollversion mit Modell-Gruppierung

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("./plots_combined")
output_dir.mkdir(parents=True, exist_ok=True)

COLORS_MODELS = {
    "ForestDiffusion_nt50_dk10": "blue",
    "ForestFlow_nt20_dk10": "green",
    "TCCM_nt50": "red",
}

COLORS_SCORES = {
    "deviation": "blue",
    "reconstruction": "green",
    "decision": "red",
}


def load_and_merge_results(result_paths):
    """Lädt mehrere Result-Dateien und merged sie."""
    merged = {
        "all_results_combined": {},
        "all_extreme_cases": {},
    }
    
    for path in result_paths:
        path = Path(path)
        if not path.exists():
            print(f"[!] Nicht gefunden: {path}")
            continue
        
        print(f"[✓] Lade: {path.name}")
        data = joblib.load(path)
        
        # Merge all_results_combined
        for dataset, models in data.get("all_results_combined", {}).items():
            if dataset not in merged["all_results_combined"]:
                merged["all_results_combined"][dataset] = {}
            for model, scores in models.items():
                if model not in merged["all_results_combined"][dataset]:
                    merged["all_results_combined"][dataset][model] = {}
                merged["all_results_combined"][dataset][model].update(scores)
        
        # Merge extreme_cases
        for model, scores in data.get("all_extreme_cases", {}).items():
            if model not in merged["all_extreme_cases"]:
                merged["all_extreme_cases"][model] = {}
            for score, datasets in scores.items():
                if score not in merged["all_extreme_cases"][model]:
                    merged["all_extreme_cases"][model][score] = {}
                merged["all_extreme_cases"][model][score].update(datasets)
    
    return merged


def get_all_datasets(merged):
    """Findet alle Datasets."""
    datasets = set(merged["all_results_combined"].keys())
    for model in merged["all_extreme_cases"].values():
        for score in model.values():
            datasets.update(score.keys())
    return sorted(datasets)


# =============================================================================
# PLOT 1: Contamination - Modellvergleich pro Score
# (X=Contamination, Y=AUROC/AUPRC, Legende=Modelle)
# =============================================================================

def plot_contamination_models(all_results, dataset_name, score_type, metric, output_dir):
    """Vergleicht Modelle für einen Score-Typ."""
    if dataset_name not in all_results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    
    for model_name in all_results[dataset_name].keys():
        if score_type not in all_results[dataset_name][model_name]:
            continue
        
        data = all_results[dataset_name][model_name][score_type]
        contam_levels = np.array(data["contamination_levels"])
        values = np.array(data[metric])
        
        color = COLORS_MODELS.get(model_name, None)
        ax.plot(contam_levels, values[:, 0], '-o',
                label=model_name, color=color, linewidth=2, markersize=6)
        ax.fill_between(contam_levels,
                        values[:, 0] - values[:, 1],
                        values[:, 0] + values[:, 1],
                        alpha=0.2, color=color)
        has_data = True
    
    if not has_data:
        plt.close()
        return
    
    ax.set_xlabel("Contamination Level", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"{dataset_name} - {score_type} - {metric.upper()}", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    filename = output_dir / f"contamination_models_{dataset_name.replace('.npz', '')}_{score_type}_{metric}.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename.name}")


# =============================================================================
# PLOT 2: Contamination - Score-Vergleich pro Modell (NEU!)
# (X=Contamination, Y=AUROC/AUPRC, Legende=Scores)
# =============================================================================

def plot_contamination_scores(all_results, dataset_name, model_name, metric, output_dir):
    """Vergleicht Scores für ein Modell."""
    if dataset_name not in all_results:
        return
    if model_name not in all_results[dataset_name]:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    
    for score_type in ["deviation", "reconstruction", "decision"]:
        if score_type not in all_results[dataset_name][model_name]:
            continue
        
        data = all_results[dataset_name][model_name][score_type]
        contam_levels = np.array(data["contamination_levels"])
        values = np.array(data[metric])
        
        color = COLORS_SCORES.get(score_type, None)
        ax.plot(contam_levels, values[:, 0], '-o',
                label=score_type, color=color, linewidth=2, markersize=6)
        ax.fill_between(contam_levels,
                        values[:, 0] - values[:, 1],
                        values[:, 0] + values[:, 1],
                        alpha=0.2, color=color)
        has_data = True
    
    if not has_data:
        plt.close()
        return
    
    ax.set_xlabel("Contamination Level", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"{dataset_name} - {model_name}\n{metric.upper()} vs Contamination", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    filename = output_dir / f"contamination_scores_{dataset_name.replace('.npz', '')}_{model_name}_{metric}.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename.name}")


# =============================================================================
# PLOT 3: Percentile - Modellvergleich pro Score
# (X=Percentile, Y=F1/Precision/Recall, Legende=Modelle)
# =============================================================================

def plot_percentile_models(all_extreme_cases, dataset_name, score_type, metric,
                           case_key, output_dir):
    """Vergleicht Modelle für einen Score-Typ."""
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    
    for model_name in all_extreme_cases.keys():
        if score_type not in all_extreme_cases[model_name]:
            continue
        if dataset_name not in all_extreme_cases[model_name][score_type]:
            continue
        if case_key not in all_extreme_cases[model_name][score_type][dataset_name]:
            continue
        
        metrics_data = all_extreme_cases[model_name][score_type][dataset_name][case_key]["threshold_metrics"]
        
        percs = sorted(metrics_data.keys())
        means = [metrics_data[p][f"{metric}_mean"] for p in percs]
        stds = [metrics_data[p][f"{metric}_std"] for p in percs]
        
        color = COLORS_MODELS.get(model_name, None)
        ax.plot(percs, means, '-o',
                label=model_name, color=color, linewidth=2, markersize=6)
        ax.fill_between(percs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=color)
        has_data = True
    
    if not has_data:
        plt.close()
        return
    
    case_title = "No Contamination" if case_key == "no_contamination" else "Full Contamination"
    ax.set_xlabel("Threshold Percentile", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{dataset_name} - {score_type} - {case_title}\n{metric.capitalize()}", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    filename = output_dir / f"percentile_models_{metric}_{dataset_name.replace('.npz', '')}_{score_type}_{case_key}.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename.name}")


# =============================================================================
# PLOT 4: Percentile - Score-Vergleich pro Modell (NEU!)
# (X=Percentile, Y=F1/Precision/Recall, Legende=Scores)
# =============================================================================

def plot_percentile_scores(all_extreme_cases, dataset_name, model_name, metric,
                           case_key, output_dir):
    """Vergleicht Scores für ein Modell."""
    if model_name not in all_extreme_cases:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    
    for score_type in ["deviation", "reconstruction", "decision"]:
        if score_type not in all_extreme_cases[model_name]:
            continue
        if dataset_name not in all_extreme_cases[model_name][score_type]:
            continue
        if case_key not in all_extreme_cases[model_name][score_type][dataset_name]:
            continue
        
        metrics_data = all_extreme_cases[model_name][score_type][dataset_name][case_key]["threshold_metrics"]
        
        percs = sorted(metrics_data.keys())
        means = [metrics_data[p][f"{metric}_mean"] for p in percs]
        stds = [metrics_data[p][f"{metric}_std"] for p in percs]
        
        color = COLORS_SCORES.get(score_type, None)
        ax.plot(percs, means, '-o',
                label=score_type, color=color, linewidth=2, markersize=6)
        ax.fill_between(percs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=color)
        has_data = True
    
    if not has_data:
        plt.close()
        return
    
    case_title = "No Contamination" if case_key == "no_contamination" else "Full Contamination"
    ax.set_xlabel("Threshold Percentile", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{dataset_name} - {model_name} - {case_title}\n{metric.capitalize()} vs Threshold", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    filename = output_dir / f"percentile_scores_{metric}_{dataset_name.replace('.npz', '')}_{model_name}_{case_key}.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename.name}")


# =============================================================================
# MAIN
# =============================================================================

def create_all_plots(merged, output_dir):
    """Erstellt alle Plots."""
    all_results = merged["all_results_combined"]
    all_extreme = merged["all_extreme_cases"]
    datasets = get_all_datasets(merged)
    models = list(all_extreme.keys())
    
    print(f"\nDatasets: {datasets}")
    print(f"Modelle: {models}")
    
    # =========================================================================
    # CONTAMINATION PLOTS (wenn all_results_combined vorhanden)
    # =========================================================================
    if all_results:
        # Plot 1: Modellvergleich pro Score
        print("\n" + "="*60)
        print("[1] CONTAMINATION - Modellvergleich pro Score")
        print("    (X=Contamination, Y=AUROC/AUPRC, Legende=Modelle)")
        print("="*60)
        for dataset_name in datasets:
            print(f"\n--- {dataset_name} ---")
            for score_type in ["deviation", "reconstruction", "decision"]:
                for metric in ["auroc", "auprc"]:
                    plot_contamination_models(
                        all_results, dataset_name, score_type, metric, output_dir
                    )
        
        # Plot 2: Score-Vergleich pro Modell
        print("\n" + "="*60)
        print("[2] CONTAMINATION - Score-Vergleich pro Modell")
        print("    (X=Contamination, Y=AUROC/AUPRC, Legende=Scores)")
        print("="*60)
        for dataset_name in datasets:
            print(f"\n--- {dataset_name} ---")
            for model_name in all_results.get(dataset_name, {}).keys():
                for metric in ["auroc", "auprc"]:
                    plot_contamination_scores(
                        all_results, dataset_name, model_name, metric, output_dir
                    )
    
    # =========================================================================
    # PERCENTILE PLOTS (Extreme Cases)
    # =========================================================================
    if all_extreme:
        # Plot 3: Modellvergleich pro Score
        print("\n" + "="*60)
        print("[3] PERCENTILE - Modellvergleich pro Score")
        print("    (X=Percentile, Y=F1/Precision/Recall, Legende=Modelle)")
        print("="*60)
        for dataset_name in datasets:
            print(f"\n--- {dataset_name} ---")
            for score_type in ["deviation", "reconstruction", "decision"]:
                for case_key in ["no_contamination", "full_contamination"]:
                    for metric in ["f1", "precision", "recall"]:
                        plot_percentile_models(
                            all_extreme, dataset_name, score_type, metric, case_key, output_dir
                        )
        
        # Plot 4: Score-Vergleich pro Modell
        print("\n" + "="*60)
        print("[4] PERCENTILE - Score-Vergleich pro Modell")
        print("    (X=Percentile, Y=F1/Precision/Recall, Legende=Scores)")
        print("="*60)
        for dataset_name in datasets:
            print(f"\n--- {dataset_name} ---")
            for model_name in models:
                for case_key in ["no_contamination", "full_contamination"]:
                    for metric in ["f1", "precision", "recall"]:
                        plot_percentile_scores(
                            all_extreme, dataset_name, model_name, metric, case_key, output_dir
                        )


if __name__ == "__main__":
    
    # =========================================================================
    # DATEIEN HIER EINTRAGEN
    # =========================================================================
    result_files = [
        Path("./results_flow/results_data/extreme_cases_5_campaign_20251213_061401.joblib"),
        Path("./results_diff_einzeln/results_data/extreme_cases_5_campaign_20251214_085059.joblib"),
        Path("./results_tccm/results_data/extreme_cases_5_campaign_20251214_211153.joblib"),
    ]
    
    output_dir = Path("./plots_combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged = load_and_merge_results(result_files)
    create_all_plots(merged, output_dir)
    
    print("\n" + "="*60)
    print(f"FERTIG! Plots in: {output_dir}")
    print("="*60)