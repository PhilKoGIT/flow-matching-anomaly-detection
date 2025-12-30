# plot_model_specific_scores.py
# Jedes Modell mit seinem spezifischen Score gegeneinander plotten

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from matplotlib.ticker import MultipleLocator

# Modell -> Score Zuordnung
MODEL_SCORE_MAPPING = {
    #"TCCM_nt20": "decision",
    "TCCM_nt20": "decision",
    "ForestFlow_nt20_dk20": "reconstruction",
    "ForestDiffusion_nt20_dk20": "deviation",
}

COLORS_MODELS = {
    "ForestDiffusion_nt20_dk20": "green",
    "ForestFlow_nt20_dk20": "blue",
    "TCCM_nt20": "red",
}

# Schönere Labels für die Legende
DISPLAY_LABELS = {
    "TCCM_nt20": "TCCM (decision)",
    "ForestFlow_nt20_dk20": "ForestFlow (reconstruction)",
    "ForestDiffusion_nt20_dk20": "ForestDiffusion (deviation)",
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
        
        for dataset, models in data.get("all_results_combined", {}).items():
            if dataset not in merged["all_results_combined"]:
                merged["all_results_combined"][dataset] = {}
            for model, scores in models.items():
                if model not in merged["all_results_combined"][dataset]:
                    merged["all_results_combined"][dataset][model] = {}
                merged["all_results_combined"][dataset][model].update(scores)
        
        for model, scores in data.get("all_extreme_cases", {}).items():
            if model not in merged["all_extreme_cases"]:
                merged["all_extreme_cases"][model] = {}
            for score, datasets in scores.items():
                if score not in merged["all_extreme_cases"][model]:
                    merged["all_extreme_cases"][model][score] = {}
                merged["all_extreme_cases"][model][score].update(datasets)
    
    return merged


def get_dataset_name(merged):
    """Findet den Dataset-Namen."""
    if merged["all_results_combined"]:
        return list(merged["all_results_combined"].keys())[0]
    for model in merged["all_extreme_cases"].values():
        for score in model.values():
            if score:
                return list(score.keys())[0]
    return "Unknown"


def get_display_name(dataset_name):
    """Wandelt den Dataset-Namen in einen schönen Anzeigenamen um."""
    name_lower = dataset_name.lower()
    if "campaign" in name_lower:
        return "Campaign Dataset"
    elif "business" in name_lower:
        return "Business Dataset"
    else:
        return dataset_name.replace('.npz', '').replace('_', ' ').title()


def plot_contamination_model_specific(ax, all_results, dataset_name, metric):
    """Plottet jeden Modell mit seinem spezifischen Score."""
    if dataset_name not in all_results:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return False
    
    has_data = False
    for model_name, score_type in MODEL_SCORE_MAPPING.items():
        if model_name not in all_results[dataset_name]:
            print(f"[!] Modell {model_name} nicht gefunden")
            continue
        if score_type not in all_results[dataset_name][model_name]:
            print(f"[!] Score {score_type} für {model_name} nicht gefunden")
            continue
        
        data = all_results[dataset_name][model_name][score_type]
        contam_levels = np.array(data["contamination_levels"])
        values = np.array(data[metric])
        
        print(f"{model_name} ({score_type}) - {metric}: {values[:, 0]}")
        
        color = COLORS_MODELS.get(model_name, None)
        label = DISPLAY_LABELS.get(model_name, model_name)
        
        ax.plot(contam_levels, values[:, 0], '-o',
                label=label, color=color, linewidth=2, markersize=6)
        ax.fill_between(contam_levels,
                        values[:, 0] - values[:, 1],
                        values[:, 0] + values[:, 1],
                        alpha=0.2, color=color)
        has_data = True
    
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("Contamination Level", fontsize=15)
    ax.set_ylabel(metric.upper(), fontsize=15)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(True, which='major', alpha=0.8)
    ax.grid(True, which='minor', alpha=0.4)
    ax.set_title(f"{metric.upper()}", fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    if has_data:
        ax.legend(framealpha=1.0, prop={'weight': 'normal', 'size': 12}, 
                  labelspacing=0.3, loc='best')
    return has_data


def plot_percentile_model_specific(ax, all_extreme_cases, dataset_name, metric, case_key):
    """Plottet jeden Modell mit seinem spezifischen Score für Percentile."""
    has_data = False
    
    for model_name, score_type in MODEL_SCORE_MAPPING.items():
        if model_name not in all_extreme_cases:
            continue
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
        label = DISPLAY_LABELS.get(model_name, model_name)
        
        ax.plot(percs, means, '-o',
                label=label, color=color, linewidth=2, markersize=6)
        ax.fill_between(percs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=color)
        has_data = True
    
    ax.tick_params(axis='both', labelsize=14)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='major', alpha=0.8)
    ax.grid(True, which='minor', alpha=0.4)
    ax.set_xlabel("Threshold Percentile", fontsize=15)
    ax.set_ylabel(metric.capitalize(), fontsize=15)
    ax.set_title(f"{metric.capitalize()}", fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    if has_data:
        ax.legend(framealpha=1.0, prop={'weight': 'normal', 'size': 12}, 
                  labelspacing=0.3, loc='best')
    return has_data


def create_model_specific_pdfs(merged, output_dir):
    """Erstellt PDFs mit modellspezifischen Scores."""
    all_results = merged["all_results_combined"]
    all_extreme = merged["all_extreme_cases"]
    dataset_name = get_dataset_name(merged)
    display_name = get_display_name(dataset_name)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Display Name: {display_name}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PDF: CONTAMINATION - Model-spezifische Scores
    # =========================================================================
    if all_results:
        pdf_path = output_dir / f"model_specific_contamination_{display_name}.pdf"
        with PdfPages(pdf_path) as pdf:
            
            # Eine Seite mit 1x2 Grid (AUROC, AUPRC)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"{display_name} - Model Comparison (Best Scores)", 
                        fontsize=18, fontweight='semibold')
            
            for col, metric in enumerate(["auroc", "auprc"]):
                plot_contamination_model_specific(axes[col], all_results, dataset_name, metric)
            
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=150)
            plt.close()
        
        print(f"[✓] Gespeichert: {pdf_path}")
    
    # =========================================================================
    # PDF: PERCENTILE - Model-spezifische Scores
    # =========================================================================
    if all_extreme:
        pdf_path = output_dir / f"model_specific_percentile_{display_name}.pdf"
        with PdfPages(pdf_path) as pdf:
            
            for case_key, case_title in [("no_contamination", "No Contamination"), 
                                          ("full_contamination", "Full Contamination")]:
                
                # Eine Seite mit 1x3 Grid (F1, Precision, Recall)
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"{display_name} - Percentile ({case_title})", 
                            fontsize=18, fontweight='semibold')
                
                for col, metric in enumerate(["f1", "precision", "recall"]):
                    plot_percentile_model_specific(axes[col], all_extreme, dataset_name, 
                                                   metric, case_key)
                
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                pdf.savefig(fig, dpi=150)
                plt.close()
        
        print(f"[✓] Gespeichert: {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    result_files = [
        Path("./0_results_diff/results_data/extreme_cases_5_campaign_20251221_233310.joblib"),
        Path("./0_results_flow/results_data/extreme_cases_5_campaign_20251221_201216.joblib"),
        Path("./0_results_tccm/results_data/extreme_cases_5_campaign_20251223_093801.joblib"),
    ]
    
    output_dir = Path("./1_results_best")
    
    merged = load_and_merge_results(result_files)
    
    print("\nGefundene Modelle in all_results_combined:")
    for ds, models in merged["all_results_combined"].items():
        print(f"  {ds}: {list(models.keys())}")
    
    print("\nGefundene Modelle in all_extreme_cases:")
    print(f"  {list(merged['all_extreme_cases'].keys())}")
    
    print("\n" + "="*60)
    print("MODEL-SCORE ZUORDNUNG:")
    for model, score in MODEL_SCORE_MAPPING.items():
        print(f"  {model} → {score}")
    print("="*60)
    
    create_model_specific_pdfs(merged, output_dir)
    
    print("\n" + "="*60)
    print(f"FERTIG! PDFs in: {output_dir}")
    print("="*60)