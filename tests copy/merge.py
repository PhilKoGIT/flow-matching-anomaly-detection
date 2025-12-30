
# plot_results_compact.py
# Kompakte Version: Alle Plots gruppiert in wenigen PDFs

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from matplotlib.ticker import MultipleLocator

SHOW_ONLY_MODELS = [
    #main
    "ForestDiffusion_nt20_dk20",
    "ForestFlow_nt20_dk20",
    "TCCM_nt20",

    # "ForestFlow_nt20_dk10",
    # "ForestFlow_nt20_dk20",
    # "ForestFlow_nt50_dk20",

    # "ForestDiffusion_nt20_dk20",
    # "ForestDiffusion_nt50_dk10",
    # "ForestDiffusion_nt50_dk20",

    # "TCCM_nt5",
    # "TCCM_nt20",
]

COLORS_MODELS = {

    #main and business
    "ForestDiffusion_nt20_dk20": "green",
    "ForestFlow_nt20_dk20": "blue",
    "TCCM_nt20": "red",
        
    "TCCM_nt5": "orange",
    # "ForestDiffusion_nt50_dk10": "green",
    # "ForestDiffusion_nt20_dk20": "red",
    # "ForestDiffusion_nt50_dk20": "blue",

    # "ForestFlow_nt20_dk10" : "green",
    # "ForestFlow_nt20_dk20": "red",
    # "ForestFlow_nt50_dk20": "blue",

    # "TCCM_nt20": "red",
    # "TCCM_nt5": "blue",
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


def get_dataset_name(merged):
    """Findet den Dataset-Namen."""
    if merged["all_results_combined"]:
        return list(merged["all_results_combined"].keys())[0]
    for model in merged["all_extreme_cases"].values():
        for score in model.values():
            if score:
                return list(score.keys())[0]
    return "Unknown"


def get_all_models(merged):
    """Findet alle Modelle (gefiltert falls SHOW_ONLY_MODELS gesetzt)."""
    models = set()
    for dataset_data in merged["all_results_combined"].values():
        models.update(dataset_data.keys())
    models.update(merged["all_extreme_cases"].keys())
    
    # Filter anwenden
    if SHOW_ONLY_MODELS:
        models = models & set(SHOW_ONLY_MODELS)
    
    return sorted(models)

def get_display_name(dataset_name):
    """Wandelt den Dataset-Namen in einen schönen Anzeigenamen um."""
    name_lower = dataset_name.lower()
    if "campaign" in name_lower:
        return "Campaign Dataset"
    elif "business" in name_lower:
        return "Business Dataset"
    else:
        return dataset_name.replace('.npz', '').replace('_', ' ').title()

# =============================================================================
# CONTAMINATION PLOTS
# =============================================================================

def plot_contamination_models_ax(ax, all_results, dataset_name, score_type, metric):
    """Plottet Modellvergleich für einen Score auf gegebene Axes."""
    if dataset_name not in all_results:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return False
    
    has_data = False
    for model_name in all_results[dataset_name].keys():
        if SHOW_ONLY_MODELS and model_name not in SHOW_ONLY_MODELS:
            continue
        if score_type not in all_results[dataset_name][model_name]:
            continue
        
        data = all_results[dataset_name][model_name][score_type]
        contam_levels = np.array(data["contamination_levels"])
        values = np.array(data[metric])
        print(f"{model_name} - {score_type} - {metric}: {values[:, 0]}")

        color = COLORS_MODELS.get(model_name, None)
        ax.plot(contam_levels, values[:, 0], '-o',
                label=model_name, color=color, linewidth=2, markersize=5)
        ax.fill_between(contam_levels,
                        values[:, 0] - values[:, 1],
                        values[:, 0] + values[:, 1],
                        alpha=0.2, color=color)
        has_data = True
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel("Contamination Level", fontsize=15)
    ax.set_ylabel(metric.upper(), fontsize=15)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Minor-Ticks (ohne Labels) alle 0.1
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # Grid für beide aktivieren
    ax.grid(True, which='major', alpha=0.8)
    ax.grid(True, which='minor', alpha=0.8)
    ax.set_title(f"{score_type.capitalize()} - {metric.upper()}", fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.05)
    if has_data:
        ax.legend(framealpha=1.0, prop={'weight': 'normal', 'size': 14}, labelspacing=0.2)
    return has_data


def plot_contamination_scores_ax(ax, all_results, dataset_name, model_name, metric):
    """Plottet Score-Vergleich für ein Modell auf gegebene Axes."""
    if dataset_name not in all_results:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return False
    if model_name not in all_results[dataset_name]:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return False
    
    has_data = False
    for score_type in ["deviation", "reconstruction", "decision"]:
        if score_type not in all_results[dataset_name][model_name]:
            continue
        
        data = all_results[dataset_name][model_name][score_type]
        contam_levels = np.array(data["contamination_levels"])
        values = np.array(data[metric])
        
        color = COLORS_SCORES.get(score_type, None)
        ax.plot(contam_levels, values[:, 0], '-o',
                label=score_type, color=color, linewidth=2, markersize=5)
        ax.fill_between(contam_levels,
                        values[:, 0] - values[:, 1],
                        values[:, 0] + values[:, 1],
                        alpha=0.2, color=color)
        has_data = True
    ax.tick_params(axis='both', labelsize=15)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Minor-Ticks (ohne Labels) alle 0.1
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # Grid für beide aktivieren
    ax.grid(True, which='major', alpha=0.8)
    ax.grid(True, which='minor', alpha=0.8)
    ax.set_xlabel("Contamination Level", fontsize=15)
    ax.set_ylabel(metric.upper(), fontsize=15)
    ax.set_title(f"{model_name}\n{metric.upper()}", fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.05)
    if has_data:
        ax.legend(framealpha=1.0, prop={'weight': 'normal', 'size': 14}, labelspacing=0.2)
    return has_data


# =============================================================================
# PERCENTILE PLOTS
# =============================================================================

def plot_percentile_models_ax(ax, all_extreme_cases, dataset_name, score_type, metric, case_key):
    """Plottet Modellvergleich für Percentile auf gegebene Axes."""
    has_data = False
    
    for model_name in all_extreme_cases.keys():
        if SHOW_ONLY_MODELS and model_name not in SHOW_ONLY_MODELS:
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
        ax.plot(percs, means, '-o',
                label=model_name, color=color, linewidth=2, markersize=5)
        ax.fill_between(percs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=color)
        has_data = True
    ax.tick_params(axis='both', labelsize=15)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='major', alpha=0.8)
    ax.grid(True, which='minor', alpha=0.8)
    ax.set_xlabel("Threshold Percentile", fontsize=15)
    ax.set_ylabel(metric.capitalize(), fontsize=15)
    ax.set_title(f"{score_type.capitalize()} - {metric.capitalize()}", fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.05)
    if has_data:
        ax.legend(framealpha=1.0, prop={'weight': 'normal', 'size': 14}, labelspacing=0.2)
    return has_data


def plot_percentile_scores_ax(ax, all_extreme_cases, dataset_name, model_name, metric, case_key):
    """Plottet Score-Vergleich für Percentile auf gegebene Axes."""
    if model_name not in all_extreme_cases:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return False
    
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
                label=score_type, color=color, linewidth=2, markersize=5)
        ax.fill_between(percs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=color)
        has_data = True
    ax.tick_params(axis='both', labelsize=13)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Minor-Ticks (ohne Labels) alle 0.1
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(10))

    # Grid für beide aktivieren
    ax.grid(True, which='major', alpha=0.8)
    ax.grid(True, which='minor', alpha=0.8)

    ax.set_xlabel("Threshold Percentile", fontsize=15)
    ax.set_ylabel(metric.capitalize(), fontsize=15)
    ax.set_title(f"{model_name}\n{metric.capitalize()}", fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.05)
    if has_data:
        ax.legend(framealpha=1.0, prop={'weight': 'normal', 'size': 14}, labelspacing=0.2)
    return has_data


# =============================================================================
# HAUPTFUNKTION: ERSTELLE KOMPAKTE PDFs
# =============================================================================

def create_compact_pdfs(merged, output_dir):
    """Erstellt kompakte PDFs mit allen Plots."""
    all_results = merged["all_results_combined"]
    all_extreme = merged["all_extreme_cases"]
    dataset_name = get_dataset_name(merged)
    display_name = get_display_name(dataset_name) 
    models = get_all_models(merged)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Modelle: {models}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PDF 1: CONTAMINATION ANALYSIS
    # =========================================================================
    if all_results:
        pdf_path = output_dir / f"contamination_analysis_{display_name}.pdf"
        with PdfPages(pdf_path) as pdf:
            
            # Seite 1: Modellvergleich pro Score (2 Zeilen × 3 Spalten)
            # Zeilen: AUROC, AUPRC | Spalten: deviation, reconstruction, decision
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"{display_name} - Contamination: score-centric", fontsize=20, fontweight='semibold')
            
            for col, score_type in enumerate(["deviation", "reconstruction", "decision"]):
                for row, metric in enumerate(["auroc", "auprc"]):
                    plot_contamination_models_ax(axes[row, col], all_results, dataset_name, score_type, metric)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
            pdf.savefig(fig)
            plt.close()
            
            # Seite 2: Score-Vergleich pro Modell (2 Zeilen × N Spalten)
            n_models = len(models)
            if n_models > 0:
                fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
                fig.suptitle(f"{display_name} - Contamination: model-centric", fontsize=20, fontweight='semibold')
                
                if n_models == 1:
                    axes = axes.reshape(2, 1)
                
                for col, model_name in enumerate(models):
                    for row, metric in enumerate(["auroc", "auprc"]):
                        plot_contamination_scores_ax(axes[row, col], all_results, dataset_name, model_name, metric)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
                pdf.savefig(fig)
                plt.close()
        
        print(f"[✓] Gespeichert: {pdf_path}")
    
    # =========================================================================
    # PDF 2: PERCENTILE ANALYSIS
    # =========================================================================
    if all_extreme:
        pdf_path = output_dir / f"percentile_analysis_{display_name}.pdf"
        with PdfPages(pdf_path) as pdf:
            
            for case_key, case_title in [("no_contamination", "No Contamination"), 
                                          ("full_contamination", "Full Contamination")]:
                
                # Seite: Modellvergleich pro Score (3 Zeilen × 3 Spalten)
                # Zeilen: F1, Precision, Recall | Spalten: deviation, reconstruction, decision
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                fig.suptitle(f"{display_name} - Percentile ({case_title}): score-centric", 
                            fontsize=20, fontweight='semibold')
                
                for col, score_type in enumerate(["deviation", "reconstruction", "decision"]):
                    for row, metric in enumerate(["f1", "precision", "recall"]):
                        plot_percentile_models_ax(axes[row, col], all_extreme, dataset_name, 
                                                  score_type, metric, case_key)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
                pdf.savefig(fig)
                plt.close()
                
                # Seite: Score-Vergleich pro Modell (3 Zeilen × N Spalten)
                n_models = len(models)
                if n_models > 0:
                    fig, axes = plt.subplots(3, n_models, figsize=(5*n_models, 12))
                    fig.suptitle(f"{display_name} - Percentile ({case_title}): model-centric", 
                                fontsize=20, fontweight='semibold')
                    
                    if n_models == 1:
                        axes = axes.reshape(3, 1)
                    
                    for col, model_name in enumerate(models):
                        for row, metric in enumerate(["f1", "precision", "recall"]):
                            plot_percentile_scores_ax(axes[row, col], all_extreme, dataset_name,
                                                      model_name, metric, case_key)
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
                    pdf.savefig(fig)
                    plt.close()
        
        print(f"[✓] Gespeichert: {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # DATEIEN HIER EINTRAGEN
    # =========================================================================
    result_files = [
        # Path("./0_results_diff/results_data/extreme_cases_5_campaign_20251221_233310.joblib"),
        # Path("./0_results_flow/results_data/extreme_cases_5_campaign_20251221_201216.joblib"),
        # Path("./0_results_tccm/results_data/extreme_cases_5_campaign_20251223_093801.joblib"),


        #only for the business dataset experiment
        Path("./0_results_business/results_data/extreme_cases_business_dataset_middle.csv_20251222_173347.joblib"),
        
    ]
    
    output_dir = Path("./1_results_business")
    

    merged = load_and_merge_results(result_files)
    print("Gefundene Modelle in all_results_combined:")
    for ds, models in merged["all_results_combined"].items():
        print(f"  {ds}: {list(models.keys())}")
    print("Gefundene Modelle in all_extreme_cases:")
    print(f"  {list(merged['all_extreme_cases'].keys())}")
    
    print("\n" + "="*60)
    print("ERSTELLE KOMPAKTE PDFs")
    print("="*60)
    create_compact_pdfs(merged, output_dir)
    
    print("\n" + "="*60)
    print(f"FERTIG! PDFs in: {output_dir}")
    print("="*60)