def load_and_merge_results(filepaths):
    """
    Lädt mehrere Ergebnisdateien und merged sie.
    Annahme: Gleicher Datensatz, aber verschiedene Modelle/Parameter.
    """
    merged_extreme_cases = {}
    dataset_names = None
    all_models = {}
    
    for filepath in filepaths:
        data = joblib.load(filepath)
        print(f"Geladen: {filepath}")
        print(f"  Models: {list(data['models_to_run'].keys())}")
        
        # Dataset-Namen übernehmen (sollten gleich sein)
        if dataset_names is None:
            dataset_names = data['dataset_names']
        else:
            assert dataset_names == data['dataset_names'], \
                f"Dataset mismatch: {dataset_names} vs {data['dataset_names']}"
        
        # Models sammeln
        all_models.update(data['models_to_run'])
        
        # Extreme cases mergen
        for model_name, scores_data in data['all_extreme_cases'].items():
            if model_name not in merged_extreme_cases:
                merged_extreme_cases[model_name] = {}
            
            for score, datasets in scores_data.items():
                if score not in merged_extreme_cases[model_name]:
                    merged_extreme_cases[model_name][score] = {}
                
                merged_extreme_cases[model_name][score].update(datasets)
    
    print(f"\nGesamt: {len(merged_extreme_cases)} Modelle geladen")
    
    return {
        'all_extreme_cases': merged_extreme_cases,
        'dataset_names': dataset_names,
        'models_to_run': all_models
    }


if __name__ == "__main__":
    folder = "results_"
    
    # ============================================================================
    # MEHRERE DATEIPFADE EINTRAGEN
    # ============================================================================
    
    # Option 1: Manuell auflisten
    results_paths = [
        Path(f"./{folder}/results_data/extreme_cases_29_Pima_20251212_164701.joblib"),
        Path(f"./{folder}/results_data/extreme_cases_29_Pima_20251212_180000.joblib"),
        Path(f"./{folder}/results_data/extreme_cases_29_Pima_20251213_100000.joblib"),
    ]
    
    # Option 2: Alle Dateien für einen Datensatz automatisch finden
    # results_dir = Path(f"./{folder}/results_data")
    # results_paths = list(results_dir.glob("extreme_cases_29_Pima_*.joblib"))
    
    # Option 3: Alle Dateien im Ordner
    # results_paths = list(results_dir.glob("extreme_cases_*.joblib"))
    
    # ============================================================================
    # LADEN UND MERGEN
    # ============================================================================
    
    data = load_and_merge_results(results_paths)
    
    all_extreme_cases = data["all_extreme_cases"]
    dataset_names = data["dataset_names"]
    models_to_run = data["models_to_run"]
    
    # Rest bleibt gleich...
    output_dir = "./results_plots"
    
    print("\nErstelle Percentile-Kurven...")
    plot_percentile_curves(all_extreme_cases, dataset_names, output_dir=output_dir)
    
    # usw.