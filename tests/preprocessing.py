# test.py

from ForestDiffusion import ForestDiffusionModel
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")

#anomalies can't be identfied if they are the first?!!! -> solved by is_first_tx feature



def uniqueness(df):
    """
    Creates a 'valid_ref' column indicating reference data anomalies.
    Assumption: correct references appear more frequently than anomalies.
    Preserves 'original_index' column if present.

    logic: combo cols combination (# iteratively increasing) frequency analysis.


    Impotant assumption: correct references appear more frequently than anomalies. 
    And that the exact same mistake (exactly same wrong reference combination) is only done once 
    """
    df = df.copy()
    
    # safe original index column if exists
    has_original_index = 'original_index' in df.columns
    if has_original_index:
        original_index_col = df['original_index'].copy()
    
    df = df.drop(columns=['business_partner_name'], errors='ignore')
    
    #relevant cols for analyzing reference validity
    combo_cols = ["ref_name", "ref_iban", "ref_swift", "pay_method", 
                  "channel", "currency", "trns_type"]
    #calculate frequencies for combinations of the relevant columns
    for col in range(len(combo_cols)):
        combo_observed = combo_cols[:col+1]
        combo_counts = (
            df.groupby(combo_observed, dropna=False)
            .size()
            .reset_index(name=f'combination_freq_{col+1}')
        )
        df = df.merge(combo_counts, on=combo_observed, how='left')

    k = len(combo_cols)
    freq_cols = [f'combination_freq_{i}' for i in range(1, k + 1)]
    #count how many different frequencies exist per row
    all_equal = (df[freq_cols].nunique(axis=1, dropna=False) == 1)

    at_least_one_one = (df[freq_cols] == 1).any(axis=1)
    
    df['valid_ref'] = (
        all_equal | (~all_equal & ~at_least_one_one)
    ).astype(int)
    
    df = df.drop(columns=combo_cols + freq_cols)
    df = df.drop(columns=["ref_bank", "paym_note"], errors='ignore')
    
    # Original-Index wiederherstellen
    if has_original_index:
        df['original_index'] = original_index_col
    
    return df


def create_time_series_features(df):
    """Enhanced time series feature engineering with better imputation."""
    df = df.copy()
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values(["bank_account_uuid", "date_post"])

    group_cols = ['bank_account_uuid', 'ref_name']

    # Rolling features
    mean_rolling = lambda x: x.rolling(5, min_periods=1).mean()
    
    
    df['amount_mean_5'] = df.groupby(group_cols)['amount'].transform(mean_rolling)

    std_rolling = lambda x: x.rolling(5, min_periods=1).std()
    df['amount_std_5'] = df.groupby(group_cols)['amount'].transform(std_rolling)
    
    # Imputation für amount_std_5: group-Median → 0
    df['amount_std_5'] = df['amount_std_5'].fillna(0)
    
    df['amount_change'] = df.groupby(group_cols)['amount'].diff()
    df['amount_change'] = df['amount_change'].fillna(0)
    
    # Time since last transaction
    ts_median = df.groupby(group_cols)['date_post'].diff().dt.days.median()

    df['time_since_last_tx'] = df.groupby(group_cols)['date_post'].diff().dt.days
    df['time_since_last_tx'] = df['time_since_last_tx'].fillna(ts_median)

    #cap for off payments that happen more than once 
    df.loc[df['time_since_last_tx'] > 55, 'time_since_last_tx'] = ts_median

    # Transaction count per partner (cumulative)
    df['partner_tx_count'] = df.groupby(group_cols).cumcount() + 1
    
    # Additional features
    df['is_first_tx'] = (df['partner_tx_count'] == 1).astype(int)

    return df


def load_business_dataset():
    """Load the business transactions dataset."""
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data" / "bd_30k.csv"
    df = pd.read_csv(file_path)
    return df


def prepare_data():
    """
    Main data preparation pipeline with index mapping.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns)
    """
    # Load original data
    df_original = load_business_dataset()
    
    # Save complete original data for later reference
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    df_original.to_csv(output_dir / "original_data.csv", index=True)
    print("✓ Original data saved to data/original_data.csv")
    
    # Start processing
    df = df_original.copy()
    df['original_index'] = df.index
    
    # Feature engineering
    df = create_time_series_features(df)
    
    # Shuffle and sort
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sort_values(["date_post"])
    df = df.drop("date_post", axis=1)

    # Create target
    target_col = "anomaly_description"
    anomaly_mask = df[target_col].notna()
    df[target_col] = anomaly_mask.astype(int)

    # Time-based split (80/20)
    split_index = int(0.8 * len(df))
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Split data
    X_train = X[:split_index].copy()
    X_test = X[split_index:].copy()
    y_train = y[:split_index].copy()
    y_test = y[split_index:].copy()

    # Create index mappings BEFORE uniqueness transformation
    train_mapping = pd.DataFrame({
        'transformed_index': range(len(X_train)),
        'original_index': X_train['original_index'].values
    })
    
    test_mapping = pd.DataFrame({
        'transformed_index': range(len(X_test)),
        'original_index': X_test['original_index'].values
    })
    
    # Save mappings
    train_mapping.to_csv(output_dir / "train_mapping.csv", index=False)
    test_mapping.to_csv(output_dir / "test_mapping.csv", index=False)
    print("✓ Index mappings saved to data/train_mapping.csv and data/test_mapping.csv")
    
    # Apply uniqueness transformation
    X_train = uniqueness(X_train)
    X_test = uniqueness(X_test)
    
    # Remove original_index column (not a feature for modeling)
    X_train = X_train.drop(columns=['original_index'])
    X_test = X_test.drop(columns=['original_index'])
    
    # Save feature column names (nur für NumPy → DataFrame Konvertierung)
    feature_columns = X_train.columns.tolist()
    with open(output_dir / "feature_columns.json", 'w') as f:
        json.dump(feature_columns, f, indent=2)
    print("✓ Feature columns saved to data/feature_columns.json")
    
    # ================================
    # DELETE : JUST FOR TESTING PURPOSES
    # ================================
    train_full = X_train.copy()
    train_full["target"] = y_train.values
    train_full["set"] = "train"
    train_full = train_full.merge(
        train_mapping,
        left_index=True,
        right_on="transformed_index",
        how="left"
    )

    test_full = X_test.copy()
    test_full["target"] = y_test.values
    test_full["set"] = "test"
    test_full = test_full.merge(
        test_mapping,
        left_index=True,
        right_on="transformed_index",
        how="left"
    )

    full_preprocessed = pd.concat([train_full, test_full], axis=0, ignore_index=True)
    full_preprocessed.to_csv(output_dir / "preprocessed_full_dataset.csv", index=False)
    print("✓ preprocessed_full_dataset.csv saved to data/")


    return X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns


def get_original_rows(transformed_indices, mapping_df, original_data_path="data/original_data.csv"):
    """
    Get original rows from transformed indices.
    
    Args:
        transformed_indices: List or array of indices in transformed data
        mapping_df: DataFrame with 'transformed_index' and 'original_index' columns
        original_data_path: Path to original data CSV
    
    Returns:
        DataFrame: Original rows corresponding to transformed indices
    """
    # Get original indices
    original_indices = mapping_df[
        mapping_df['transformed_index'].isin(transformed_indices)
    ]['original_index'].values
    
    # Load original data
    df_original = pd.read_csv(original_data_path, index_col=0)
    
    # Return original rows
    return df_original.loc[original_indices]


def main():
    """Main execution function."""
    X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns = prepare_data()
    
    print("\n=== Data Preparation Complete ===")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_columns)}")
    print(f"Anomalies in train: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Anomalies in test: {y_test.sum()} ({y_test.mean():.2%})")
    print(f"\nFeature columns: {feature_columns}")
    
    return X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns


if __name__ == "__main__":
    main()






# from test import prepare_data

# # Daten laden
# X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns = prepare_data()

# # Jetzt arbeiten...
# print(X_train.head())