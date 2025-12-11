# preprocessing_bd_supervised.py
"""
Preprocessing for business transaction anomaly detection.
Properly handles categorical variables with different encoding strategies:
- Low cardinality: One-Hot Encoding (pay_method, channel, currency, trns_type)
- High cardinality: Frequency Encoding (ref_name, bank_account_uuid, ref_iban, ref_swift)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")


def load_business_dataset():
    """Load the business transactions dataset."""
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data" / "business_dataset.csv"
    df = pd.read_csv(file_path)
    return df


def create_features(df, is_training=True, encoding_maps=None):
    """
    Create features with proper categorical handling.
    
    Categorical variables:
    - LOW CARDINALITY (One-Hot): pay_method, channel, currency, trns_type
    - HIGH CARDINALITY (Frequency): ref_name, ref_iban, ref_swift, bank_account_uuid
    
    Args:
        df: DataFrame with transactions
        is_training: If True, create encoding maps. If False, use provided maps.
        encoding_maps: Dict of encoding mappings (required if is_training=False)
    
    Returns:
        df: DataFrame with features
        encoding_maps: Dict of encoding mappings
    """
    df = df.copy()
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values(['bank_account_uuid', 'ref_name', 'date_post']).reset_index(drop=True)
    
    if encoding_maps is None:
        encoding_maps = {}
    
    # ===========================================
    # 1. AMOUNT FEATURES (for AMOUNT_ANOMALY)
    # ===========================================
    series_cols = ['bank_account_uuid', 'ref_name']
    
    # Rolling statistics per series
    df['amount_rolling_mean'] = df.groupby(series_cols)['amount'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    df['amount_rolling_std'] = df.groupby(series_cols)['amount'].transform(
        lambda x: x.rolling(10, min_periods=1).std().fillna(0)
    )
    
    # Z-score
    df['amount_zscore'] = (df['amount'] - df['amount_rolling_mean']) / (df['amount_rolling_std'] + 1)
    
    # Ratio to rolling mean
    df['amount_ratio_to_mean'] = df['amount'] / (df['amount_rolling_mean'] + 1)
    
    # ===========================================
    # 2. TIMING FEATURES (for TIMING_ANOMALY, FREQUENCY_ANOMALY)
    # ===========================================
    df['day_of_month'] = df['date_post'].dt.day
    df['day_of_week'] = df['date_post'].dt.dayofweek
    
    # Expected day of month per series
    df['expected_dom'] = df.groupby(series_cols)['day_of_month'].transform('median')
    df['dom_deviation'] = (df['day_of_month'] - df['expected_dom']).abs()
    
    # Time since last transaction
    df['days_since_last'] = df.groupby(series_cols)['date_post'].diff().dt.days
    df['days_since_last'] = df['days_since_last'].fillna(30)
    
    # Multiple transactions in same month
    df['year_month'] = df['date_post'].dt.to_period('M').astype(str)
    df['tx_count_this_month'] = df.groupby(series_cols + ['year_month']).cumcount() + 1
    df['is_duplicate_month'] = (df['tx_count_this_month'] > 1).astype(int)
    
    # ===========================================
    # 3. HIGH CARDINALITY CATEGORICALS - Frequency Encoding
    #    (ref_name, ref_iban, ref_swift, bank_account_uuid)
    # ===========================================
    high_card_cols = ['ref_name', 'ref_iban', 'ref_swift', 'bank_account_uuid']
    
    for col in high_card_cols:
        if is_training:
            # Calculate frequency in training data
            freq_map = df[col].value_counts(normalize=True).to_dict()
            encoding_maps[f'{col}_freq'] = freq_map
        else:
            freq_map = encoding_maps.get(f'{col}_freq', {})
        
        # Apply frequency encoding - unseen values get 0 (very rare)
        df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
    
    # ===========================================
    # 4. LOW CARDINALITY CATEGORICALS - One-Hot Encoding
    #    (pay_method, channel, currency, trns_type)
    # ===========================================
    low_card_cols = ['pay_method', 'channel', 'currency', 'trns_type']
    
    for col in low_card_cols:
        if is_training:
            # Store unique values from training
            unique_vals = df[col].unique().tolist()
            encoding_maps[f'{col}_categories'] = unique_vals
        else:
            unique_vals = encoding_maps.get(f'{col}_categories', [])
        
        # One-hot encode
        for val in unique_vals:
            df[f'{col}_{val}'] = (df[col] == val).astype(int)
        
        # For test: if there's a new category not in training, it gets all zeros
        # which is fine - the model treats it as "unknown"
    
    # ===========================================
    # 5. REFERENCE CONSISTENCY FEATURES
    # ===========================================
    
    # Is this the usual IBAN for this ref_name?
    if is_training:
        # Most common IBAN per ref_name
        most_common_iban = df.groupby('ref_name')['ref_iban'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['ref_name_usual_iban'] = most_common_iban
        
        # Most common SWIFT per ref_name
        most_common_swift = df.groupby('ref_name')['ref_swift'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['ref_name_usual_swift'] = most_common_swift
        
        # Most common pay_method per series
        most_common_method = df.groupby(series_cols)['pay_method'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['series_usual_method'] = most_common_method
        
        # Most common channel per series
        most_common_channel = df.groupby(series_cols)['channel'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['series_usual_channel'] = most_common_channel
    
    # Apply consistency checks
    usual_iban_map = encoding_maps.get('ref_name_usual_iban', {})
    usual_swift_map = encoding_maps.get('ref_name_usual_swift', {})
    usual_method_map = encoding_maps.get('series_usual_method', {})
    usual_channel_map = encoding_maps.get('series_usual_channel', {})
    
    df['usual_iban'] = df['ref_name'].map(usual_iban_map)
    df['is_usual_iban'] = (df['ref_iban'] == df['usual_iban']).astype(int)
    # Unknown ref_name -> assume unusual (0)
    df.loc[df['usual_iban'].isna(), 'is_usual_iban'] = 0
    
    df['usual_swift'] = df['ref_name'].map(usual_swift_map)
    df['is_usual_swift'] = (df['ref_swift'] == df['usual_swift']).astype(int)
    df.loc[df['usual_swift'].isna(), 'is_usual_swift'] = 0
    
    # Series key for method/channel lookup
    df['series_key'] = list(zip(df['bank_account_uuid'], df['ref_name']))
    
    df['usual_method'] = df['series_key'].map(usual_method_map)
    df['is_usual_method'] = (df['pay_method'] == df['usual_method']).astype(int)
    df.loc[df['usual_method'].isna(), 'is_usual_method'] = 0
    
    df['usual_channel'] = df['series_key'].map(usual_channel_map)
    df['is_usual_channel'] = (df['channel'] == df['usual_channel']).astype(int)
    df.loc[df['usual_channel'].isna(), 'is_usual_channel'] = 0
    
    # Combined flags
    df['uses_usual_banking'] = (df['is_usual_iban'] & df['is_usual_swift']).astype(int)
    df['uses_usual_method_channel'] = (df['is_usual_method'] & df['is_usual_channel']).astype(int)
    
    # ===========================================
    # 6. SERIES-LEVEL FEATURES
    # ===========================================
    df['tx_count_in_series'] = df.groupby(series_cols).cumcount() + 1
    df['is_first_tx'] = (df['tx_count_in_series'] == 1).astype(int)
    
    # How many transactions does this ref_name have? (frequency-based)
    df['ref_name_tx_count'] = df.groupby('ref_name')['ref_name'].transform('count')
    df['is_regular_series'] = (df['ref_name_tx_count'] > 5).astype(int)
    
    return df, encoding_maps


def select_features(df, encoding_maps):
    """Select final feature columns."""
    
    # Base numeric features
    feature_cols = [
        # Amount features
        'amount',
        'amount_zscore',
        'amount_ratio_to_mean',
        
        # Timing features
        'day_of_month',
        'day_of_week',
        'dom_deviation',
        'days_since_last',
        'is_duplicate_month',
        
        # High-cardinality frequency features
        'ref_name_freq',
        'ref_iban_freq',
        'ref_swift_freq',
        'bank_account_uuid_freq',
        
        # Consistency features
        'is_usual_iban',
        'is_usual_swift',
        'uses_usual_banking',
        'is_usual_method',
        'is_usual_channel',
        'uses_usual_method_channel',
        
        # Series features
        'tx_count_in_series',
        'is_first_tx',
        'is_regular_series',
    ]
    
    # Add one-hot encoded columns
    low_card_cols = ['pay_method', 'channel', 'currency', 'trns_type']
    for col in low_card_cols:
        categories = encoding_maps.get(f'{col}_categories', [])
        for cat in categories:
            col_name = f'{col}_{cat}'
            if col_name in df.columns:
                feature_cols.append(col_name)
    
    # Keep only existing columns
    available_cols = [c for c in feature_cols if c in df.columns]
    
    return df[available_cols]


def prepare_data_supervised():
    """
    Main data preparation pipeline.
    """
    df_original = load_business_dataset()
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    df_original.to_csv(output_dir / "original_data.csv", index=True)
    
    df = df_original.copy()
    df['original_index'] = df.index
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    
    # Sort by date
    df = df.sort_values('date_post').reset_index(drop=True)
    
    # Target
    target_col = "anomaly_description"
    df['target'] = df[target_col].notna().astype(int)
    
    # ===========================================
    # TIME-BASED SPLIT
    # ===========================================
    split_date = df['date_post'].quantile(0.6)
    
    # SEMI-SUPERVISED: Training only normal data
    train_mask = (df['date_post'] <= split_date) & (df['target'] == 0)
    test_mask = df['date_post'] > split_date
    
    df_train_raw = df[train_mask].copy()
    df_test_raw = df[test_mask].copy()
    
    # Move anomalies from train period to test
    train_period_anomalies = df[(df['date_post'] <= split_date) & (df['target'] == 1)].copy()
    if len(train_period_anomalies) > 0:
        df_test_raw = pd.concat([df_test_raw, train_period_anomalies], axis=0)
        df_test_raw = df_test_raw.sort_values('date_post').reset_index(drop=True)
        print(f"Moved {len(train_period_anomalies)} anomalies from train period to test set")
    
    print(f"Split date: {split_date}")
    print(f"Train: {len(df_train_raw)} rows (CLEAN - {df_train_raw['target'].sum()} anomalies)")
    print(f"Test: {len(df_test_raw)} rows ({df_test_raw['target'].sum()} anomalies)")
    
    # ===========================================
    # FEATURE ENGINEERING
    # ===========================================
    
    # Train: fit encodings on training data only
    df_train, encoding_maps = create_features(df_train_raw, is_training=True)
    
    # Test: use encodings from training, compute features on full data
    df_full_raw = pd.concat([df_train_raw, df_test_raw], axis=0)
    df_full, _ = create_features(df_full_raw, is_training=False, encoding_maps=encoding_maps)
    
    # Extract test rows
    test_indices = df_test_raw['original_index'].values
    df_test = df_full[df_full['original_index'].isin(test_indices)].copy()
    
    # ===========================================
    # PREPARE FINAL DATASETS
    # ===========================================
    y_train = df_train['target']
    y_test = df_test['target']
    
    # Mappings
    train_mapping = pd.DataFrame({
        'transformed_index': range(len(df_train)),
        'original_index': df_train['original_index'].values
    })
    test_mapping = pd.DataFrame({
        'transformed_index': range(len(df_test)),
        'original_index': df_test['original_index'].values
    })
    
    train_mapping.to_csv(output_dir / "train_mapping.csv", index=False)
    test_mapping.to_csv(output_dir / "test_mapping.csv", index=False)
    
    # Select features
    X_train = select_features(df_train, encoding_maps)
    X_test = select_features(df_test, encoding_maps)
    
    # Save feature columns
    feature_columns = X_train.columns.tolist()
    with open(output_dir / "feature_columns.json", 'w') as f:
        json.dump(feature_columns, f, indent=2)
    
    # Save encoding maps (for later use)
    import pickle
    with open(output_dir / "encoding_maps.pkl", 'wb') as f:
        pickle.dump(encoding_maps, f)
    
    # ===========================================
    # DEBUG OUTPUT
    # ===========================================
    train_full = X_train.copy()
    train_full["target"] = y_train.values
    train_full["set"] = "train"
    train_full["original_index"] = df_train['original_index'].values
    
    test_full = X_test.copy()
    test_full["target"] = y_test.values
    test_full["set"] = "test"
    test_full["original_index"] = df_test['original_index'].values
    
    full_preprocessed = pd.concat([train_full, test_full], axis=0, ignore_index=True)
    full_preprocessed.to_csv(output_dir / "preprocessed_full_dataset.csv", index=False)
    print("✓ preprocessed_full_dataset.csv saved to data/")
    
    # Print feature info
    print(f"\n=== Feature Summary ===")
    print(f"Total features: {len(feature_columns)}")
    print(f"Numeric features: amount, amount_zscore, amount_ratio_to_mean, etc.")
    onehot_features = [c for c in feature_columns if any(x in c for x in ['pay_method_', 'channel_', 'currency_', 'trns_type_'])]
    print(f"One-hot features ({len(onehot_features)}): {onehot_features[:5]}..." if len(onehot_features) > 5 else f"One-hot features: {onehot_features}")
    print(f"Frequency features: ref_name_freq, ref_iban_freq, ref_swift_freq, bank_account_uuid_freq")
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns


def main():
    """Main execution function."""
    X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns = prepare_data_supervised()
    
    print("\n=== Data Preparation Complete ===")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_columns)}")
    print(f"Anomalies in train: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Anomalies in test: {y_test.sum()} ({y_test.mean():.2%})")
    print(f"\nFeature columns:\n{feature_columns}")
    
    return X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns


if __name__ == "__main__":
    main()