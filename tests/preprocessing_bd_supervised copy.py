# preprocessing_improved.py
"""
Improved preprocessing for business transaction anomaly detection.
Features are designed to detect:
- AMOUNT_ANOMALY: Amount spikes (1.5-2x normal)
- FREQUENCY_ANOMALY: Duplicate monthly payments
- PAYEE_ANOMALY: Changed ref_name/IBAN/SWIFT
- TIMING_ANOMALY: Payment date shifted
- CHANNEL_ANOMALY: Unusual payment method/channel
- IBAN_MISMATCH: Only IBAN changed
- SUBTLE_PAYEE_MISMATCH: Slight name change + new IBAN
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")


def load_business_dataset():
    """Load the business transactions dataset."""
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data" / "business_dataset.csv"
    df = pd.read_csv(file_path)
    return df


def create_features(df, is_training=True, encoders=None):
    """
    Create features optimized for detecting the specific anomaly types.
    
    Args:
        df: DataFrame with transactions
        is_training: If True, fit encoders. If False, use provided encoders.
        encoders: Dict of fitted encoders (required if is_training=False)
    
    Returns:
        df: DataFrame with features
        encoders: Dict of fitted encoders
    """
    df = df.copy()
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values(['bank_account_uuid', 'ref_name', 'date_post']).reset_index(drop=True)
    
    if encoders is None:
        encoders = {}
    
    # Group keys
    account_col = 'bank_account_uuid'
    series_cols = ['bank_account_uuid', 'ref_name']
    
    # ===========================================
    # 1. AMOUNT FEATURES (for AMOUNT_ANOMALY)
    # ===========================================
    
    # Rolling statistics per series
    df['amount_rolling_mean'] = df.groupby(series_cols)['amount'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    df['amount_rolling_std'] = df.groupby(series_cols)['amount'].transform(
        lambda x: x.rolling(10, min_periods=1).std().fillna(0)
    )
    
    # Z-score: How many std devs from rolling mean?
    df['amount_zscore'] = (df['amount'] - df['amount_rolling_mean']) / (df['amount_rolling_std'] + 1)
    
    # Ratio to rolling mean (catches 1.5-2x spikes)
    df['amount_ratio_to_mean'] = df['amount'] / (df['amount_rolling_mean'] + 1)
    
    # Percentile rank within series
    df['amount_percentile'] = df.groupby(series_cols)['amount'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # ===========================================
    # 2. TIMING FEATURES (for TIMING_ANOMALY, FREQUENCY_ANOMALY)
    # ===========================================
    
    df['day_of_month'] = df['date_post'].dt.day
    df['month'] = df['date_post'].dt.month
    df['year'] = df['date_post'].dt.year
    df['day_of_week'] = df['date_post'].dt.dayofweek
    
    # Expected day of month per series (median)
    df['expected_dom'] = df.groupby(series_cols)['day_of_month'].transform('median')
    df['dom_deviation'] = (df['day_of_month'] - df['expected_dom']).abs()
    
    # Time since last transaction in same series
    df['days_since_last'] = df.groupby(series_cols)['date_post'].diff().dt.days
    df['days_since_last'] = df['days_since_last'].fillna(30)  # Default ~1 month
    
    # Expected interval (median days between transactions)
    df['expected_interval'] = df.groupby(series_cols)['days_since_last'].transform('median')
    df['interval_deviation'] = (df['days_since_last'] - df['expected_interval']).abs()
    
    # FREQUENCY ANOMALY: Multiple transactions in same month for same series
    df['year_month'] = df['date_post'].dt.to_period('M').astype(str)
    df['tx_count_this_month'] = df.groupby(series_cols + ['year_month']).cumcount() + 1
    df['is_duplicate_month'] = (df['tx_count_this_month'] > 1).astype(int)
    
    # ===========================================
    # 3. REFERENCE CONSISTENCY FEATURES 
    #    (for PAYEE_ANOMALY, IBAN_MISMATCH, SUBTLE_PAYEE_MISMATCH)
    # ===========================================
    
    # For each ref_name, what's the "expected" IBAN?
    # Count how often each (ref_name, ref_iban) combo appears
    iban_counts = df.groupby(['ref_name', 'ref_iban']).size().reset_index(name='iban_combo_count')
    df = df.merge(iban_counts, on=['ref_name', 'ref_iban'], how='left')
    
    # Most common IBAN per ref_name
    most_common_iban = df.groupby('ref_name')['iban_combo_count'].transform('max')
    df['is_common_iban'] = (df['iban_combo_count'] == most_common_iban).astype(int)
    
    # Same for SWIFT
    swift_counts = df.groupby(['ref_name', 'ref_swift']).size().reset_index(name='swift_combo_count')
    df = df.merge(swift_counts, on=['ref_name', 'ref_swift'], how='left')
    most_common_swift = df.groupby('ref_name')['swift_combo_count'].transform('max')
    df['is_common_swift'] = (df['swift_combo_count'] == most_common_swift).astype(int)
    
    # Combined: Does this transaction use the "usual" banking details?
    df['uses_usual_banking'] = (df['is_common_iban'] & df['is_common_swift']).astype(int)
    
    # How many unique IBANs has this ref_name used?
    df['ref_name_iban_diversity'] = df.groupby('ref_name')['ref_iban'].transform('nunique')
    
    # Is this a new IBAN for this ref_name? (first time seen)
    df['iban_first_seen'] = df.groupby(['ref_name', 'ref_iban']).cumcount()
    df['is_new_iban'] = (df['iban_first_seen'] == 0).astype(int)
    
    # ===========================================
    # 4. CHANNEL/METHOD FEATURES (for CHANNEL_ANOMALY)
    # ===========================================
    
    # Most common payment method per series
    method_counts = df.groupby(series_cols + ['pay_method']).size().reset_index(name='method_count')
    df = df.merge(method_counts, on=series_cols + ['pay_method'], how='left')
    max_method = df.groupby(series_cols)['method_count'].transform('max')
    df['is_usual_method'] = (df['method_count'] == max_method).astype(int)
    
    # Most common channel per series
    channel_counts = df.groupby(series_cols + ['channel']).size().reset_index(name='channel_count')
    df = df.merge(channel_counts, on=series_cols + ['channel'], how='left')
    max_channel = df.groupby(series_cols)['channel_count'].transform('max')
    df['is_usual_channel'] = (df['channel_count'] == max_channel).astype(int)
    
    # Combined: usual method AND channel?
    df['uses_usual_method_channel'] = (df['is_usual_method'] & df['is_usual_channel']).astype(int)
    
    # Encode categorical features
    cat_cols = ['pay_method', 'channel', 'currency', 'trns_type']
    for col in cat_cols:
        if is_training:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen categories
            df[f'{col}_encoded'] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # ===========================================
    # 5. SERIES-LEVEL FEATURES
    # ===========================================
    
    # Transaction count in series so far
    df['tx_count_in_series'] = df.groupby(series_cols).cumcount() + 1
    df['is_first_tx'] = (df['tx_count_in_series'] == 1).astype(int)
    
    # Total transactions per ref_name (popularity)
    df['ref_name_total_tx'] = df.groupby('ref_name')['ref_name'].transform('count')
    
    # Is this a "regular" series (many transactions) or one-off?
    df['is_regular_series'] = (df['ref_name_total_tx'] > 5).astype(int)
    
    # ===========================================
    # 6. ACCOUNT-LEVEL FEATURES
    # ===========================================
    
    # How many different ref_names does this account have?
    df['account_ref_diversity'] = df.groupby(account_col)['ref_name'].transform('nunique')
    
    # Amount relative to account's typical transaction
    df['account_amount_mean'] = df.groupby(account_col)['amount'].transform('mean')
    df['amount_vs_account_mean'] = df['amount'] / (df['account_amount_mean'] + 1)
    
    return df, encoders


def select_features(df):
    """Select final feature columns and drop intermediate columns."""
    
    feature_cols = [
        # ID columns (kept for compatibility, may be dropped by downstream code)
        'bank_account_uuid',
        
        # Amount features
        'amount',
        'amount_zscore',
        'amount_ratio_to_mean',
        'amount_percentile',
        
        # Timing features
        'day_of_month',
        'day_of_week',
        'dom_deviation',
        'days_since_last',
        'interval_deviation',
        'is_duplicate_month',
        
        # Reference consistency
        'is_common_iban',
        'is_common_swift',
        'uses_usual_banking',
        'ref_name_iban_diversity',
        'is_new_iban',
        
        # Channel/Method
        'is_usual_method',
        'is_usual_channel',
        'uses_usual_method_channel',
        'pay_method_encoded',
        'channel_encoded',
        'currency_encoded',
        'trns_type_encoded',
        
        # Series features
        'tx_count_in_series',
        'is_first_tx',
        'is_regular_series',
        
        # Account features
        'account_ref_diversity',
        'amount_vs_account_mean',
    ]
    
    # Keep only existing columns
    available_cols = [c for c in feature_cols if c in df.columns]
    
    return df[available_cols]


def prepare_data_supervised():
    """
    Main data preparation pipeline.
    
    Key differences from original:
    1. Time-based split (not random) to simulate production
    2. Features designed for specific anomaly types
    3. Proper handling of train/test feature computation
    """
    df_original = load_business_dataset()
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    df_original.to_csv(output_dir / "original_data.csv", index=True)
    
    df = df_original.copy()
    df['original_index'] = df.index
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    
    # Sort by date for time-based split
    df = df.sort_values('date_post').reset_index(drop=True)
    
    # Target
    target_col = "anomaly_description"
    df['target'] = df[target_col].notna().astype(int)
    
    # ===========================================
    # TIME-BASED SPLIT (more realistic)
    # ===========================================
    # Use first 60% of time period for training
    split_date = df['date_post'].quantile(0.5)
    
    # SEMI-SUPERVISED: Training contains ONLY normal data
    # Test contains both normal and anomalies (from the later time period)
    train_mask = (df['date_post'] <= split_date) & (df['target'] == 0)
    test_mask = df['date_post'] > split_date
    
    df_train_raw = df[train_mask].copy()
    df_test_raw = df[test_mask].copy()
    
    # Also add anomalies from training period to test set
    # (so we don't lose them entirely)
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
    
    # Train: Features computed only on training data
    df_train, encoders = create_features(df_train_raw, is_training=True)
    
    # Test: Features computed on ALL data (train + test) - this is correct!
    # We simulate having history available at prediction time
    df_full_raw = pd.concat([df_train_raw, df_test_raw], axis=0)
    df_full, _ = create_features(df_full_raw, is_training=False, encoders=encoders)
    
    # Extract test rows by original_index
    test_indices = df_test_raw['original_index'].values
    df_test = df_full[df_full['original_index'].isin(test_indices)].copy()
    
    # ===========================================
    # PREPARE FINAL DATASETS
    # ===========================================
    
    y_train = df_train['target']
    y_test = df_test['target']
    
    # Store mappings
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
    X_train = select_features(df_train)
    X_test = select_features(df_test)
    
    # Save feature columns
    feature_columns = X_train.columns.tolist()
    with open(output_dir / "feature_columns.json", 'w') as f:
        json.dump(feature_columns, f, indent=2)
    
    # Save encoders
    import pickle
    with open(output_dir / "encoders.pkl", 'wb') as f:
        pickle.dump(encoders, f)
    
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
    
    # Quick feature importance check
    print("\n=== Feature Value Ranges ===")
    for col in feature_columns[:10]:
        print(f"{col}: [{X_train[col].min():.2f}, {X_train[col].max():.2f}]")
    
    return X_train, X_test, y_train, y_test, train_mapping, test_mapping, feature_columns


if __name__ == "__main__":
    main()