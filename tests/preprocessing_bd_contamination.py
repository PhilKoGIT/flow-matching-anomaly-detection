# preprocessing_bd_contamination.py
"""
Preprocessing for business transaction dataset for contamination studies.
Outputs data in same format as load_adbench_npz() for easy integration.

Returns: X_train_normal, X_train_abnormal, X_test, y_test (all numpy arrays, scaled)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


def load_business_dataset():
    """Load the raw business transactions dataset."""
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
    """
    df = df.copy()
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values(['bank_account_uuid', 'ref_name', 'date_post']).reset_index(drop=True)
    
    if encoding_maps is None:
        encoding_maps = {}
    
    series_cols = ['bank_account_uuid', 'ref_name']
    
    # ===========================================
    # 1. AMOUNT FEATURES
    # ===========================================
    df['amount_rolling_mean'] = df.groupby(series_cols)['amount'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    df['amount_rolling_std'] = df.groupby(series_cols)['amount'].transform(
        lambda x: x.rolling(10, min_periods=1).std().fillna(0)
    )
    df['amount_zscore'] = (df['amount'] - df['amount_rolling_mean']) / (df['amount_rolling_std'] + 1)
    df['amount_ratio_to_mean'] = df['amount'] / (df['amount_rolling_mean'] + 1)
    
    # ===========================================
    # 2. TIMING FEATURES
    # ===========================================
    df['day_of_month'] = df['date_post'].dt.day
    df['day_of_week'] = df['date_post'].dt.dayofweek
    df['expected_dom'] = df.groupby(series_cols)['day_of_month'].transform('median')
    df['dom_deviation'] = (df['day_of_month'] - df['expected_dom']).abs()
    df['days_since_last'] = df.groupby(series_cols)['date_post'].diff().dt.days
    df['days_since_last'] = df['days_since_last'].fillna(30)
    
    df['year_month'] = df['date_post'].dt.to_period('M').astype(str)
    df['tx_count_this_month'] = df.groupby(series_cols + ['year_month']).cumcount() + 1
    df['is_duplicate_month'] = (df['tx_count_this_month'] > 1).astype(int)
    
    # ===========================================
    # 3. HIGH CARDINALITY - Frequency Encoding
    # ===========================================
    high_card_cols = ['ref_name', 'ref_iban', 'ref_swift', 'bank_account_uuid']
    
    for col in high_card_cols:
        if is_training:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            encoding_maps[f'{col}_freq'] = freq_map
        else:
            freq_map = encoding_maps.get(f'{col}_freq', {})
        df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
    
    # ===========================================
    # 4. LOW CARDINALITY - One-Hot Encoding
    # ===========================================
    low_card_cols = ['pay_method', 'channel', 'currency', 'trns_type']
    
    for col in low_card_cols:
        if is_training:
            unique_vals = df[col].unique().tolist()
            encoding_maps[f'{col}_categories'] = unique_vals
        else:
            unique_vals = encoding_maps.get(f'{col}_categories', [])
        
        for val in unique_vals:
            df[f'{col}_{val}'] = (df[col] == val).astype(int)
    
    # ===========================================
    # 5. REFERENCE CONSISTENCY FEATURES
    # ===========================================
    if is_training:
        most_common_iban = df.groupby('ref_name')['ref_iban'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['ref_name_usual_iban'] = most_common_iban
        
        most_common_swift = df.groupby('ref_name')['ref_swift'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['ref_name_usual_swift'] = most_common_swift
        
        most_common_method = df.groupby(series_cols)['pay_method'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['series_usual_method'] = most_common_method
        
        most_common_channel = df.groupby(series_cols)['channel'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).to_dict()
        encoding_maps['series_usual_channel'] = most_common_channel
    
    usual_iban_map = encoding_maps.get('ref_name_usual_iban', {})
    usual_swift_map = encoding_maps.get('ref_name_usual_swift', {})
    usual_method_map = encoding_maps.get('series_usual_method', {})
    usual_channel_map = encoding_maps.get('series_usual_channel', {})
    
    df['usual_iban'] = df['ref_name'].map(usual_iban_map)
    df['is_usual_iban'] = (df['ref_iban'] == df['usual_iban']).astype(int)
    df.loc[df['usual_iban'].isna(), 'is_usual_iban'] = 0
    
    df['usual_swift'] = df['ref_name'].map(usual_swift_map)
    df['is_usual_swift'] = (df['ref_swift'] == df['usual_swift']).astype(int)
    df.loc[df['usual_swift'].isna(), 'is_usual_swift'] = 0
    
    df['series_key'] = list(zip(df['bank_account_uuid'], df['ref_name']))
    
    df['usual_method'] = df['series_key'].map(usual_method_map)
    df['is_usual_method'] = (df['pay_method'] == df['usual_method']).astype(int)
    df.loc[df['usual_method'].isna(), 'is_usual_method'] = 0
    
    df['usual_channel'] = df['series_key'].map(usual_channel_map)
    df['is_usual_channel'] = (df['channel'] == df['usual_channel']).astype(int)
    df.loc[df['usual_channel'].isna(), 'is_usual_channel'] = 0
    
    df['uses_usual_banking'] = (df['is_usual_iban'] & df['is_usual_swift']).astype(int)
    df['uses_usual_method_channel'] = (df['is_usual_method'] & df['is_usual_channel']).astype(int)
    
    # ===========================================
    # 6. SERIES-LEVEL FEATURES
    # ===========================================
    df['tx_count_in_series'] = df.groupby(series_cols).cumcount() + 1
    df['is_first_tx'] = (df['tx_count_in_series'] == 1).astype(int)
    df['ref_name_tx_count'] = df.groupby('ref_name')['ref_name'].transform('count')
    df['is_regular_series'] = (df['ref_name_tx_count'] > 5).astype(int)
    
    return df, encoding_maps


def select_features(df, encoding_maps):
    """Select final numeric feature columns."""
    
    feature_cols = [
        'amount', 'amount_zscore', 'amount_ratio_to_mean',
        'day_of_month', 'day_of_week', 'dom_deviation',
        'days_since_last', 'is_duplicate_month',
        'ref_name_freq', 'ref_iban_freq', 'ref_swift_freq', 'bank_account_uuid_freq',
        'is_usual_iban', 'is_usual_swift', 'uses_usual_banking',
        'is_usual_method', 'is_usual_channel', 'uses_usual_method_channel',
        'tx_count_in_series', 'is_first_tx', 'is_regular_series',
    ]
    
    # Add one-hot columns
    low_card_cols = ['pay_method', 'channel', 'currency', 'trns_type']
    for col in low_card_cols:
        categories = encoding_maps.get(f'{col}_categories', [])
        for cat in categories:
            col_name = f'{col}_{cat}'
            if col_name in df.columns:
                feature_cols.append(col_name)
    
    available_cols = [c for c in feature_cols if c in df.columns]
    return df[available_cols]


def load_business_dataset_for_contamination(test_size=0.5, random_state=42):
    """
    Load and preprocess business dataset for contamination studies.
    
    Returns same format as load_adbench_npz():
        X_train_normal: Normal training samples (scaled)
        X_train_abnormal: Abnormal samples for contamination (scaled)  
        X_test: Test set with normal + abnormal (scaled)
        y_test: Labels for test set
        
    The split is:
    1. All data → Features engineered on ALL data (to have consistent encodings)
    2. Normal data → 50/50 split into train_normal / test_normal
    3. Abnormal data → 50/50 split into train_abnormal (for contamination) / test_abnormal
    4. Test = test_normal + test_abnormal
    """
    
    # Load raw data
    df_original = load_business_dataset()
    df = df_original.copy()
    
    # Parse date
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    
    # Create target
    target_col = "anomaly_description"
    df['target'] = df[target_col].notna().astype(int)
    
    # ===========================================
    # FEATURE ENGINEERING ON ALL DATA
    # (to ensure consistent encodings)
    # ===========================================
    df_features, encoding_maps = create_features(df, is_training=True)
    
    # Select features
    X_df = select_features(df_features, encoding_maps)
    y = df_features['target'].values
    
    # Convert to numpy
    X = X_df.to_numpy(dtype=float)
    
    # ===========================================
    # SPLIT NORMAL / ABNORMAL
    # ===========================================
    X_normal = X[y == 0]
    X_abnormal = X[y == 1]
    y_normal = y[y == 0]
    y_abnormal = y[y == 1]
    
    # 50/50 split for normal data
    X_train_normal, X_test_normal = train_test_split(
        X_normal, test_size=test_size, random_state=random_state
    )
    
    # 50/50 split for abnormal data
    X_train_abnormal, X_test_abnormal = train_test_split(
        X_abnormal, test_size=test_size, random_state=random_state
    )
    
    # Test set = test_normal + test_abnormal
    X_test = np.vstack([X_test_normal, X_test_abnormal])
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_abnormal))])
    
    # ===========================================
    # STANDARDIZATION
    # Fit scaler on train_normal only (clean data)
    # ===========================================
    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(X_train_normal)
    X_train_abnormal = scaler.transform(X_train_abnormal)
    X_test = scaler.transform(X_test)
    
    # Print info
    print(f"\nBusiness Dataset loaded for contamination study:")
    print(f"  X_train_normal: {X_train_normal.shape}")
    print(f"  X_train_abnormal (for contamination): {X_train_abnormal.shape}")
    print(f"  X_test: {X_test.shape} (Normal: {len(X_test_normal)}, Anomalies: {len(X_test_abnormal)})")
    print(f"  Features: {X_df.columns.tolist()}")
    
    return X_train_normal, X_train_abnormal, X_test, y_test


# For quick testing
if __name__ == "__main__":
    X_train_normal, X_train_abnormal, X_test, y_test = load_business_dataset_for_contamination()
    
    print(f"\nMax contamination ratio possible: {len(X_train_abnormal) / (len(X_train_abnormal) + len(X_train_normal)):.4f}")
