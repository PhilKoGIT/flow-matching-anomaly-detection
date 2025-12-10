"""
Preprocessing Pipeline für Business Dataset - UNSUPERVISED
Kompatibel mit dem Evaluator (load_dataset Funktion)

Chronologischer Split: 80% Train / 20% Test
Train enthält NUR normale Transaktionen (Anomalien werden entfernt)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List
import json


# =============================================================================
# KONFIGURATION
# =============================================================================

HIGH_CARD_COLS = ['ref_name', 'ref_iban', 'ref_swift', 'paym_note']
HASH_DIM = 16

LOW_CARD_COLS = ['currency', 'trns_type', 'pay_method', 'channel', 'ref_bank']

GROUP_COLS = ['bank_account_uuid', 'ref_name']


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def hash_high_cardinality(df: pd.DataFrame, cols: List[str], n_features: int, 
                          hashers: dict = None) -> Tuple[np.ndarray, dict]:
    """Hasht High-Cardinality Spalten."""
    hashed_arrays = []
    if hashers is None:
        hashers = {}
        fit = True
    else:
        fit = False
    
    for col in cols:
        col_data = df[col].fillna('__MISSING__').astype(str)
        
        if fit:
            hasher = FeatureHasher(n_features=n_features, input_type='string')
            col_hashed = hasher.fit_transform([[val] for val in col_data]).toarray()
            hashers[col] = hasher
        else:
            col_hashed = hashers[col].transform([[val] for val in col_data]).toarray()
        
        hashed_arrays.append(col_hashed)
    
    return np.hstack(hashed_arrays), hashers


def onehot_encode(df: pd.DataFrame, cols: List[str], 
                  encoder: OneHotEncoder = None) -> Tuple[np.ndarray, OneHotEncoder]:
    """One-Hot Encoding für Low-Cardinality Spalten."""
    df_subset = df[cols].fillna('__MISSING__').astype(str)
    
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df_subset)
    else:
        encoded = encoder.transform(df_subset)
    
    return encoded, encoder


def create_time_series_features(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Erstellt Zeitreihen-Features."""
    df = df.copy()
    
    if df['date_post'].dtype == 'object' or df['date_post'].dtype == 'int64':
        df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    
    df = df.sort_values(['date_post'])
    
    # Rolling Features (mit shift um Data Leakage zu vermeiden)
    df['amount_rolling_mean_5'] = (
        df.groupby(group_cols)['amount']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df['amount_rolling_std_5'] = (
        df.groupby(group_cols)['amount']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
    )
    
    df['amount_rolling_mean_5'] = df['amount_rolling_mean_5'].fillna(df['amount'])
    df['amount_rolling_std_5'] = df['amount_rolling_std_5'].fillna(0)
    
    # Z-Score
    df['amount_zscore'] = (df['amount'] - df['amount_rolling_mean_5']) / (df['amount_rolling_std_5'] + 1e-8)
    df['amount_zscore'] = df['amount_zscore'].clip(-10, 10)
    
    # Zeit seit letzter Transaktion
    df['time_since_last_tx'] = df.groupby(group_cols)['date_post'].diff().dt.days
    global_median = df['time_since_last_tx'].median()
    df['time_since_last_tx'] = df['time_since_last_tx'].fillna(global_median if not pd.isna(global_median) else 30)
    
    # Datum Features
    df['day_of_month'] = df['date_post'].dt.day
    df['day_of_week'] = df['date_post'].dt.dayofweek
    
    # Transaktionszähler
    df['tx_count_in_group'] = df.groupby(group_cols).cumcount() + 1
    df['is_first_tx'] = (df['tx_count_in_group'] == 1).astype(int)
    
    return df


# =============================================================================
# GLOBALE ENCODER (werden beim ersten Aufruf gesetzt)
# =============================================================================

_hashers = None
_onehot_encoder = None
_scaler = None


def _transform_df(df: pd.DataFrame, fit: bool = False) -> np.ndarray:
    """Transformiert DataFrame zu NumPy Array."""
    global _hashers, _onehot_encoder, _scaler
    
    df = df.copy()
    df = create_time_series_features(df, GROUP_COLS)
    
    # Hashing
    if fit:
        hashed, _hashers = hash_high_cardinality(df, HIGH_CARD_COLS, HASH_DIM, None)
    else:
        hashed, _ = hash_high_cardinality(df, HIGH_CARD_COLS, HASH_DIM, _hashers)
    
    # One-Hot
    if fit:
        onehot, _onehot_encoder = onehot_encode(df, LOW_CARD_COLS, None)
    else:
        onehot, _ = onehot_encode(df, LOW_CARD_COLS, _onehot_encoder)
    
    # Numerische Features
    ts_cols = [
        'amount', 'amount_rolling_mean_5', 'amount_rolling_std_5',
        'amount_zscore', 'time_since_last_tx', 'day_of_month',
        'day_of_week', 'tx_count_in_group', 'is_first_tx'
    ]
    numeric = df[ts_cols].values
    
    # Kombinieren
    X = np.hstack([hashed, onehot, numeric])
    
    # Skalieren
    if fit:
        _scaler = StandardScaler()
        X = _scaler.fit_transform(X)
    else:
        X = _scaler.transform(X)
    
    return X


# =============================================================================
# HAUPTFUNKTION (kompatibel mit Evaluator)
# =============================================================================

def prepare_data_unsupervised():
    """
    Lädt und preprocessed das Business Dataset für unsupervised Anomaly Detection.
    
    Returns:
        X_train_df: DataFrame mit Features (für Kompatibilität)
        X_test_df: DataFrame mit Features
        y_train: Series mit Labels (alle 0 im unsupervised Fall)
        y_test: Series mit Labels
        train_mapping: DataFrame mit Index-Mapping
        test_mapping: DataFrame mit Index-Mapping
        feature_columns: Liste der Feature-Namen
    """
    global _hashers, _onehot_encoder, _scaler
    _hashers = None
    _onehot_encoder = None
    _scaler = None
    
    # Daten laden
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data" / "business_dataset.csv"
    
    if not file_path.exists():
        file_path = base_dir / "output" / "business_dataset.csv"
    if not file_path.exists():
        file_path = Path("output") / "business_dataset.csv"
    if not file_path.exists():
        file_path = Path("data") / "business_dataset.csv"
        
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Original Index
    df['original_index'] = df.index
    
    # Datum parsen und sortieren
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values('date_post').reset_index(drop=True)
    
    # Target
    y = df['anomaly_description'].notna().astype(int)
    
    # Chronologischer Split 80/20
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    y_train_full = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    # UNSUPERVISED: Nur normale Daten im Training
    normal_mask = y_train_full == 0
    df_train = df_train[normal_mask].reset_index(drop=True)
    y_train = y_train_full[normal_mask].reset_index(drop=True)
    
    print(f"Train (only normal): {len(df_train)} samples")
    print(f"Test: {len(df_test)} samples ({y_test.sum()} anomalies)")
    
    # Mappings erstellen
    train_mapping = pd.DataFrame({
        'transformed_index': range(len(df_train)),
        'original_index': df_train['original_index'].values
    })
    test_mapping = pd.DataFrame({
        'transformed_index': range(len(df_test)),
        'original_index': df_test['original_index'].values
    })
    
    # Transformieren
    X_train = _transform_df(df_train, fit=True)
    X_test = _transform_df(df_test, fit=False)
    
    # Feature Namen
    feature_names = []
    for col in HIGH_CARD_COLS:
        for i in range(HASH_DIM):
            feature_names.append(f'{col}_hash_{i}')
    feature_names.extend([f'onehot_{i}' for i in range(X_train.shape[1] - HASH_DIM * len(HIGH_CARD_COLS) - 9)])
    feature_names.extend([
        'amount', 'amount_rolling_mean_5', 'amount_rolling_std_5',
        'amount_zscore', 'time_since_last_tx', 'day_of_month',
        'day_of_week', 'tx_count_in_group', 'is_first_tx'
    ])
    
    # Als DataFrames zurückgeben (für Kompatibilität mit Evaluator)
    X_train_df = pd.DataFrame(X_train, columns=feature_names[:X_train.shape[1]])
    X_train_df['bank_account_uuid'] = df_train['bank_account_uuid'].values
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
    X_test_df['bank_account_uuid'] = df_test['bank_account_uuid'].values
    
    return X_train_df, X_test_df, y_train, y_test, train_mapping, test_mapping, feature_names


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    X_train_df, X_test_df, y_train, y_test, train_map, test_map, features = prepare_data_unsupervised()
    
    print("\n=== UNSUPERVISED PREPROCESSING COMPLETE ===")
    print(f"X_train shape: {X_train_df.shape}")
    print(f"X_test shape: {X_test_df.shape}")
    print(f"y_train: {len(y_train)} (all zeros)")
    print(f"y_test: {len(y_test)} ({y_test.sum()} anomalies = {y_test.mean()*100:.2f}%)")
