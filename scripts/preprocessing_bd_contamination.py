"""
Preprocessing Pipeline für das Business Dataset (Contamination Studies).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Set
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


FEATURE_COLS = [
    'amount_zscore_series', 'amount_ratio_to_mean',
    'day_deviation_from_usual',
    'iban_mismatch', 'swift_mismatch', 'known_name_new_iban',
    'method_mismatch', 'channel_mismatch', 'is_card_mobile',
    'tx_count_this_month_so_far',
    'is_new_series', 'is_new_ref_name', 'is_new_iban',
    'series_tx_count_before_log', 'ref_name_count_before_log', 'iban_count_before_log',
    'day_of_month', 'day_of_week', 'month', 'days_since_last_in_series',
    'amount',
]

def compute_features_no_leakage(
    df: pd.DataFrame,
    known_series_stats: Dict = None,
    known_ref_names: Set = None,
    known_ibans: Set = None,
    known_name_iban_pairs: Set = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict, Set, Set, Set]:

    df = df.copy()
    
    if df['date_post'].dtype == 'object' or df['date_post'].dtype == 'int64':
        df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    
    df = df.sort_values('date_post').reset_index(drop=True)
    
    df['series_id'] = df['bank_account_uuid'] + '_' + df['ref_name']
    df['year_month'] = df['date_post'].dt.to_period('M')
    
    if fit:
        series_stats = {}
        seen_ref_names = set()
        seen_ibans = set()
        seen_name_iban_pairs = set()
    else:
        series_stats = {}
        for k, v in (known_series_stats or {}).items():
            series_stats[k] = {
                'amounts': list(v.get('amounts', [])),
                'ibans': list(v.get('ibans', [])),
                'swifts': list(v.get('swifts', [])),
                'days': list(v.get('days', [])),
                'methods': list(v.get('methods', [])),
                'channels': list(v.get('channels', [])),
                'year_months': list(v.get('year_months', []))
            }
        seen_ref_names = set(known_ref_names) if known_ref_names else set()
        seen_ibans = set(known_ibans) if known_ibans else set()
        seen_name_iban_pairs = set(known_name_iban_pairs) if known_name_iban_pairs else set()
    
    n = len(df)
    features = {
        'amount_zscore_series': np.zeros(n),
        'amount_ratio_to_mean': np.ones(n),
        'day_deviation_from_usual': np.zeros(n),
        'iban_mismatch': np.zeros(n),
        'swift_mismatch': np.zeros(n),
        'method_mismatch': np.zeros(n),
        'channel_mismatch': np.zeros(n),
        'is_card_mobile': np.zeros(n),
        'tx_count_this_month_so_far': np.ones(n),
        'is_new_series': np.zeros(n),
        'is_new_ref_name': np.zeros(n),
        'is_new_iban': np.zeros(n),
        'known_name_new_iban': np.zeros(n),
        'series_tx_count_before': np.zeros(n),
        'ref_name_count_before': np.zeros(n),
        'iban_count_before': np.zeros(n),
        'day_of_month': np.zeros(n),
        'day_of_week': np.zeros(n),
        'month': np.zeros(n),
        'days_since_last_in_series': np.full(n, 30.0),
        'amount': np.zeros(n),
    }
    
    ref_name_counts = {}
    iban_counts = {}
    last_date_per_series = {}
    
    for idx, row in df.iterrows():
        series_id = row['series_id']
        ref_name = row['ref_name']
        ref_iban = row['ref_iban']
        ref_swift = row['ref_swift']
        pay_method = row['pay_method']
        channel = row['channel']
        amount = row['amount']
        tx_date = row['date_post']
        day = tx_date.day
        year_month = row['year_month']
        
        
        features['amount'][idx] = amount
        features['day_of_month'][idx] = day
        features['day_of_week'][idx] = tx_date.dayofweek
        features['month'][idx] = tx_date.month
        features['is_card_mobile'][idx] = 1 if (pay_method == 'CARD' and channel == 'MOBILE_APP') else 0
        
        features['is_new_series'][idx] = 0 if series_id in series_stats else 1
        features['is_new_ref_name'][idx] = 0 if ref_name in seen_ref_names else 1
        features['is_new_iban'][idx] = 0 if ref_iban in seen_ibans else 1
        
        name_iban_pair = (ref_name, ref_iban)
        if ref_name in seen_ref_names and name_iban_pair not in seen_name_iban_pairs:
            features['known_name_new_iban'][idx] = 1
        
        features['ref_name_count_before'][idx] = ref_name_counts.get(ref_name, 0)
        features['iban_count_before'][idx] = iban_counts.get(ref_iban, 0)
        
        if series_id in series_stats:
            stats = series_stats[series_id]
            features['series_tx_count_before'][idx] = len(stats['amounts'])
            
            if len(stats['amounts']) > 0:
                past_mean = np.mean(stats['amounts'])
                past_std = np.std(stats['amounts']) if len(stats['amounts']) > 1 else past_mean * 0.1
                if past_std > 0:
                    features['amount_zscore_series'][idx] = (amount - past_mean) / past_std
                features['amount_ratio_to_mean'][idx] = amount / past_mean if past_mean > 0 else 1.0
            
            if len(stats['days']) > 0:
                usual_day = np.median(stats['days'])
                features['day_deviation_from_usual'][idx] = abs(day - usual_day)
            
            if len(stats['ibans']) > 0:
                most_common = Counter(stats['ibans']).most_common(1)[0][0]
                features['iban_mismatch'][idx] = 0 if ref_iban == most_common else 1
            
            if len(stats['swifts']) > 0:
                most_common = Counter(stats['swifts']).most_common(1)[0][0]
                features['swift_mismatch'][idx] = 0 if ref_swift == most_common else 1
            
            if len(stats['methods']) > 0:
                most_common = Counter(stats['methods']).most_common(1)[0][0]
                features['method_mismatch'][idx] = 0 if pay_method == most_common else 1
            
            if len(stats['channels']) > 0:
                most_common = Counter(stats['channels']).most_common(1)[0][0]
                features['channel_mismatch'][idx] = 0 if channel == most_common else 1
            
            features['tx_count_this_month_so_far'][idx] = sum(
                1 for ym in stats['year_months'] if ym == year_month
            ) + 1
            
            if series_id in last_date_per_series:
                days_diff = (tx_date - last_date_per_series[series_id]).days
                features['days_since_last_in_series'][idx] = days_diff
                
        if series_id not in series_stats:
            series_stats[series_id] = {
                'amounts': [], 'ibans': [], 'swifts': [],
                'days': [], 'methods': [], 'channels': [], 'year_months': []
            }
        
        series_stats[series_id]['amounts'].append(amount)
        series_stats[series_id]['ibans'].append(ref_iban)
        series_stats[series_id]['swifts'].append(ref_swift)
        series_stats[series_id]['days'].append(day)
        series_stats[series_id]['methods'].append(pay_method)
        series_stats[series_id]['channels'].append(channel)
        series_stats[series_id]['year_months'].append(year_month)
        
        ref_name_counts[ref_name] = ref_name_counts.get(ref_name, 0) + 1
        iban_counts[ref_iban] = iban_counts.get(ref_iban, 0) + 1
        
        seen_ref_names.add(ref_name)
        seen_ibans.add(ref_iban)
        seen_name_iban_pairs.add(name_iban_pair)
        last_date_per_series[series_id] = tx_date
    
    for feat_name, feat_values in features.items():
        df[feat_name] = feat_values
    
    df['amount_zscore_series'] = df['amount_zscore_series'].clip(-10, 10)
    df['amount_ratio_to_mean'] = df['amount_ratio_to_mean'].clip(0, 10)

    df['series_tx_count_before_log'] = np.log1p(df['series_tx_count_before'])
    df['ref_name_count_before_log'] = np.log1p(df['ref_name_count_before'])
    df['iban_count_before_log'] = np.log1p(df['iban_count_before'])
    
    return df, series_stats, seen_ref_names, seen_ibans, seen_name_iban_pairs



def load_business_dataset_for_contamination(
    test_size: float = 0.5,
    scale_on_all_train: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "data_contamination" / "business_dataset.csv"
    
    for alt_path in [
        base_dir / "output" / "business_dataset.csv",
        Path("output") / "business_dataset.csv",
        Path("data") / "business_dataset.csv"
    ]:
        if not file_path.exists():
            file_path = alt_path
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    df['date_post'] = pd.to_datetime(df['date_post'], format='%Y%m%d')
    df = df.sort_values('date_post').reset_index(drop=True)
    
    print(f"\nTotal transactions: {len(df)}")
    print(f"Date range: {df['date_post'].min()} to {df['date_post'].max()}")
    
    split_idx = int(len(df) * (1 - test_size))
    split_date = df.iloc[split_idx]['date_post']
    
    df_train = df[df['date_post'] <= split_date].copy()
    df_test = df[df['date_post'] > split_date].copy()
    
    print(f"\nChronological split at: {split_date}")
    print(f"Train: {len(df_train)} transactions (until {split_date})")
    print(f"Test: {len(df_test)} transactions (after {split_date})")
    output_path = base_dir / "df_test_raw.csv"
    df_test.to_csv(output_path, index=False)
    print(f"Test DataFrame saved to: {output_path}")
    
    print("\nComputing features on training data...")
    df_train_processed, series_stats, seen_refs, seen_ibans, seen_pairs = \
        compute_features_no_leakage(df_train, fit=True)
    
    print("Computing features on test data...")
    df_test_processed, _, _, _, _ = compute_features_no_leakage(
        df_test,
        known_series_stats=series_stats,
        known_ref_names=seen_refs,
        known_ibans=seen_ibans,
        known_name_iban_pairs=seen_pairs,
        fit=False
    )

    y_train = df_train_processed['anomaly_description'].notna().astype(int).values
    y_test = df_test_processed['anomaly_description'].notna().astype(int).values
    
    train_normal_mask = y_train == 0
    train_abnormal_mask = y_train == 1


    X_train_all = df_train_processed[FEATURE_COLS].values.astype(float)
    X_train_normal = X_train_all[train_normal_mask]
    X_train_abnormal = X_train_all[train_abnormal_mask]
    X_test = df_test_processed[FEATURE_COLS].values.astype(float)
    
    print(f"\nTrain split:")
    print(f"  Normal: {len(X_train_normal)}")
    print(f"  Abnormal: {len(X_train_abnormal)}")
    print(f"Test: {len(X_test)} (Normal: {(y_test==0).sum()}, Abnormal: {(y_test==1).sum()})")
    
    X_train_normal = np.nan_to_num(X_train_normal, nan=0.0)
    X_train_abnormal = np.nan_to_num(X_train_abnormal, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print("\nScaling data...")
    scaler = StandardScaler()
    
    if scale_on_all_train:
        scaler.fit(X_train_all)
        print("  Scaler fitted on ALL training data (normal + abnormal)")
    else:
        scaler.fit(X_train_normal)
        print("  Scaler fitted on NORMAL training data only")
    
    X_train_normal = scaler.transform(X_train_normal)
    X_train_abnormal = scaler.transform(X_train_abnormal) if len(X_train_abnormal) > 0 else X_train_abnormal
    X_test = scaler.transform(X_test)
    
    print(f"\n{'='*60}")
    print("FINAL SHAPES")
    print('='*60)
    print(f"X_train_normal:  {X_train_normal.shape}")
    print(f"X_train_abnormal: {X_train_abnormal.shape}")
    print(f"X_test:          {X_test.shape}")
    print(f"y_test:          {y_test.shape} (Anomaly rate: {y_test.mean()*100:.2f}%)")
    
    print(f"\n{'='*60}")
    print("ANOMALY TYPES IN DATASET")
    print('='*60)
    
    all_anomalies = df[df['anomaly_description'].notna()]['anomaly_description']
    anomaly_types = all_anomalies.apply(lambda x: x.split(':')[0]).value_counts()
    
    train_anomalies = df_train_processed[df_train_processed['anomaly_description'].notna()]['anomaly_description']
    test_anomalies = df_test_processed[df_test_processed['anomaly_description'].notna()]['anomaly_description']
    
    print("\nTotal:")
    print(anomaly_types)
    print(f"\nIn Train: {len(train_anomalies)}")
    print(f"In Test: {len(test_anomalies)}")



    print(f"\n{'='*60}")
    print("ANOMALIES IN TEST SET")
    print('='*60)

    test_anomalies_df = df_test_processed[df_test_processed['anomaly_description'].notna()]

    for idx, row in test_anomalies_df.iterrows():
        print(f"\nDate: {row['date_post']}")
        print(f"Amount: {row['amount']}")
        print(f"Ref Name: {row['ref_name']}")
        print(f"Anomaly: {row['anomaly_description']}")
        print("-" * 40)
        
        return X_train_normal, X_train_abnormal, X_test, y_test

def create_contaminated_training_set(
    X_train_normal: np.ndarray,
    X_train_abnormal: np.ndarray,
    contamination_ratio: float
) -> np.ndarray:

    if contamination_ratio <= 0:
        return X_train_normal.copy()
    
    n_normal = len(X_train_normal)
    # formula: n_abnormal / (n_normal + n_abnormal) = ratio
    # => n_abnormal = ratio * n_normal / (1 - ratio)
    n_abnormal_needed = int(contamination_ratio * n_normal / (1 - contamination_ratio))
    n_abnormal_available = len(X_train_abnormal)
    n_abnormal_to_add = min(n_abnormal_needed, n_abnormal_available)
    
    if n_abnormal_to_add == 0:
        print(f"Warning: No abnormal samples available for contamination")
        return X_train_normal.copy()
    
    X_train_contaminated = np.vstack([
        X_train_normal,
        X_train_abnormal[:n_abnormal_to_add]
    ])
    
    actual_ratio = n_abnormal_to_add / len(X_train_contaminated)
    print(f"Contamination: Added {n_abnormal_to_add} abnormal samples")
    print(f"  Requested ratio: {contamination_ratio*100:.1f}%")
    print(f"  Actual ratio: {actual_ratio*100:.2f}%")
    print(f"  Final training set size: {len(X_train_contaminated)}")
    
    return X_train_contaminated


if __name__ == "__main__":
    print("Testing preprocessing pipeline for Business Dataset (Contamination)...")
    print()