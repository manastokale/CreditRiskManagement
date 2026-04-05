"""
Fraud & Risk Detection — Isolation Forest anomaly detection + composite risk score.

Uses behavioral transaction features (fraud labels excluded from training)
to detect anomalous accounts, then combines with rule-based signals into
a 0-100 composite risk score.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ═════════════════════════════════════════════════════════════════════════════
# ISOLATION FOREST ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════

ANOMALY_FEATURES = [
    'total_spend', 'txn_count', 'avg_txn_amt', 'max_txn_amt',
    'total_foreign_amt', 'total_markup_fees', 'first_purchase_count'
]


def train_isolation_forest(anomaly_df: pd.DataFrame,
                            feature_cols: list = None,
                            contamination: float = None,
                            fraud_rate_df: pd.DataFrame = None,
                            save_path: Optional[str] = None,
                            scaler_path: Optional[str] = None) -> Tuple:
    """
    Fit Isolation Forest on behavioral features.
    has_fraud_case is EXCLUDED from training but used for post-hoc evaluation.

    Args:
        anomaly_df: Per-account aggregated transaction features
        contamination: Expected anomaly fraction. If None, calibrated from fraud_rate_df.
        fraud_rate_df: Customer base with has_fraud_case for calibration

    Returns: (model, scaler, predictions_df)
    """
    if feature_cols is None:
        feature_cols = ANOMALY_FEATURES

    feature_cols = [c for c in feature_cols if c in anomaly_df.columns]

    # Calibrate contamination from actual fraud rate
    if contamination is None and fraud_rate_df is not None:
        fraud_count = fraud_rate_df['has_fraud_case'].sum()
        total = len(fraud_rate_df)
        contamination = max(0.005, min(0.1, fraud_count / total * 2))
        print(f"  Calibrated contamination: {contamination:.4f} "
              f"(fraud rate: {fraud_count}/{total})")
    elif contamination is None:
        contamination = 0.02  # Default

    # Prepare features
    X = anomaly_df[feature_cols].fillna(0).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Predict: -1 = anomaly, 1 = normal
    raw_preds = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    result = anomaly_df.copy()
    result['anomaly_flag'] = (raw_preds == -1).astype(int)
    result['anomaly_score'] = anomaly_scores

    anomaly_count = result['anomaly_flag'].sum()
    print(f"  Detected {anomaly_count} anomalous accounts "
          f"({anomaly_count/len(result)*100:.1f}%)")

    # Save artifacts
    if save_path:
        joblib.dump(model, save_path)
        print(f"  → Isolation Forest saved to {save_path}")
    if scaler_path:
        joblib.dump(scaler, scaler_path)
        print(f"  → Scaler saved to {scaler_path}")

    return model, scaler, result


def evaluate_anomaly_detector(predictions_df: pd.DataFrame,
                                customer_base: pd.DataFrame) -> dict:
    """
    Evaluate Isolation Forest against known fraud cases (post-hoc).
    Returns precision/recall/F1 for fraud detection.
    """
    merged = predictions_df.merge(
        customer_base[['current_account_nbr', 'has_fraud_case']],
        on='current_account_nbr',
        how='left'
    )
    merged['has_fraud_case'] = merged['has_fraud_case'].fillna(0).astype(int)

    tp = ((merged['anomaly_flag'] == 1) & (merged['has_fraud_case'] == 1)).sum()
    fp = ((merged['anomaly_flag'] == 1) & (merged['has_fraud_case'] == 0)).sum()
    fn = ((merged['anomaly_flag'] == 0) & (merged['has_fraud_case'] == 1)).sum()
    tn = ((merged['anomaly_flag'] == 0) & (merged['has_fraud_case'] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }
    print(f"  Anomaly Detection vs Fraud Labels:")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"    Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# COMPOSITE RISK SCORE (0-100)
# ═════════════════════════════════════════════════════════════════════════════

# Risk score component weights — interpretable and documented
RISK_WEIGHTS = {
    'delinquency_depth':   0.30,  # cu_nbr_days_dlq normalized
    'nsf_frequency':       0.15,  # ca_nsf_count_lst_12_months
    'utilization_level':   0.25,  # ca_current_utilz
    'anomaly_flag':        0.20,  # From Isolation Forest
    'payment_history':     0.10,  # max_delinquency_level
}


def compute_risk_score(customer_base: pd.DataFrame,
                        anomaly_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Isolation Forest anomaly flag with rule-based components
    into a single 0-100 composite risk score.

    Components:
    - Delinquency depth (30%): Days delinquent, normalized 0-100
    - NSF frequency (15%): NSF count in last 12 months
    - Utilization level (25%): Current utilization percentage
    - Anomaly flag (20%): Binary from Isolation Forest
    - Payment history (10%): Max delinquency level from parsed history

    All weights are interpretable and documented for regulatory compliance.
    """
    df = customer_base.copy()

    # Merge anomaly predictions
    if 'anomaly_flag' not in df.columns:
        df = df.merge(
            anomaly_predictions[['current_account_nbr', 'anomaly_flag', 'anomaly_score']],
            on='current_account_nbr',
            how='left'
        )
    df['anomaly_flag'] = df['anomaly_flag'].fillna(0)

    # ── Normalize components to 0-100 ────────────────────────────────────

    # Delinquency depth: 0 days = 0, 90+ days = 100
    df['risk_delinquency'] = np.clip(
        pd.to_numeric(df['cu_nbr_days_dlq'], errors='coerce').fillna(0) / 90 * 100,
        0, 100
    )

    # NSF frequency: 0 = 0, 5+ = 100
    df['risk_nsf'] = np.clip(
        pd.to_numeric(df['ca_nsf_count_lst_12_months'], errors='coerce').fillna(0) / 5 * 100,
        0, 100
    )

    # Utilization: already 0-100 (percentage)
    df['risk_utilization'] = np.clip(
        pd.to_numeric(df['ca_current_utilz'], errors='coerce').fillna(0),
        0, 100
    )

    # Anomaly flag: 0 or 100
    df['risk_anomaly'] = df['anomaly_flag'].astype(float) * 100

    # Payment history: max delinquency 0-7, normalize to 0-100
    df['risk_payment_hist'] = np.clip(
        pd.to_numeric(df.get('max_delinquency_level', 0), errors='coerce').fillna(0) / 7 * 100,
        0, 100
    )

    # ── Weighted composite ───────────────────────────────────────────────
    df['risk_score'] = (
        RISK_WEIGHTS['delinquency_depth']  * df['risk_delinquency'] +
        RISK_WEIGHTS['nsf_frequency']      * df['risk_nsf'] +
        RISK_WEIGHTS['utilization_level']  * df['risk_utilization'] +
        RISK_WEIGHTS['anomaly_flag']       * df['risk_anomaly'] +
        RISK_WEIGHTS['payment_history']    * df['risk_payment_hist']
    ).round(2)

    df['risk_score'] = np.clip(df['risk_score'], 0, 100)

    print(f"  Risk Score Distribution:")
    print(f"    Mean: {df['risk_score'].mean():.1f}")
    print(f"    Median: {df['risk_score'].median():.1f}")
    print(f"    Std: {df['risk_score'].std():.1f}")
    print(f"    Min: {df['risk_score'].min():.1f}")
    print(f"    Max: {df['risk_score'].max():.1f}")

    return df
