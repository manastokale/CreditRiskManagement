"""
Account Segmentation — Rule-based labeling + Random Forest classifier.

Assigns accounts to four risk buckets:
1. Eligible - No Risk
2. Eligible - With Risk
3. No Increase Needed
4. Non-Performing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Segment labels ───────────────────────────────────────────────────────────
SEGMENTS = {
    0: 'Non-Performing',
    1: 'No Increase Needed',
    2: 'Eligible - With Risk',
    3: 'Eligible - No Risk',
}

SEGMENT_COLORS = {
    'Non-Performing': '#dc3545',
    'No Increase Needed': '#6c757d',
    'Eligible - With Risk': '#ffc107',
    'Eligible - No Risk': '#28a745',
}


# ═════════════════════════════════════════════════════════════════════════════
# RULE-BASED LABELING
# ═════════════════════════════════════════════════════════════════════════════

def assign_rule_based_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign segment labels based on explicit business rules.
    Check Non-Performing FIRST — those override everything.

    Thresholds calibrated against typical credit portfolio distributions.
    """
    df = df.copy()

    # Ensure numeric
    for col in ['ca_current_utilz', 'cu_nbr_days_dlq', 'ca_nsf_count_lst_12_months',
                'has_fraud_case', 'delinquent_cycle_count', 'max_delinquency_level',
                'sales_trend_slope', 'avg_monthly_sales_6m', 'utilization_ratio',
                'cu_cur_balance', 'ca_mob']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    def classify_account(row):
        # ── 1. Non-Performing (check first, overrides all) ───────────
        if row.get('cu_nbr_days_dlq', 0) > 60:
            return 0
        if row.get('max_delinquency_level', 0) >= 4:
            return 0
        if row.get('has_fraud_case', 0) == 1:
            return 0
        if row.get('delinquent_cycle_count', 0) >= 6:
            return 0
        # Inactive: no spending in 6 months AND zero balance
        if (row.get('avg_monthly_sales_6m', 0) == 0 and
            row.get('cu_cur_balance', 0) <= 0 and
            row.get('ca_mob', 0) > 6):
            return 0

        # ── 2. Eligible - With Risk ─────────────────────────────────
        has_risk = False
        if row.get('ca_current_utilz', 0) > 75:
            has_risk = True
        if row.get('ca_nsf_count_lst_12_months', 0) >= 2:
            has_risk = True
        if row.get('delinquent_cycle_count', 0) >= 2:
            has_risk = True
        if row.get('cu_nbr_days_dlq', 0) > 15:
            has_risk = True
        if row.get('max_delinquency_level', 0) >= 2:
            has_risk = True

        # Check if eligible (spending is growing or substantial)
        spending_active = (row.get('avg_monthly_sales_6m', 0) > 0 or
                          row.get('sales_trend_slope', 0) > 0)

        if has_risk and spending_active:
            return 2  # Eligible - With Risk

        # ── 3. No Increase Needed ───────────────────────────────────
        # Stable but not growing
        if (row.get('sales_trend_slope', 0) <= 0 and
            row.get('ca_current_utilz', 0) < 50 and
            row.get('avg_monthly_sales_6m', 0) > 0):
            return 1  # No Increase Needed

        if not spending_active:
            return 1  # Dormant-ish

        # ── 4. Eligible - No Risk (everything else positive) ────────
        if (row.get('ca_current_utilz', 0) <= 75 and
            row.get('cu_nbr_days_dlq', 0) == 0 and
            row.get('ca_nsf_count_lst_12_months', 0) == 0 and
            row.get('delinquent_cycle_count', 0) <= 1 and
            spending_active):
            return 3  # Eligible - No Risk

        return 1  # Default: No Increase Needed

    df['segment_label'] = df.apply(classify_account, axis=1)
    df['segment_name'] = df['segment_label'].map(SEGMENTS)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

CLASSIFICATION_FEATURES = [
    'cu_bhv_scr', 'cu_crd_bureau_scr', 'ca_current_utilz',
    'ca_avg_utilz_lst_3_mnths', 'ca_avg_utilz_lst_6_mnths',
    'cu_nbr_days_dlq', 'ca_nsf_count_lst_12_months',
    'ca_mob', 'ca_mnths_since_active',
    'avg_monthly_sales_6m', 'sales_trend_slope',
    'delinquent_cycle_count', 'max_delinquency_level',
    'utilization_ratio', 'otb_ratio',
    'cu_crd_line', 'cu_cur_balance',
]


def train_segmentation_classifier(df: pd.DataFrame,
                                    feature_cols: list = None,
                                    test_size: float = 0.25,
                                    save_path: Optional[str] = None) -> Tuple:
    """
    Train Random Forest on rule-based labels.
    Returns: (model, classification_report_str, confusion_matrix, test_df)
    """
    if feature_cols is None:
        feature_cols = CLASSIFICATION_FEATURES

    # Filter to available features
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Prepare data
    data = df.dropna(subset=feature_cols + ['segment_label']).copy()
    X = data[feature_cols].astype(float)
    y = data['segment_label'].astype(int)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train with balanced class weights
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    target_names = [SEGMENTS[i] for i in sorted(SEGMENTS.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n  Random Forest Segmentation (weighted F1: {f1:.4f})")
    print(report)

    if save_path:
        joblib.dump(model, save_path)
        print(f"  → Model saved to {save_path}")

    # Return test data with predictions for analysis
    test_df = X_test.copy()
    test_df['y_true'] = y_test.values
    test_df['y_pred'] = y_pred

    return model, report, cm, test_df, feature_cols


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONAL: K-MEANS CLUSTERING FEATURE
# ═════════════════════════════════════════════════════════════════════════════

def add_kmeans_cluster_feature(df: pd.DataFrame,
                                 feature_cols: list = None,
                                 n_clusters: int = 6) -> Tuple:
    """
    Run K-Means clustering and add cluster_id as an additional feature.
    Returns: (df_with_cluster, kmeans_model, scaler)
    """
    if feature_cols is None:
        feature_cols = CLASSIFICATION_FEATURES

    feature_cols = [c for c in feature_cols if c in df.columns]
    data = df[feature_cols].fillna(0).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['cluster_id'] = kmeans.fit_predict(X_scaled)

    return df, kmeans, scaler
