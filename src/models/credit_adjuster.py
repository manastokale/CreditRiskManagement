"""
Credit Line Adjustment — Rule-based + XGBoost regression.

Generates recommended credit line adjustments based on segment,
risk score, and predicted Q4 spending. Enforces hard business rules.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ═════════════════════════════════════════════════════════════════════════════
# RULE-BASED ADJUSTMENT
# ═════════════════════════════════════════════════════════════════════════════

ADJUSTMENT_RULES = {
    3: 0.20,   # Eligible - No Risk: +20%
    2: 0.10,   # Eligible - With Risk: +10%
    1: 0.00,   # No Increase Needed: flat
    0: -0.05,  # Non-Performing: -5% (reduce/freeze)
}


def apply_rule_based_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply segment-based credit line adjustment rules.

    Hard stop: cu_line_incr_excl_flag = 'Y' → never increase.
    """
    df = df.copy()

    # Ensure numeric
    df['cu_crd_line'] = pd.to_numeric(df['cu_crd_line'], errors='coerce').fillna(0)
    df['cu_cur_balance'] = pd.to_numeric(df['cu_cur_balance'], errors='coerce').fillna(0)

    # Map segment to adjustment percentage
    df['adjustment_pct'] = df['segment_label'].map(ADJUSTMENT_RULES).fillna(0)

    # ── Hard stop: exclusion flag ────────────────────────────────────────
    excl_mask = df['cu_line_incr_excl_flag'].astype(str).str.upper() == 'Y'
    # For excluded accounts: zero out any positive adjustment
    df.loc[excl_mask, 'adjustment_pct'] = np.minimum(
        df.loc[excl_mask, 'adjustment_pct'], 0
    )

    # Apply percentage
    df['rule_recommended_line'] = df['cu_crd_line'] * (1 + df['adjustment_pct'])

    # ── Clamp outputs ────────────────────────────────────────────────────
    # Minimum: current balance + 10% buffer (so we don't recommend below what they owe)
    min_line = df['cu_cur_balance'] * 1.10
    # Maximum: 1.5x current limit
    max_line = df['cu_crd_line'] * 1.50
    # Don't go below current limit in general
    max_line = np.maximum(max_line, df['cu_crd_line'])

    df['rule_recommended_line'] = np.clip(
        df['rule_recommended_line'], min_line, max_line
    )

    # For accounts with zero credit line, keep at zero
    df.loc[df['cu_crd_line'] == 0, 'rule_recommended_line'] = 0

    # ── FINAL exclusion enforcement (after clamping) ─────────────────────
    # Excluded accounts must NEVER receive an increase, even from clamping
    df.loc[excl_mask, 'rule_recommended_line'] = np.minimum(
        df.loc[excl_mask, 'rule_recommended_line'],
        df.loc[excl_mask, 'cu_crd_line']
    )

    df['rule_adjustment_delta'] = df['rule_recommended_line'] - df['cu_crd_line']

    return df


# ═════════════════════════════════════════════════════════════════════════════
# XGBOOST REGRESSION FOR SMOOTH ADJUSTMENTS
# ═════════════════════════════════════════════════════════════════════════════

CREDIT_FEATURES = [
    'cu_crd_line', 'cu_cur_balance', 'cu_otb',
    'ca_current_utilz', 'ca_avg_utilz_lst_3_mnths', 'ca_avg_utilz_lst_6_mnths',
    'cu_bhv_scr', 'cu_crd_bureau_scr',
    'cu_nbr_days_dlq', 'ca_nsf_count_lst_12_months',
    'ca_mob', 'ca_mnths_since_active', 'ca_mnths_since_cl_chng',
    'avg_monthly_sales_6m', 'sales_trend_slope',
    'delinquent_cycle_count', 'max_delinquency_level',
    'utilization_ratio', 'otb_ratio',
    'risk_score', 'segment_label',
]


def train_credit_adjuster(df: pd.DataFrame,
                            feature_cols: list = None,
                            target_col: str = 'rule_adjustment_delta',
                            predicted_spend_col: str = None,
                            test_size: float = 0.25,
                            save_path: Optional[str] = None) -> Tuple:
    """
    Train XGBoost regressor to predict credit line adjustment delta.
    Uses rule-based delta as target for smoother, continuous predictions.

    Returns: (model, predictions_on_test, metrics, feature_cols_used)
    """
    if feature_cols is None:
        feature_cols = CREDIT_FEATURES.copy()

    # Add predicted Q4 spend if available
    if predicted_spend_col and predicted_spend_col in df.columns:
        feature_cols.append(predicted_spend_col)

    feature_cols = [c for c in feature_cols if c in df.columns]

    # Prepare data
    data = df.dropna(subset=feature_cols + [target_col]).copy()
    X = data[feature_cols].astype(float)
    y = data[target_col].astype(float)

    # Random split (not temporal — this is cross-sectional)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Validation for early stopping
    val_size = int(len(X_train) * 0.15)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]

    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric='mae',
    )

    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    metrics = {'MAE ($)': round(mae, 2), 'RMSE ($)': round(rmse, 2), 'R²': round(r2, 4)}
    print(f"  Credit Adjuster XGBoost: MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")

    if save_path:
        joblib.dump(model, save_path)
        print(f"  → Model saved to {save_path}")

    return model, preds, metrics, feature_cols


def generate_final_recommendations(df: pd.DataFrame,
                                     model,
                                     feature_cols: list,
                                     predicted_spend_col: str = None) -> pd.DataFrame:
    """
    Generate final credit line recommendations using the trained XGBoost model.
    Applies clamping and exclusion rules.
    """
    df = df.copy()

    # Ensure numeric
    df['cu_crd_line'] = pd.to_numeric(df['cu_crd_line'], errors='coerce').fillna(0)
    df['cu_cur_balance'] = pd.to_numeric(df['cu_cur_balance'], errors='coerce').fillna(0)

    # Prepare features
    feat_cols_available = [c for c in feature_cols if c in df.columns]
    X = df[feat_cols_available].fillna(0).astype(float)

    # Predict delta
    predicted_delta = model.predict(X)
    df['predicted_adjustment_delta'] = predicted_delta

    # Apply to current line
    df['recommended_credit_line'] = df['cu_crd_line'] + df['predicted_adjustment_delta']

    # ── Hard stop: exclusion flag ────────────────────────────────────────
    excl_mask = df['cu_line_incr_excl_flag'].astype(str).str.upper() == 'Y'
    df.loc[excl_mask & (df['predicted_adjustment_delta'] > 0), 'recommended_credit_line'] = \
        df.loc[excl_mask & (df['predicted_adjustment_delta'] > 0), 'cu_crd_line']
    df.loc[excl_mask & (df['predicted_adjustment_delta'] > 0), 'predicted_adjustment_delta'] = 0

    # ── Clamp ────────────────────────────────────────────────────────────
    min_line = df['cu_cur_balance'] * 1.10
    max_line = df['cu_crd_line'] * 1.50

    df['recommended_credit_line'] = np.clip(
        df['recommended_credit_line'], min_line, max_line
    )

    # Zero-line accounts stay at zero
    df.loc[df['cu_crd_line'] == 0, 'recommended_credit_line'] = 0

    # ── FINAL exclusion enforcement (after clamping) ─────────────────────
    df.loc[excl_mask, 'recommended_credit_line'] = np.minimum(
        df.loc[excl_mask, 'recommended_credit_line'],
        df.loc[excl_mask, 'cu_crd_line']
    )

    # Recalculate delta after clamping and exclusion
    df['adjustment_delta'] = df['recommended_credit_line'] - df['cu_crd_line']
    df.loc[df['cu_crd_line'] == 0, 'adjustment_delta'] = 0

    return df


# ═════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT TABLE
# ═════════════════════════════════════════════════════════════════════════════

def build_final_output(df: pd.DataFrame,
                        output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build the final deliverable: one row per account with all predictions.
    Sorted by adjustment_delta descending.
    """
    from .segmentation import SEGMENTS

    output_cols = [
        'current_account_nbr',
        'cu_crd_line',
        'q4_forecast',
        'segment_name',
        'risk_score',
        'anomaly_flag',
        'recommended_credit_line',
        'adjustment_delta',
    ]

    # Rename for clarity
    out = df.copy()
    if 'cu_crd_line' in out.columns:
        out = out.rename(columns={'cu_crd_line': 'current_credit_line'})
        output_cols = [c.replace('cu_crd_line', 'current_credit_line') for c in output_cols]

    # Ensure all columns exist
    for col in output_cols:
        if col not in out.columns:
            out[col] = np.nan

    # Select and sort
    result = out[output_cols].sort_values('adjustment_delta', ascending=False)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"  → Final output saved to {output_path} ({len(result):,} rows)")

    return result
