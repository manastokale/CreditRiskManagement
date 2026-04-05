"""
Feature Engineering Module — Build joined tables and derived features.

Creates customer_base, customer_monthly, and transaction_base_enriched
tables with all lag, ratio, trend, and flag features.
"""

import numpy as np
import pandas as pd
import duckdb
from typing import Optional

from . import preprocessing as pp


# ═════════════════════════════════════════════════════════════════════════════
# 1. CUSTOMER BASE — one row per account
# ═════════════════════════════════════════════════════════════════════════════

def build_customer_base(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Join account_dim + syf_id + latest rams_batch_cur + fraud_claim_case.
    Returns one row per account with all static + behavioral features.
    """
    query = """
    WITH latest_rams AS (
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY cu_account_nbr
                       ORDER BY cu_processing_date DESC
                   ) AS rn
            FROM rams_batch_cur
        ) sub
        WHERE rn = 1
    ),
    fraud_accounts AS (
        SELECT DISTINCT current_account_nbr,
               1 AS has_fraud_case,
               MAX(gross_fraud_amt) AS max_fraud_amt
        FROM fraud_claim_case
        GROUP BY current_account_nbr
    )
    SELECT
        a.*,
        s.confidence_level,
        r.cu_bhv_scr,
        r.ca_cash_bal_pct_crd_line,
        r.ca_cash_bal_pct_cash_line,
        r.cu_nbr_days_dlq,
        r.cu_nbr_of_plastics,
        r.ca_avg_utilz_lst_6_mnths,
        r.cu_cash_line_am,
        r.cu_crd_bureau_scr,
        r.cu_crd_line,
        r.cu_cur_balance,
        r.cu_cur_nbr_due,
        r.ca_current_utilz,
        r.cu_line_incr_excl_flag,
        r.ca_max_dlq_lst_6_mnths,
        r.ca_mnths_since_active,
        r.ca_mnths_since_cl_chng,
        r.ca_mob,
        r.ca_nsf_count_lst_12_months,
        r.cu_otb,
        r.rb_new_bhv_scr,
        r.rb_crd_gr_new_crd_gr,
        r.cu_processing_date,
        r.mo_tot_sales_array_1,
        r.mo_tot_sales_array_2,
        r.mo_tot_sales_array_3,
        r.mo_tot_sales_array_4,
        r.mo_tot_sales_array_5,
        r.mo_tot_sales_array_6,
        r.ca_avg_utilz_lst_3_mnths,
        COALESCE(f.has_fraud_case, 0) AS has_fraud_case,
        f.max_fraud_amt
    FROM account_dim a
    LEFT JOIN syf_id s
        ON a.current_account_nbr = s.account_nbr_pty
    LEFT JOIN latest_rams r
        ON a.current_account_nbr = r.cu_account_nbr
    LEFT JOIN fraud_accounts f
        ON a.current_account_nbr = f.current_account_nbr
    """
    df = con.execute(query).fetchdf()

    # ── Parse payment history ────────────────────────────────────────────
    df = pp.add_payment_history_features(df, 'payment_hist_1_12_mths')

    # ── Derive sales trend features from monthly arrays ──────────────────
    sales_cols = [f'mo_tot_sales_array_{i}' for i in range(1, 7)]
    for c in sales_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['avg_monthly_sales_6m'] = df[sales_cols].mean(axis=1)
    # Slope: (most recent - oldest) / 6
    df['sales_trend_slope'] = (df['mo_tot_sales_array_1'] - df['mo_tot_sales_array_6']) / 6.0

    # ── Ratio features ───────────────────────────────────────────────────
    df['cu_crd_line'] = pd.to_numeric(df['cu_crd_line'], errors='coerce').fillna(0)
    df['cu_cur_balance'] = pd.to_numeric(df['cu_cur_balance'], errors='coerce').fillna(0)
    df['cu_otb'] = pd.to_numeric(df['cu_otb'], errors='coerce').fillna(0)

    df['utilization_ratio'] = np.where(
        df['cu_crd_line'].values > 0,
        df['cu_cur_balance'].values / np.maximum(df['cu_crd_line'].values, 1),
        0.0
    )
    df['otb_ratio'] = np.where(
        df['cu_crd_line'].values > 0,
        df['cu_otb'].values / np.maximum(df['cu_crd_line'].values, 1),
        0.0
    )

    # ── Impute card_activation_date sentinel ─────────────────────────────
    df = pp.impute_sentinel(df, 'card_activation_date', 'NEVER_ACTIVATED')

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. CUSTOMER MONTHLY — one row per account per month
# ═════════════════════════════════════════════════════════════════════════════

def build_customer_monthly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Join statement_fact with transaction_base aggregated by account × month.
    Compute monthly aggregates: total spend, txn count, avg/max txn amount.
    """
    query = """
    WITH txn_monthly AS (
        SELECT
            current_account_nbr,
            DATE_TRUNC('month', CAST(transaction_date AS DATE)) AS txn_month,
            SUM(CASE WHEN transaction_code IN ('253', '259') THEN transaction_amt ELSE 0 END) AS total_spend,
            COUNT(*) AS txn_count,
            AVG(transaction_amt) AS avg_txn_amt,
            MAX(transaction_amt) AS max_txn_amt,
            SUM(CASE WHEN transaction_code = '254' THEN transaction_amt ELSE 0 END) AS total_cash_advance,
            SUM(CASE WHEN transaction_code = '255' THEN transaction_amt ELSE 0 END) AS total_returns,
            SUM(curr_markup_fee) AS total_markup_fees,
            SUM(CASE WHEN frgn_curr_code != '840' AND frgn_curr_code != '' THEN transaction_amt ELSE 0 END) AS total_foreign_txn_amt,
            SUM(CASE WHEN first_purchase_ind = 'Y' THEN 1 ELSE 0 END) AS first_purchase_count
        FROM transaction_base
        WHERE transaction_date IS NOT NULL
        GROUP BY current_account_nbr, DATE_TRUNC('month', CAST(transaction_date AS DATE))
    ),
    stmt_monthly AS (
        SELECT
            current_account_nbr,
            DATE_TRUNC('month', CAST(billing_cycle_date AS DATE)) AS stmt_month,
            MAX(prev_balance) AS prev_balance,
            MAX(return_check_cnt_total) AS return_check_cnt_total,
            MAX(return_check_cnt_ytd) AS return_check_cnt_ytd,
            MAX(return_check_cnt_last_mth) AS return_check_cnt_last_mth
        FROM statement_fact
        WHERE billing_cycle_date IS NOT NULL
        GROUP BY current_account_nbr, DATE_TRUNC('month', CAST(billing_cycle_date AS DATE))
    )
    SELECT
        COALESCE(t.current_account_nbr, s.current_account_nbr) AS current_account_nbr,
        COALESCE(t.txn_month, s.stmt_month) AS month,
        COALESCE(t.total_spend, 0) AS total_spend,
        COALESCE(t.txn_count, 0) AS txn_count,
        t.avg_txn_amt,
        t.max_txn_amt,
        COALESCE(t.total_cash_advance, 0) AS total_cash_advance,
        COALESCE(t.total_returns, 0) AS total_returns,
        COALESCE(t.total_markup_fees, 0) AS total_markup_fees,
        COALESCE(t.total_foreign_txn_amt, 0) AS total_foreign_txn_amt,
        COALESCE(t.first_purchase_count, 0) AS first_purchase_count,
        s.prev_balance,
        s.return_check_cnt_total,
        s.return_check_cnt_ytd,
        s.return_check_cnt_last_mth
    FROM txn_monthly t
    FULL OUTER JOIN stmt_monthly s
        ON t.current_account_nbr = s.current_account_nbr
        AND t.txn_month = s.stmt_month
    ORDER BY current_account_nbr, month
    """
    df = con.execute(query).fetchdf()

    # Ensure proper types
    df['month'] = pd.to_datetime(df['month'])
    df['total_spend'] = pd.to_numeric(df['total_spend'], errors='coerce').fillna(0)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features to customer_monthly.
    Creates: spend_lag_1, spend_lag_2, spend_lag_3,
             spend_rolling_mean_3, spend_rolling_std_3
    """
    df = df.sort_values(['current_account_nbr', 'month']).copy()
    grp = df.groupby('current_account_nbr')['total_spend']

    df['spend_lag_1'] = grp.shift(1)
    df['spend_lag_2'] = grp.shift(2)
    df['spend_lag_3'] = grp.shift(3)

    df['spend_rolling_mean_3'] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['spend_rolling_std_3'] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std()
    )
    df['spend_rolling_std_3'] = df['spend_rolling_std_3'].fillna(0)

    return df


def add_spend_to_limit_ratio(customer_monthly: pd.DataFrame,
                              customer_base: pd.DataFrame) -> pd.DataFrame:
    """Add spend-to-limit ratio by joining with customer_base credit line."""
    merged = customer_monthly.merge(
        customer_base[['current_account_nbr', 'cu_crd_line']],
        on='current_account_nbr',
        how='left'
    )
    crd_line = pd.to_numeric(merged['cu_crd_line'], errors='coerce').fillna(0)
    merged['spend_to_limit_ratio'] = np.where(
        crd_line.values > 0,
        merged['total_spend'].values / np.maximum(crd_line.values, 1),
        0.0
    )
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# 3. TRANSACTION BASE ENRICHED — with fraud flags
# ═════════════════════════════════════════════════════════════════════════════

def build_transaction_enriched(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Join transaction_base with fraud_claim_tran to flag fraudulent transactions.
    """
    query = """
    SELECT
        t.*,
        CASE WHEN f.current_account_nbr IS NOT NULL THEN 1 ELSE 0 END AS is_fraud_txn
    FROM transaction_base t
    LEFT JOIN fraud_claim_tran f
        ON t.current_account_nbr = f.current_account_nbr
        AND CAST(t.transaction_date AS DATE) = CAST(f.transaction_dt AS DATE)
        AND ABS(t.transaction_amt - f.transaction_am) < 0.01
    """
    return con.execute(query).fetchdf()


# ═════════════════════════════════════════════════════════════════════════════
# 4. Q4 FEATURE MATRIX — for the spending forecast model
# ═════════════════════════════════════════════════════════════════════════════

def build_q4_feature_matrix(customer_monthly: pd.DataFrame,
                             customer_base: pd.DataFrame,
                             cutoff_month: str = '2024-10-01') -> tuple:
    """
    Build the feature matrix for Q4 prediction.
    Train: months before cutoff_month
    Test: months >= cutoff_month

    Returns: (train_df, test_df, feature_cols, target_col)
    """
    # Work on a fresh copy to avoid side effects
    cm = customer_monthly.copy()

    # Add lag features if not already present
    if 'spend_lag_1' not in cm.columns:
        cm = add_lag_features(cm)

    # Add spend_to_limit_ratio if not already present
    if 'spend_to_limit_ratio' not in cm.columns:
        cm = add_spend_to_limit_ratio(cm, customer_base)

    # Merge static features from customer_base
    static_features = [
        'current_account_nbr', 'cu_bhv_scr', 'cu_crd_bureau_scr',
        'ca_current_utilz', 'ca_avg_utilz_lst_3_mnths',
        'ca_avg_utilz_lst_6_mnths', 'ca_mob', 'ca_nsf_count_lst_12_months',
        'ca_mnths_since_active', 'cu_nbr_days_dlq',
        'avg_monthly_sales_6m', 'sales_trend_slope',
        'delinquent_cycle_count', 'max_delinquency_level',
        'utilization_ratio', 'otb_ratio', 'has_fraud_case'
    ]
    # Only include columns that exist and aren't already in cm
    available_static = [c for c in static_features
                        if c in customer_base.columns and
                        (c == 'current_account_nbr' or c not in cm.columns)]
    if len(available_static) > 1:  # Must have at least account nbr + 1 feature
        cm = cm.merge(customer_base[available_static], on='current_account_nbr', how='left')

    # Also get cu_crd_line from customer_base if not in cm
    if 'cu_crd_line' not in cm.columns and 'cu_crd_line' in customer_base.columns:
        cm = cm.merge(
            customer_base[['current_account_nbr', 'cu_crd_line']].drop_duplicates(),
            on='current_account_nbr', how='left'
        )

    # Define feature columns
    feature_cols = [
        'spend_lag_1', 'spend_lag_2', 'spend_lag_3',
        'spend_rolling_mean_3', 'spend_rolling_std_3',
        'txn_count', 'avg_txn_amt', 'max_txn_amt',
        'total_cash_advance', 'total_returns',
        'prev_balance', 'spend_to_limit_ratio',
        'cu_bhv_scr', 'cu_crd_bureau_scr', 'cu_crd_line',
        'ca_current_utilz', 'ca_avg_utilz_lst_3_mnths',
        'ca_avg_utilz_lst_6_mnths', 'ca_mob',
        'ca_nsf_count_lst_12_months', 'ca_mnths_since_active',
        'avg_monthly_sales_6m', 'sales_trend_slope',
        'delinquent_cycle_count', 'max_delinquency_level',
        'utilization_ratio', 'otb_ratio', 'has_fraud_case'
    ]
    feature_cols = [c for c in feature_cols if c in cm.columns]
    target_col = 'total_spend'

    # Temporal split
    cutoff = pd.Timestamp(cutoff_month)
    train = cm[cm['month'] < cutoff].dropna(subset=feature_cols + [target_col])
    test = cm[cm['month'] >= cutoff].dropna(subset=feature_cols + [target_col])

    return train, test, feature_cols, target_col


# ═════════════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTION FEATURE MATRIX
# ═════════════════════════════════════════════════════════════════════════════

def build_anomaly_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Aggregate transaction-level data per account for Isolation Forest.
    NOTE: has_fraud_case is intentionally EXCLUDED from features
    but returned for evaluation.
    """
    query = """
    SELECT
        current_account_nbr,
        SUM(transaction_amt) AS total_spend,
        COUNT(*) AS txn_count,
        AVG(transaction_amt) AS avg_txn_amt,
        MAX(transaction_amt) AS max_txn_amt,
        SUM(CASE WHEN frgn_curr_code != '840' AND frgn_curr_code != ''
            THEN transaction_amt ELSE 0 END) AS total_foreign_amt,
        SUM(curr_markup_fee) AS total_markup_fees,
        SUM(CASE WHEN first_purchase_ind = 'Y' THEN 1 ELSE 0 END) AS first_purchase_count
    FROM transaction_base
    GROUP BY current_account_nbr
    """
    return con.execute(query).fetchdf()
