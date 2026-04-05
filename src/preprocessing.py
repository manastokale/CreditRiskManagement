"""
Preprocessing Module — Data cleaning, encoding, and correlation pruning.

Handles missing values, payment history parsing, correlation-based
feature selection, and date parsing utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


# ── Payment History Flag Encoding ────────────────────────────────────────────
# From the Synchrony mapping document:
#
# Debit postings only:  A=0dlq, B=1, C=2, D=3, E=4, F=5, G=6, H=7
# Credit postings only: I=0dlq, J=1, K=2, L=3, M=4, N=5, O=6, P=7
# No postings:          0-7 = number of cycles delinquent
# Other:                Z = zero balance/no activity, Q = not yet due / data good
#
# External status flags (may appear in payment hist):
#   A=Auth prohibited, B=Bankrupt, C=Closed, E=Revoked, F=Frozen,
#   I=Interest accrual prohibited, L=Lost, U=Stolen, Z=Charged off

DELINQUENCY_MAP = {
    # ── No delinquency ──
    'A': 0, 'I': 0, 'Q': 0, 'Z': 0,
    '0': 0,
    '#': 0, '%': 0, '+': 0, '-': 0,  # credit balance variants
    # ── Delinquency levels ──
    'B': 1, 'J': 1, '1': 1,
    'C': 2, 'K': 2, '2': 2,
    'D': 3, 'L': 3, '3': 3,
    'E': 4, 'M': 4, 'U': 4, '4': 4,  # U=Stolen → treat as severe
    'F': 5, 'N': 5, '5': 5,
    'G': 6, 'O': 6, '6': 6,
    'H': 7, 'P': 7, '7': 7,
}

# External status flags that indicate risk
RISK_STATUS_FLAGS = {'B', 'C', 'E', 'F', 'L', 'U'}


def parse_payment_history(hist_str: str) -> Tuple[int, int, int]:
    """
    Parse a 12-character payment history string into numeric features.

    Returns:
        (delinquent_cycle_count, max_delinquency_level, risk_flag_count)
    """
    if not isinstance(hist_str, str) or len(hist_str) == 0:
        return (0, 0, 0)

    delinquent_count = 0
    max_delinquency = 0
    risk_flags = 0

    for ch in hist_str.upper():
        dlq = DELINQUENCY_MAP.get(ch, 0)
        if dlq > 0:
            delinquent_count += 1
            max_delinquency = max(max_delinquency, dlq)
        if ch in RISK_STATUS_FLAGS:
            risk_flags += 1

    return (delinquent_count, max_delinquency, risk_flags)


def add_payment_history_features(df: pd.DataFrame,
                                  col: str = 'payment_hist_1_12_mths') -> pd.DataFrame:
    """
    Add parsed payment history features to a DataFrame.
    Creates: delinquent_cycle_count, max_delinquency_level, risk_flag_count
    """
    df = df.copy()
    parsed = df[col].fillna('').apply(parse_payment_history)
    df['delinquent_cycle_count'] = parsed.apply(lambda x: x[0])
    df['max_delinquency_level'] = parsed.apply(lambda x: x[1])
    df['risk_flag_count'] = parsed.apply(lambda x: x[2])
    return df


# ── Missing Value Handling ───────────────────────────────────────────────────

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a clean missing-percentage summary per column.
    """
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    summary = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': pct,
        'dtype': df.dtypes
    }).sort_values('missing_pct', ascending=False)
    summary = summary[summary['missing_count'] > 0]
    return summary


def drop_high_missing_columns(df: pd.DataFrame,
                               threshold: float = 0.70) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with missing percentage above threshold.
    Returns (cleaned_df, list_of_dropped_columns).
    """
    pct_missing = df.isnull().mean()
    to_drop = pct_missing[pct_missing > threshold].index.tolist()
    return df.drop(columns=to_drop), to_drop


def impute_sentinel(df: pd.DataFrame, col: str,
                     sentinel_value='MISSING') -> pd.DataFrame:
    """
    Impute missing values with a sentinel value (for categorical / meaningful missingness).
    Casts column to string first to prevent mixed-type issues with Parquet/Arrow.
    """
    df = df.copy()
    df[col] = df[col].astype(str).replace({'nan': sentinel_value, '<NA>': sentinel_value, 'None': sentinel_value})
    df[col] = df[col].fillna(sentinel_value)
    return df


# ── Correlation-Based Feature Selection ──────────────────────────────────────

def drop_correlated_features(df: pd.DataFrame,
                              threshold: float = 0.85,
                              verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop one column from each pair of highly-correlated numeric features.
    Keeps the column that appears first (leftmost) in the DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    # Upper triangle mask
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for c in correlated:
            if c not in to_drop:
                to_drop.add(c)
                if verbose:
                    print(f"  Dropping '{c}' (corr={corr_matrix.loc[col, c]:.3f} with '{col}')")

    dropped = list(to_drop)
    return df.drop(columns=dropped, errors='ignore'), dropped


# ── Date Utilities ───────────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """Convert date columns to datetime, handling mixed formats."""
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def extract_month(df: pd.DataFrame, date_col: str,
                   new_col: str = 'year_month') -> pd.DataFrame:
    """Extract year-month period from a date column."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors='coerce')
    df[new_col] = dt.dt.to_period('M')
    return df


# ── Categorical Profiling ────────────────────────────────────────────────────

def profile_categorical(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    """Value counts with percentage for a categorical column."""
    counts = df[col].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({'count': counts, 'pct': pct}).head(top_n)


# ── External Status Reason Code Mapping ──────────────────────────────────────
# From the mapping document and instruction doc
EXT_STATUS_MAP = {
    0: 'Normal',
    'A': 'Authorization Prohibited',
    'B': 'Bankrupt',
    'C': 'Closed',
    'E': 'Revoked',
    'F': 'Frozen',
    'I': 'Interest Accrual Prohibited',
    'L': 'Lost',
    'U': 'Stolen',
    'Z': 'Charged Off',
}


def map_external_status(df: pd.DataFrame,
                         col: str = 'external_status_reason_code') -> pd.DataFrame:
    """Add a human-readable external status description."""
    df = df.copy()
    df['ext_status_desc'] = df[col].map(EXT_STATUS_MAP).fillna('Unknown')
    return df
