"""
Q4 Spending Forecast — Holt-Winters, XGBoost, and LSTM models.

Three-model comparison for predicting Oct-Dec spending per account.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "Model") -> dict:
    """Compute MAE ($), RMSE ($), and R² for a forecast."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    metrics = {'model': model_name, 'MAE ($)': round(mae, 2),
               'RMSE ($)': round(rmse, 2), 'R²': round(r2, 4)}
    print(f"  {model_name}: MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")
    return metrics


def comparison_table(results: list) -> pd.DataFrame:
    """Build a comparison table from a list of metric dicts."""
    return pd.DataFrame(results).set_index('model')


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1: HOLT-WINTERS EXPONENTIAL SMOOTHING (Baseline)
# ═════════════════════════════════════════════════════════════════════════════

def train_holt_winters(customer_monthly: pd.DataFrame,
                        min_months: int = 3,
                        forecast_periods: int = 3) -> pd.DataFrame:
    """
    Fit per-account Holt-Winters with additive trend, no seasonal.
    Returns DataFrame with columns: current_account_nbr, hw_predicted_q4_spend.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    results = []
    accounts = customer_monthly.groupby('current_account_nbr')

    for acct, grp in accounts:
        ts = grp.sort_values('month')['total_spend'].values
        # Need at least min_months of non-zero history
        nonzero = ts[ts > 0]
        if len(nonzero) < min_months:
            continue

        try:
            model = ExponentialSmoothing(
                ts, trend='add', seasonal=None,
                initialization_method='estimated'
            )
            fit = model.fit(optimized=True, use_brute=False)
            forecast = fit.forecast(forecast_periods)
            q4_total = max(0, np.sum(forecast))  # Clamp negative
        except Exception:
            q4_total = np.mean(ts) * forecast_periods  # Fallback

        results.append({
            'current_account_nbr': acct,
            'hw_predicted_q4_spend': q4_total
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2: XGBOOST WITH LAG FEATURES (Main Model)
# ═════════════════════════════════════════════════════════════════════════════

def train_xgboost_spend(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         feature_cols: list,
                         target_col: str = 'total_spend',
                         save_path: Optional[str] = None) -> Tuple:
    """
    Train XGBoost regressor with temporal split and early stopping.
    Returns: (model, predictions, metrics_dict)
    """
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[target_col].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df[target_col].astype(float)

    # Validation split from training period (last 20%)
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mae',
    )

    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)  # Clamp negative predictions

    metrics = evaluate_predictions(y_test.values, preds, "XGBoost")

    if save_path:
        joblib.dump(model, save_path)
        print(f"  → Model saved to {save_path}")

    return model, preds, metrics


def predict_q4_xgboost(model, customer_monthly: pd.DataFrame,
                        feature_cols: list,
                        q4_months: list = None) -> pd.DataFrame:
    """
    Generate per-account Q4 total spend predictions using trained XGBoost.
    Aggregates monthly predictions into Q4 total.
    """
    if q4_months is None:
        q4_months = ['2024-10', '2024-11', '2024-12']

    q4_data = customer_monthly[
        customer_monthly['month'].dt.to_period('M').astype(str).isin(q4_months)
    ].copy()

    if len(q4_data) == 0:
        return pd.DataFrame(columns=['current_account_nbr', 'xgb_predicted_q4_spend'])

    X_q4 = q4_data[feature_cols].astype(float)
    q4_data['predicted_spend'] = np.maximum(model.predict(X_q4), 0)

    result = q4_data.groupby('current_account_nbr').agg(
        xgb_predicted_q4_spend=('predicted_spend', 'sum')
    ).reset_index()

    return result


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3: LSTM (Deep Learning Exploration)
# ═════════════════════════════════════════════════════════════════════════════

def build_lstm_sequences(customer_monthly: pd.DataFrame,
                          seq_length: int = 9,
                          min_history: int = 12) -> Tuple:
    """
    Build sequences of shape (accounts × seq_length × features).
    Input: 9 months → Predict: Q4 total spend (sum of months 10-12).
    Only includes accounts with >= min_history months of data.

    Returns: (X_sequences, y_targets, account_ids)
    """
    feature_cols = ['total_spend', 'prev_balance']

    sequences, targets, acct_ids = [], [], []

    for acct, grp in customer_monthly.groupby('current_account_nbr'):
        grp = grp.sort_values('month')
        if len(grp) < min_history:
            continue

        # Fill missing feature values
        for c in feature_cols:
            if c in grp.columns:
                grp[c] = pd.to_numeric(grp[c], errors='coerce').fillna(0)
            else:
                grp[c] = 0

        vals = grp[feature_cols].values
        spend = grp['total_spend'].values

        if len(vals) >= seq_length + 3:
            X = vals[:seq_length]  # First 9 months
            y = np.sum(spend[seq_length:seq_length+3])  # Next 3 months total
            sequences.append(X)
            targets.append(y)
            acct_ids.append(acct)

    if len(sequences) == 0:
        return np.array([]), np.array([]), []

    return np.array(sequences), np.array(targets), acct_ids


def train_lstm_spend(X: np.ndarray, y: np.ndarray,
                      test_size: float = 0.2,
                      save_path: Optional[str] = None) -> Tuple:
    """
    Train a 2-layer LSTM with dropout and early stopping.
    Returns: (model, predictions_on_test, metrics_dict, y_test)
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # Train/test split (temporal ordering preserved)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalize
    X_mean, X_std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)) + 1e-8
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std

    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train_norm, y_train_norm,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Predict and denormalize
    preds_norm = model.predict(X_test_norm, verbose=0).flatten()
    preds = preds_norm * y_std + y_mean
    preds = np.maximum(preds, 0)

    metrics = evaluate_predictions(y_test, preds, "LSTM")

    if save_path:
        model.save(save_path)
        print(f"  → LSTM model saved to {save_path}")

    return model, preds, metrics, y_test
