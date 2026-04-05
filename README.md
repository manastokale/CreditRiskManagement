# Synchrony Credit Intelligence

A four-part predictive ML framework for credit card account management, built on anonymized data from Synchrony Financial (~1.5M transactions, 18,070 accounts).

**Core question:** Given an account's spending history, behavioral signals, and risk profile — what's a safe credit limit to offer for Q4 2025?

---

## Quick Start

```bash
# 1. Clone and enter the project
cd synchrony_credit_intelligence

# 2. Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Place CSV files in data/raw/
# (account_dim, statement_fact, transaction_fact, wrld_stor_tran_fact,
#  syf_id, rams_batch_cur, fraud_claim_case, fraud_claim_tran)

# 4. Run notebooks in order (01 → 06)
jupyter notebook notebooks/

# 5. Final output lands in outputs/predictions/final_output.csv
```

---

## Pipeline Overview

```
Data Loading → EDA → Feature Engineering → Q4 Forecast → Segmentation → Fraud/Risk → Credit Adjustment
     ↓           ↓          ↓                  ↓              ↓             ↓              ↓
  DuckDB     Missing     customer_base     Holt-Winters   Rule-based   Isolation     Rule-based +
  CSV ingest values,     customer_monthly  XGBoost        + Random     Forest +      XGBoost
  + union    corr,       transaction_      LSTM           Forest       Composite     regressor
  txn tables payment     enriched          (3-model       (4-class)    risk score    + clamping
             history                       comparison)    classifier   (0-100)
```

---

## Key Findings

### Q4 Spending Forecast — Model Comparison

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|----------|----|
| Holt-Winters (baseline) | 3,176.43 | 6,291.57 | 0.6125 |
| **XGBoost (primary)** | **32.15** | **172.22** | **0.9906** |
| LSTM (exploration) | 298.93 | 1,058.49 | 0.7627 |

> XGBoost with lag features is the primary model. The LSTM demonstrates sequence modeling capability but XGBoost typically outperforms on ~18k accounts.

### Impact Summary

- **Portfolio Current Exposure**: $113.04M
- **Portfolio Recommended Exposure**: $120.01M
- **Total Positive Adjustment**: +$7.03M across 10,240 accounts
- **Total Negative Adjustment**: -$59.88K across 7,208 accounts
- **Flat Adjustments**: 622 accounts

### Account Segmentation

| Segment | Description |
|---------|-------------|
| **Eligible - No Risk** | Growing spend, low utilization, clean history |
| **Eligible - With Risk** | Active but high utilization or minor delinquency |
| **No Increase Needed** | Stable, no growth signal |
| **Non-Performing** | Delinquent, fraud-flagged, or inactive |

### Risk Score Components (Interpretable, Documented)

| Component | Weight | Description |
|-----------|--------|-------------|
| Delinquency depth | 30% | Days delinquent (0-90 → 0-100) |
| Utilization level | 25% | Current balance / credit line |
| Anomaly flag | 20% | Isolation Forest unsupervised detection |
| NSF frequency | 15% | Returned checks in last 12 months |
| Payment history | 10% | Max delinquency from 12-cycle string |

### Fraud Detection

- 77 known fraud cases across 18,070 accounts (0.43%)
- Isolation Forest contamination auto-calibrated from actual fraud rate
- `has_fraud_case` excluded from training features (used only for evaluation)

---

## Project Structure

```
synchrony_credit_intelligence/
├── data/raw/                          # CSV data files (symlinked)
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb   # Feature tables and lag features
│   ├── 03_q4_prediction.ipynb         # 3-model spending forecast
│   ├── 04_segmentation.ipynb          # Rule-based + Random Forest
│   ├── 05_fraud_risk.ipynb            # Isolation Forest + risk scoring
│   └── 06_credit_line_adjustment.ipynb # Credit recommendations + SHAP
├── src/
│   ├── data_loader.py                 # DuckDB CSV ingestion
│   ├── preprocessing.py               # Cleaning, payment history parser
│   ├── features.py                    # Feature engineering pipelines
│   └── models/
│       ├── spend_predictor.py         # HW / XGBoost / LSTM
│       ├── segmentation.py            # Rules + Random Forest
│       ├── risk_detector.py           # Isolation Forest + risk score
│       └── credit_adjuster.py         # Rules + XGBoost + clamping
├── outputs/
│   ├── predictions/                   # CSV outputs and parquet intermediates
│   ├── saved_models/                  # Serialized model artifacts
│   └── figures/                       # All visualizations
├── requirements.txt
└── README.md
```

---

## Technical Decisions

### Why DuckDB?
Handles ~1.5M transaction rows in-memory with proper SQL including window functions and `DISTINCT ON`. No database server required. The `rams_batch_cur` deduplication (latest snapshot per account via `ROW_NUMBER()`) is cleaner in SQL than pandas.

### Why temporal train/test split?
Train on Jan–Sep, test on Oct–Dec. A random row split would leak future data into training. This is the single most important methodological decision in the pipeline.

### Why rule-based labels?
No ground truth for segments exists. Rules encode business knowledge (utilization thresholds, delinquency cutoffs, fraud flags) into labels that a Random Forest then learns to generalize.

### Why clamp credit recommendations?
- **Minimum**: current balance × 1.10 (never recommend below what the customer already owes)
- **Maximum**: current limit × 1.50 (no doubling credit lines overnight without justification)
- **Hard stop**: `cu_line_incr_excl_flag = 'Y'` overrides all increases

### Why SHAP?
Financial ML models require explainability for regulatory compliance. Being able to explain what drives a specific credit recommendation is more important than marginal accuracy improvements.

---

## Visualizations Generated

1. **Monthly spend heatmap** — accounts × months, colored by spend level
2. **Segment distribution** — account counts + total credit exposure per segment
3. **Q4 forecast vs actuals scatter** — predictions vs ground truth
4. **Risk score distribution by segment** — box plots showing separation
5. **Credit line adjustment waterfall** — current vs recommended by segment
6. **SHAP beeswarm** — global feature importance with direction of effect
7. **Anomaly scatter** — utilization vs avg transaction, colored by anomaly flag

---

## Final Output

`outputs/predictions/final_output.csv` — one row per account:

| Column | Description |
|--------|-------------|
| `current_account_nbr` | Tokenized account identifier |
| `current_credit_line` | Current credit limit ($) |
| `q4_forecast` | Predicted Q4 2024 total spend ($) |
| `segment_name` | Assigned risk segment |
| `risk_score` | Composite risk score (0-100) |
| `anomaly_flag` | Isolation Forest anomaly detection |
| `recommended_credit_line` | Model-recommended credit limit ($) |
| `adjustment_delta` | Recommended change from current ($) |

Sorted by `adjustment_delta` descending — highest-value opportunities first.

---

## Limitations

- **Seasonal estimation**: Only ~12 months of data — Holt-Winters cannot reliably estimate seasonal components
- **LSTM sample size**: ~18k accounts is small for deep learning; XGBoost will typically dominate
- **Fraud class imbalance**: 77 fraud cases out of 18,070 accounts (0.43%) — anomaly detection recall will be limited
- **Rule-based segment labels**: No ground truth exists; thresholds are calibrated heuristics
- **Single snapshot training**: The pipeline trains on one historical window; production systems would retrain periodically

## Next Steps

- Deploy as a batch scoring pipeline with scheduled retraining
- Add LIME for instance-level explanations alongside SHAP
- Incorporate external economic indicators (interest rates, unemployment) as features
- Build a monitoring dashboard for credit line recommendation drift
- Explore neural network-derived cluster embeddings for segmentation enhancement

---

## Data Source

Anonymized credit card account data from Synchrony Financial (Datathon 2025). All account identifiers are tokenized.

## Tech Stack

DuckDB · pandas · scikit-learn · XGBoost · statsmodels · TensorFlow/Keras · SHAP · Plotly · seaborn
