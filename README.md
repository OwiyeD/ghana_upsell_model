# Ghana Upsell Model

A machine-learning pipeline that predicts which customers are most likely to
upgrade to a higher-tier product or service plan. The model is built around
customer behavioural and demographic signals common in the Ghanaian telecom
market.

---

## Project Structure

```
ghana_upsell_model/
├── src/
│   ├── data_processing.py     # Data loading, cleaning, splitting & synthetic data
│   ├── feature_engineering.py # Feature creation, encoding and scaling
│   ├── model.py               # XGBoost classifier training and evaluation
│   └── predict.py             # Pipeline persistence and batch scoring
├── tests/
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   └── test_model_predict.py
├── train.py                   # End-to-end training entry-point
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### Using synthetic data (no CSV required)

```bash
python train.py
```

This generates 5 000 synthetic customer records, trains the model and saves
the pipeline to `models/ghana_upsell_pipeline.joblib`.

### Using your own data

```bash
python train.py --data-path data/customers.csv --output-dir models/
```

The CSV must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `customer_id` | str | Unique customer identifier |
| `age` | int | Customer age (18–85) |
| `tenure_months` | int | Months as a customer |
| `current_plan` | str | `basic`, `standard`, or `premium` |
| `monthly_spend_ghs` | float | Average monthly spend in Ghanaian Cedi |
| `data_usage_gb` | float | Monthly data consumed (GB) |
| `call_minutes` | float | Monthly call minutes |
| `sms_count` | int | Monthly SMS count |
| `num_complaints` | int | Number of complaints raised |
| `payment_on_time_rate` | float | Proportion of on-time payments (0–1) |
| `region` | str | Ghanaian region (e.g. `greater_accra`) |
| `upsell` | int | Target: 1 = upsell, 0 = no upsell |

---

## Inference

```python
from src.predict import load_pipeline, predict

pipeline = load_pipeline("models/ghana_upsell_pipeline.joblib")
results = predict(new_customer_df, pipeline, threshold=0.5)
print(results[["customer_id", "upsell_probability", "upsell_prediction"]])
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Model Overview

The pipeline uses **XGBoost** (gradient-boosted trees) for binary
classification. Key engineered features include:

- **Plan tier** — ordinal encoding of `basic / standard / premium`
- **Spend per GB** — efficiency of data spend
- **Call-to-data ratio** — voice vs. data usage balance
- **Complaint rate** — complaints normalised by tenure
- **Tenure × spend interaction** — combined loyalty and value signal

Class imbalance is handled automatically by adjusting `scale_pos_weight`
during training.