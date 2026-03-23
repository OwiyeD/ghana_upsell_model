"""Feature engineering module for the Ghana Upsell Model.

Transforms raw customer data into model-ready features.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_processing import PLAN_TIERS, REGIONS

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_spend_ghs",
    "data_usage_gb",
    "call_minutes",
    "sms_count",
    "num_complaints",
    "payment_on_time_rate",
    "plan_tier",
    "spend_per_gb",
    "call_data_ratio",
    "complaint_rate",
    "tenure_spend_interaction",
]

CATEGORICAL_FEATURES = ["region"]

TARGET_COL = "upsell"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from cleaned customer data.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        DataFrame with engineered features added.
    """
    df = df.copy()

    # Encode plan as an ordinal numeric
    df["plan_tier"] = df["current_plan"].map(PLAN_TIERS).fillna(1).astype(int)

    # Spend efficiency: spend per GB of data used
    df["spend_per_gb"] = np.where(
        df["data_usage_gb"] > 0,
        df["monthly_spend_ghs"] / df["data_usage_gb"],
        df["monthly_spend_ghs"],
    )

    # Usage ratio: call minutes per GB
    df["call_data_ratio"] = np.where(
        df["data_usage_gb"] > 0,
        df["call_minutes"] / df["data_usage_gb"],
        df["call_minutes"],
    )

    # Complaint rate relative to tenure
    df["complaint_rate"] = np.where(
        df["tenure_months"] > 0,
        df["num_complaints"] / df["tenure_months"],
        df["num_complaints"],
    )

    # Tenure × spend interaction
    df["tenure_spend_interaction"] = df["tenure_months"] * df["monthly_spend_ghs"]

    logger.info("Feature engineering complete: %d features created", len(NUMERIC_FEATURES))
    return df


def encode_categoricals(
    df: pd.DataFrame,
    encoders: Optional[dict] = None,
    fit: bool = True,
) -> tuple:
    """One-hot encode categorical features.

    Args:
        df: DataFrame with engineered features.
        encoders: Dict of existing encoders (used during inference).
        fit: Whether to fit new encoders (True for training, False for inference).

    Returns:
        Tuple of (transformed DataFrame, encoders dict).
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)

        if fit:
            encoders[col] = list(dummies.columns)
        else:
            # Align columns to those seen during training
            expected_cols = encoders.get(col, [])
            for c in expected_cols:
                if c not in dummies.columns:
                    dummies[c] = 0.0
            dummies = dummies[[c for c in expected_cols if c in dummies.columns]]

        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col], errors="ignore")

    return df, encoders


def scale_numerics(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple:
    """Standardise numeric features.

    Args:
        df: DataFrame with numeric features.
        scaler: Existing scaler (used during inference).
        fit: Whether to fit a new scaler.

    Returns:
        Tuple of (transformed DataFrame, scaler).
    """
    df = df.copy()
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])

    return df, scaler


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns to pass to the model.

    Args:
        df: DataFrame after feature engineering and encoding.

    Returns:
        List of feature column names (excludes id and target columns).
    """
    exclude = {"customer_id", TARGET_COL, "current_plan"}
    return [c for c in df.columns if c not in exclude]


def prepare_features(
    df: pd.DataFrame,
    encoders: Optional[dict] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple:
    """Full feature preparation pipeline.

    Args:
        df: Cleaned customer DataFrame.
        encoders: Existing categorical encoders.
        scaler: Existing numeric scaler.
        fit: Whether to fit new encoders/scaler.

    Returns:
        Tuple of (feature DataFrame, target Series, encoders, scaler).
    """
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)
    df, scaler = scale_numerics(df, scaler=scaler, fit=fit)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None

    return X, y, encoders, scaler
