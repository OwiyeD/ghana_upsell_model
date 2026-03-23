"""Data processing module for the Ghana Upsell Model.

Handles loading, cleaning, and validation of customer data.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Expected feature columns in raw customer data
REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "current_plan",
    "monthly_spend_ghs",
    "data_usage_gb",
    "call_minutes",
    "sms_count",
    "num_complaints",
    "payment_on_time_rate",
    "region",
    "upsell",
]

PLAN_TIERS = {
    "basic": 1,
    "standard": 2,
    "premium": 3,
}

REGIONS = [
    "greater_accra",
    "ashanti",
    "western",
    "eastern",
    "central",
    "volta",
    "northern",
    "upper_east",
    "upper_west",
    "bono",
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load customer data from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with raw customer data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that a DataFrame contains all required columns.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer data.

    Steps:
    - Drop duplicate customer records (keep latest).
    - Clip numeric columns to valid ranges.
    - Fill missing values with sensible defaults.
    - Normalise string columns to lowercase.

    Args:
        df: Raw customer DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Remove duplicate customers, keep first occurrence
    df = df.drop_duplicates(subset=["customer_id"], keep="first")

    # Normalise string columns
    df["current_plan"] = df["current_plan"].str.lower().str.strip()
    df["region"] = df["region"].str.lower().str.strip()

    # Clip numerics to realistic ranges
    df["age"] = df["age"].clip(lower=18, upper=85)
    df["tenure_months"] = df["tenure_months"].clip(lower=0)
    df["monthly_spend_ghs"] = df["monthly_spend_ghs"].clip(lower=0)
    df["data_usage_gb"] = df["data_usage_gb"].clip(lower=0)
    df["call_minutes"] = df["call_minutes"].clip(lower=0)
    df["sms_count"] = df["sms_count"].clip(lower=0)
    df["num_complaints"] = df["num_complaints"].clip(lower=0)
    df["payment_on_time_rate"] = df["payment_on_time_rate"].clip(lower=0.0, upper=1.0)

    # Fill missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["monthly_spend_ghs"] = df["monthly_spend_ghs"].fillna(df["monthly_spend_ghs"].median())
    df["data_usage_gb"] = df["data_usage_gb"].fillna(0.0)
    df["call_minutes"] = df["call_minutes"].fillna(0.0)
    df["sms_count"] = df["sms_count"].fillna(0.0)
    df["num_complaints"] = df["num_complaints"].fillna(0)
    df["payment_on_time_rate"] = df["payment_on_time_rate"].fillna(1.0)
    df["current_plan"] = df["current_plan"].fillna("basic")
    df["region"] = df["region"].fillna("greater_accra")

    logger.info("Cleaned data: %d rows, %d columns", len(df), len(df.columns))
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "upsell",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation and test sets.

    Args:
        df: Cleaned customer DataFrame.
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of the remaining data to use for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    effective_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=effective_val_size,
        random_state=random_state,
        stratify=train_val[target_col],
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(train), len(val), len(test)
    )
    return train, val, test


def generate_sample_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic customer data for development and testing.

    Args:
        n_samples: Number of customer records to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic customer data.
    """
    rng = np.random.default_rng(random_state)

    ages = rng.integers(18, 65, size=n_samples)
    tenure_months = rng.integers(1, 72, size=n_samples)
    current_plans = rng.choice(list(PLAN_TIERS.keys()), size=n_samples, p=[0.5, 0.35, 0.15])
    monthly_spend = rng.uniform(20, 300, size=n_samples).round(2)
    data_usage_gb = rng.exponential(scale=5, size=n_samples).round(2)
    call_minutes = rng.exponential(scale=150, size=n_samples).round(0)
    sms_count = rng.integers(0, 500, size=n_samples)
    num_complaints = rng.integers(0, 5, size=n_samples)
    payment_on_time_rate = rng.beta(a=8, b=2, size=n_samples).round(4)
    regions = rng.choice(REGIONS, size=n_samples)

    # Target: higher-spend, heavy-data, tenured customers on non-premium plans
    # are more likely to upsell
    plan_enc = np.array([PLAN_TIERS[p] for p in current_plans])
    upsell_prob = (
        0.1
        + 0.3 * (plan_enc < 3).astype(float)
        + 0.2 * (data_usage_gb > 8).astype(float)
        + 0.15 * (tenure_months > 24).astype(float)
        + 0.1 * (monthly_spend > 150).astype(float)
        - 0.1 * (num_complaints > 2).astype(float)
    ).clip(0, 1)
    upsell = rng.binomial(1, upsell_prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"GH{str(i).zfill(6)}" for i in range(1, n_samples + 1)],
            "age": ages,
            "tenure_months": tenure_months,
            "current_plan": current_plans,
            "monthly_spend_ghs": monthly_spend,
            "data_usage_gb": data_usage_gb,
            "call_minutes": call_minutes,
            "sms_count": sms_count,
            "num_complaints": num_complaints,
            "payment_on_time_rate": payment_on_time_rate,
            "region": regions,
            "upsell": upsell,
        }
    )
    return df
