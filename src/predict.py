"""Prediction / inference module for the Ghana Upsell Model.

Loads a persisted model pipeline and scores new customer records.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import pandas as pd

from src.data_processing import clean_data, validate_schema
from src.feature_engineering import prepare_features

logger = logging.getLogger(__name__)

PIPELINE_FILENAME = "ghana_upsell_pipeline.joblib"


def save_pipeline(
    model,
    encoders: dict,
    scaler,
    output_dir: Union[str, Path] = "models",
) -> Path:
    """Persist the trained model pipeline to disk.

    Args:
        model: Trained XGBClassifier.
        encoders: Fitted categorical encoders dict.
        scaler: Fitted StandardScaler.
        output_dir: Directory to save the pipeline artefact.

    Returns:
        Path to the saved pipeline file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = {"model": model, "encoders": encoders, "scaler": scaler}
    path = output_dir / PIPELINE_FILENAME
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s", path)
    return path


def load_pipeline(pipeline_path: Union[str, Path]) -> Dict:
    """Load a persisted model pipeline from disk.

    Args:
        pipeline_path: Path to the pipeline artefact.

    Returns:
        Dict containing 'model', 'encoders', and 'scaler'.

    Raises:
        FileNotFoundError: If the pipeline file does not exist.
    """
    pipeline_path = Path(pipeline_path)
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    logger.info("Pipeline loaded from %s", pipeline_path)
    return pipeline


def predict(
    df: pd.DataFrame,
    pipeline: Dict,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score a batch of customer records.

    Args:
        df: Raw or cleaned customer DataFrame (with all feature columns).
        pipeline: Dict with 'model', 'encoders', and 'scaler'.
        threshold: Classification decision threshold.

    Returns:
        DataFrame with original columns plus 'upsell_probability' and
        'upsell_prediction' columns.
    """
    model = pipeline["model"]
    encoders = pipeline["encoders"]
    scaler = pipeline["scaler"]

    df_clean = clean_data(df)
    X, _, _, _ = prepare_features(df_clean, encoders=encoders, scaler=scaler, fit=False)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = df_clean.copy()
    result["upsell_probability"] = probabilities.round(4)
    result["upsell_prediction"] = predictions

    logger.info(
        "Scored %d customers — predicted upsell rate: %.2f%%",
        len(result),
        predictions.mean() * 100,
    )
    return result
