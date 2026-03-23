"""Model training and evaluation for the Ghana Upsell Model.

Implements an XGBoost-based binary classifier with evaluation utilities.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}


def build_model(params: Optional[Dict[str, Any]] = None) -> XGBClassifier:
    """Build an XGBoost classifier.

    Args:
        params: Optional hyperparameter overrides.

    Returns:
        Untrained XGBClassifier instance.
    """
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    return XGBClassifier(**model_params)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """Train the XGBoost classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        params: Optional hyperparameter overrides.

    Returns:
        Trained XGBClassifier.
    """
    # Adjust scale_pos_weight for class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    if pos > 0:
        scale = neg / pos
        params = {**(params or {}), "scale_pos_weight": scale}

    model = build_model(params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("Model training complete")
    return model


def evaluate_model(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate the model on a dataset.

    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: True labels.
        threshold: Classification decision threshold.

    Returns:
        Dict with evaluation metrics.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    logger.info(
        "Evaluation — ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1"],
    )
    return metrics


def get_feature_importance(
    model: XGBClassifier, feature_names: list
) -> pd.DataFrame:
    """Return feature importances as a sorted DataFrame.

    Args:
        model: Trained classifier.
        feature_names: List of feature column names.

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
