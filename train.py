"""Main training script for the Ghana Upsell Model.

Usage:
    python train.py [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]

Examples:
    # Train on synthetic data
    python train.py

    # Train on a custom CSV
    python train.py --data-path data/customers.csv --output-dir models/
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data_processing import clean_data, generate_sample_data, load_data, split_data
from src.feature_engineering import prepare_features
from src.model import evaluate_model, get_feature_importance, train_model
from src.predict import save_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Ghana Upsell Model")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the input CSV file. Defaults to synthetic data if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the trained pipeline (default: models/).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of synthetic samples when --data-path is not provided (default: 5000).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --- Load data ---
    if args.data_path:
        logger.info("Loading data from %s", args.data_path)
        raw_df = load_data(args.data_path)
    else:
        logger.info("No data path provided — generating %d synthetic samples", args.n_samples)
        raw_df = generate_sample_data(n_samples=args.n_samples)

    # --- Clean ---
    clean_df = clean_data(raw_df)
    logger.info("Upsell rate in dataset: %.2f%%", clean_df["upsell"].mean() * 100)

    # --- Split ---
    train_df, val_df, test_df = split_data(clean_df)

    # --- Feature engineering (fit on train, transform all) ---
    X_train, y_train, encoders, scaler = prepare_features(train_df, fit=True)
    X_val, y_val, _, _ = prepare_features(val_df, encoders=encoders, scaler=scaler, fit=False)
    X_test, y_test, _, _ = prepare_features(test_df, encoders=encoders, scaler=scaler, fit=False)

    # --- Train ---
    model = train_model(X_train, y_train, X_val, y_val)

    # --- Evaluate ---
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics:       %s", test_metrics)

    # --- Feature importance ---
    fi = get_feature_importance(model, list(X_train.columns))
    logger.info("Top-10 features:\n%s", fi.head(10).to_string(index=False))

    # --- Save pipeline ---
    pipeline_path = save_pipeline(model, encoders, scaler, output_dir=args.output_dir)
    logger.info("Training complete. Pipeline saved to %s", pipeline_path)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pipeline_path": str(pipeline_path),
    }


if __name__ == "__main__":
    main()
