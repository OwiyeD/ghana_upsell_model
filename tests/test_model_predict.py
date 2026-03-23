"""Tests for model and predict modules."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import clean_data, generate_sample_data, split_data
from src.feature_engineering import prepare_features
from src.model import (
    build_model,
    evaluate_model,
    get_feature_importance,
    train_model,
)
from src.predict import load_pipeline, predict, save_pipeline


@pytest.fixture(scope="module")
def trained_pipeline():
    """Return a trained pipeline for use in multiple tests."""
    raw = generate_sample_data(n_samples=600, random_state=2)
    clean = clean_data(raw)
    train_df, val_df, test_df = split_data(clean)

    X_train, y_train, encoders, scaler = prepare_features(train_df, fit=True)
    X_val, y_val, _, _ = prepare_features(val_df, encoders=encoders, scaler=scaler, fit=False)
    X_test, y_test, _, _ = prepare_features(test_df, encoders=encoders, scaler=scaler, fit=False)

    model = train_model(X_train, y_train, X_val, y_val)
    return model, encoders, scaler, X_test, y_test, test_df


class TestBuildModel:
    def test_returns_classifier(self):
        model = build_model()
        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)

    def test_custom_params_applied(self):
        model = build_model({"n_estimators": 50})
        assert model.n_estimators == 50


class TestTrainModel:
    def test_model_is_fitted(self, trained_pipeline):
        model, *_ = trained_pipeline
        # XGBoost fitted models have feature_importances_
        assert model.feature_importances_ is not None

    def test_predict_proba_shape(self, trained_pipeline):
        model, _, _, X_test, y_test, _ = trained_pipeline
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestEvaluateModel:
    def test_returns_expected_metrics(self, trained_pipeline):
        model, _, _, X_test, y_test, _ = trained_pipeline
        metrics = evaluate_model(model, X_test, y_test)
        for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "confusion_matrix"):
            assert key in metrics

    def test_roc_auc_above_chance(self, trained_pipeline):
        model, _, _, X_test, y_test, _ = trained_pipeline
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["roc_auc"] > 0.5


class TestGetFeatureImportance:
    def test_returns_sorted_dataframe(self, trained_pipeline):
        model, _, _, X_test, _, _ = trained_pipeline
        fi = get_feature_importance(model, list(X_test.columns))
        assert isinstance(fi, pd.DataFrame)
        assert list(fi.columns) == ["feature", "importance"]
        assert fi["importance"].is_monotonic_decreasing

    def test_all_features_present(self, trained_pipeline):
        model, _, _, X_test, _, _ = trained_pipeline
        fi = get_feature_importance(model, list(X_test.columns))
        assert set(fi["feature"]) == set(X_test.columns)


class TestSaveLoadPipeline:
    def test_round_trip(self, trained_pipeline):
        model, encoders, scaler, _, _, _ = trained_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_pipeline(model, encoders, scaler, output_dir=tmpdir)
            assert path.exists()
            loaded = load_pipeline(path)
            assert "model" in loaded
            assert "encoders" in loaded
            assert "scaler" in loaded

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline("/nonexistent/path/pipeline.joblib")


class TestPredict:
    def test_adds_prediction_columns(self, trained_pipeline):
        model, encoders, scaler, _, _, test_df = trained_pipeline
        pipeline = {"model": model, "encoders": encoders, "scaler": scaler}
        result = predict(test_df, pipeline)
        assert "upsell_probability" in result.columns
        assert "upsell_prediction" in result.columns

    def test_probability_in_range(self, trained_pipeline):
        model, encoders, scaler, _, _, test_df = trained_pipeline
        pipeline = {"model": model, "encoders": encoders, "scaler": scaler}
        result = predict(test_df, pipeline)
        assert result["upsell_probability"].between(0, 1).all()

    def test_prediction_binary(self, trained_pipeline):
        model, encoders, scaler, _, _, test_df = trained_pipeline
        pipeline = {"model": model, "encoders": encoders, "scaler": scaler}
        result = predict(test_df, pipeline)
        assert set(result["upsell_prediction"].unique()).issubset({0, 1})
