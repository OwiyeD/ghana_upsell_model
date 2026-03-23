"""Tests for feature_engineering module."""

import pytest
import pandas as pd
import numpy as np

from src.data_processing import clean_data, generate_sample_data
from src.feature_engineering import (
    encode_categoricals,
    engineer_features,
    get_feature_columns,
    prepare_features,
    scale_numerics,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def clean_df():
    raw = generate_sample_data(n_samples=300, random_state=1)
    return clean_data(raw)


class TestEngineerFeatures:
    def test_adds_plan_tier(self, clean_df):
        df = engineer_features(clean_df)
        assert "plan_tier" in df.columns

    def test_plan_tier_values(self, clean_df):
        df = engineer_features(clean_df)
        assert set(df["plan_tier"].unique()).issubset({1, 2, 3})

    def test_adds_spend_per_gb(self, clean_df):
        df = engineer_features(clean_df)
        assert "spend_per_gb" in df.columns
        assert (df["spend_per_gb"] >= 0).all()

    def test_adds_call_data_ratio(self, clean_df):
        df = engineer_features(clean_df)
        assert "call_data_ratio" in df.columns

    def test_adds_complaint_rate(self, clean_df):
        df = engineer_features(clean_df)
        assert "complaint_rate" in df.columns

    def test_adds_tenure_spend_interaction(self, clean_df):
        df = engineer_features(clean_df)
        assert "tenure_spend_interaction" in df.columns

    def test_does_not_modify_original(self, clean_df):
        original_cols = list(clean_df.columns)
        engineer_features(clean_df)
        assert list(clean_df.columns) == original_cols


class TestEncodeCategoricals:
    def test_region_encoded(self, clean_df):
        df = engineer_features(clean_df)
        encoded, encoders = encode_categoricals(df)
        assert "region" not in encoded.columns
        assert any(c.startswith("region_") for c in encoded.columns)

    def test_encoders_returned(self, clean_df):
        df = engineer_features(clean_df)
        _, encoders = encode_categoricals(df)
        assert "region" in encoders

    def test_inference_columns_aligned(self, clean_df):
        df = engineer_features(clean_df)
        train_encoded, encoders = encode_categoricals(df.iloc[:200])
        test_encoded, _ = encode_categoricals(df.iloc[200:], encoders=encoders, fit=False)
        train_region_cols = [c for c in train_encoded.columns if c.startswith("region_")]
        test_region_cols = [c for c in test_encoded.columns if c.startswith("region_")]
        assert set(train_region_cols) == set(test_region_cols)


class TestScaleNumerics:
    def test_returns_dataframe_and_scaler(self, clean_df):
        df = engineer_features(clean_df)
        scaled, scaler = scale_numerics(df)
        assert isinstance(scaled, pd.DataFrame)
        assert scaler is not None

    def test_numeric_features_scaled(self, clean_df):
        df = engineer_features(clean_df)
        scaled, _ = scale_numerics(df)
        for col in NUMERIC_FEATURES:
            if col in scaled.columns:
                assert abs(scaled[col].mean()) < 1.0


class TestPrepareFeatures:
    def test_returns_X_y(self, clean_df):
        X, y, encoders, scaler = prepare_features(clean_df, fit=True)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_no_target_in_X(self, clean_df):
        X, _, _, _ = prepare_features(clean_df, fit=True)
        assert "upsell" not in X.columns

    def test_no_customer_id_in_X(self, clean_df):
        X, _, _, _ = prepare_features(clean_df, fit=True)
        assert "customer_id" not in X.columns

    def test_consistent_columns_between_train_and_test(self, clean_df):
        train = clean_df.iloc[:200]
        test = clean_df.iloc[200:]
        X_train, _, encoders, scaler = prepare_features(train, fit=True)
        X_test, _, _, _ = prepare_features(test, encoders=encoders, scaler=scaler, fit=False)
        assert list(X_train.columns) == list(X_test.columns)
