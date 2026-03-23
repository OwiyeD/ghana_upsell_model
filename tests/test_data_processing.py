"""Tests for data_processing module."""

import pytest
import pandas as pd
import numpy as np

from src.data_processing import (
    clean_data,
    generate_sample_data,
    split_data,
    validate_schema,
    REQUIRED_COLUMNS,
)


@pytest.fixture
def sample_df():
    return generate_sample_data(n_samples=200, random_state=0)


class TestGenerateSampleData:
    def test_returns_dataframe(self):
        df = generate_sample_data(100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        df = generate_sample_data(150)
        assert len(df) == 150

    def test_has_required_columns(self):
        df = generate_sample_data(100)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_unique_customer_ids(self):
        df = generate_sample_data(100)
        assert df["customer_id"].nunique() == 100

    def test_binary_target(self):
        df = generate_sample_data(500)
        assert set(df["upsell"].unique()).issubset({0, 1})

    def test_reproducible_with_seed(self):
        df1 = generate_sample_data(100, random_state=7)
        df2 = generate_sample_data(100, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)


class TestValidateSchema:
    def test_valid_schema_passes(self, sample_df):
        validate_schema(sample_df)  # should not raise

    def test_missing_column_raises(self, sample_df):
        bad_df = sample_df.drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(bad_df)


class TestCleanData:
    def test_returns_copy(self, sample_df):
        cleaned = clean_data(sample_df)
        assert cleaned is not sample_df

    def test_removes_duplicates(self, sample_df):
        duplicated = pd.concat([sample_df, sample_df.head(10)], ignore_index=True)
        cleaned = clean_data(duplicated)
        assert len(cleaned) == len(sample_df)

    def test_age_clipped(self):
        df = generate_sample_data(50)
        df.loc[0, "age"] = 5
        df.loc[1, "age"] = 100
        cleaned = clean_data(df)
        assert cleaned["age"].min() >= 18
        assert cleaned["age"].max() <= 85

    def test_payment_rate_clipped(self):
        df = generate_sample_data(50)
        df.loc[0, "payment_on_time_rate"] = -0.5
        df.loc[1, "payment_on_time_rate"] = 1.5
        cleaned = clean_data(df)
        assert cleaned["payment_on_time_rate"].between(0.0, 1.0).all()

    def test_string_columns_lowercase(self, sample_df):
        sample_df = sample_df.copy()
        sample_df.loc[0, "current_plan"] = "BASIC"
        sample_df.loc[0, "region"] = "ASHANTI"
        cleaned = clean_data(sample_df)
        assert cleaned["current_plan"].str.islower().all()
        assert cleaned["region"].str.islower().all()

    def test_no_nulls_in_key_columns(self, sample_df):
        cleaned = clean_data(sample_df)
        for col in ["age", "monthly_spend_ghs", "current_plan", "region"]:
            assert cleaned[col].isna().sum() == 0


class TestSplitData:
    def test_split_sizes(self, sample_df):
        train, val, test = split_data(sample_df)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_no_overlap(self, sample_df):
        train, val, test = split_data(sample_df)
        train_ids = set(train["customer_id"])
        val_ids = set(val["customer_id"])
        test_ids = set(test["customer_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_stratification(self, sample_df):
        train, val, test = split_data(sample_df)
        overall_rate = sample_df["upsell"].mean()
        train_rate = train["upsell"].mean()
        assert abs(train_rate - overall_rate) < 0.05
