"""Phase 2 validation: KW preprocessing and time-aware features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import TEMPORAL_SPLIT_DATE
from src.preprocessing import (
    FEATURE_COLUMNS,
    assert_no_temporal_leakage,
    engineer_pipe_features,
    prepare_pipe_break_data,
    temporal_train_test_split,
)
from src.schema import LEAKAGE_FORBIDDEN_COLUMNS, TARGET_COLUMN


def test_engineer_pipe_features_contract():
    df = engineer_pipe_features()
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert TARGET_COLUMN in df.columns
    assert "years_since_last_break" in df.columns
    assert_no_temporal_leakage(df)

    lower_cols = {c.lower() for c in df.columns}
    leaked = [c for c in LEAKAGE_FORBIDDEN_COLUMNS if c.lower() in lower_cols]
    assert leaked == []


def test_years_since_last_break_consistency():
    df = engineer_pipe_features()
    zero_prior = df["prior_break_count"] == 0
    assert df.loc[zero_prior, "years_since_last_break"].isna().all()

    positive_prior = df["prior_break_count"] > 0
    assert df.loc[positive_prior, "years_since_last_break"].notna().all()
    assert (df.loc[positive_prior, "years_since_last_break"] >= 0).all()


def test_worked_example_33550_time_aware_features():
    df = engineer_pipe_features()
    asset = df[df["asset_id"] == "33550"].copy()
    assert not asset.empty

    pos = asset[asset["snapshot_date"] == pd.Timestamp("1994-01-19")].iloc[0]
    neg = asset[asset["snapshot_date"] == pd.Timestamp("1989-01-19")].iloc[0]

    assert pos[TARGET_COLUMN] == 1
    assert pos["age_years"] == 35
    assert pos["prior_break_count"] == 0
    assert pd.isna(pos["years_since_last_break"])

    assert neg[TARGET_COLUMN] == 0
    assert neg["age_years"] == 30
    assert neg["prior_break_count"] == 0
    assert pd.isna(neg["years_since_last_break"])

    later_pos = asset[asset["snapshot_date"] == pd.Timestamp("2005-07-07")].iloc[0]
    assert later_pos["prior_break_count"] == 3
    assert later_pos["years_since_last_break"] == pytest.approx(
        (pd.Timestamp("2005-07-07") - pd.Timestamp("2004-10-15")).days / 365.25,
        rel=1e-6,
    )


def test_temporal_split_is_strictly_ordered():
    df = engineer_pipe_features()
    train_df, test_df = temporal_train_test_split(df)
    cutoff = pd.Timestamp(TEMPORAL_SPLIT_DATE)

    assert train_df["snapshot_date"].max() < cutoff
    assert test_df["snapshot_date"].min() >= cutoff


def test_temporal_split_has_both_classes():
    df = engineer_pipe_features()
    train_df, test_df = temporal_train_test_split(df)

    assert {0, 1}.issubset(set(train_df[TARGET_COLUMN].unique()))
    assert {0, 1}.issubset(set(test_df[TARGET_COLUMN].unique()))


def test_prepare_pipe_break_data_exposes_only_feature_columns():
    X_train, X_test, y_train, y_test, cleaned = prepare_pipe_break_data()

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS
    assert "snapshot_date" not in X_train.columns
    assert "asset_id" not in X_train.columns
    assert TARGET_COLUMN not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})
    assert {0, 1}.issubset(set(y_train.unique()))
    assert {0, 1}.issubset(set(y_test.unique()))
    assert len(X_train) + len(X_test) == len(cleaned)
