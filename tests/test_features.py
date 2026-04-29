"""Tests for src/features/engineering.py — leakage, correctness, shape."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    create_features,
    drop_warmup_rows,
    get_feature_columns,
)


@pytest.fixture
def weekly_df():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-06", periods=60, freq="W-MON")
    rows = []
    for state in ["California", "Texas"]:
        base = 100_000 if state == "California" else 60_000
        for date in dates:
            rows.append(
                {
                    "state": state,
                    "date": date,
                    "total": max(0.0, base + rng.normal(0, 5000)),
                    "category": "Beverages",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def config():
    return {
        "lag_periods": [1, 2, 4],
        "rolling_windows": [4],
        "rolling_stats": ["mean", "std"],
        "holiday_country": "US",
    }


# ── Shape and columns ──────────────────────────────────────────────────────


def test_create_features_returns_more_columns(weekly_df, config):
    result = create_features(weekly_df, config)
    assert len(result.columns) > len(weekly_df.columns)


def test_create_features_preserves_row_count(weekly_df, config):
    result = create_features(weekly_df, config)
    assert len(result) == len(weekly_df)


def test_get_feature_columns_excludes_base_columns(weekly_df, config):
    result = create_features(weekly_df, config)
    feat_cols = get_feature_columns(result)
    for base_col in ["state", "date", "total", "category"]:
        assert base_col not in feat_cols


def test_get_feature_columns_all_numeric(weekly_df, config):
    result = create_features(weekly_df, config)
    feat_cols = get_feature_columns(result)
    assert len(feat_cols) > 0
    for col in feat_cols:
        assert result[col].dtype in [
            np.float64,
            np.int64,
            np.int32,
            np.float32,
            np.uint32,
        ]


# ── Lag feature correctness ────────────────────────────────────────────────


def test_lag_1_equals_previous_week(weekly_df, config):
    result = create_features(weekly_df, config)
    ca = (
        result[result["state"] == "California"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    # lag_1 at row i should equal total at row i-1
    for i in range(1, len(ca)):
        if not np.isnan(ca.loc[i, "lag_1"]):
            assert ca.loc[i, "lag_1"] == pytest.approx(ca.loc[i - 1, "total"])


def test_lag_2_equals_two_weeks_ago(weekly_df, config):
    result = create_features(weekly_df, config)
    ca = (
        result[result["state"] == "California"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    for i in range(2, len(ca)):
        if not np.isnan(ca.loc[i, "lag_2"]):
            assert ca.loc[i, "lag_2"] == pytest.approx(ca.loc[i - 2, "total"])


# ── No cross-state leakage ─────────────────────────────────────────────────


def test_lag_features_do_not_leak_across_states(weekly_df, config):
    result = create_features(weekly_df, config)
    # First row of each state should have NaN lag_1 (no prior row for that state)
    for state in ["California", "Texas"]:
        state_df = (
            result[result["state"] == state].sort_values("date").reset_index(drop=True)
        )
        assert np.isnan(
            state_df.loc[0, "lag_1"]
        ), f"lag_1 at row 0 for {state} should be NaN"


# ── Rolling feature correctness ────────────────────────────────────────────


def test_rolling_mean_uses_past_data_only(weekly_df, config):
    result = create_features(weekly_df, config)
    ca = (
        result[result["state"] == "California"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    # rolling_mean_4 at row i = mean of total[max(0,i-4) : i] (shift(1) means window ends at i-1)  # noqa: E501
    for i in range(5, 15):
        expected = ca.loc[max(0, i - 4) : i - 1, "total"].mean()
        actual = ca.loc[i, "rolling_mean_4"]
        if not np.isnan(actual):
            assert actual == pytest.approx(expected, rel=1e-4)


# ── No leakage: rolling uses shift(1) ────────────────────────────────────


def test_rolling_feature_at_t_does_not_use_total_at_t(weekly_df, config):
    result = create_features(weekly_df, config)
    ca = (
        result[result["state"] == "California"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    # Modify one total value — the rolling_mean at that same row must NOT change
    modified = ca.copy()
    original_rolling = ca.loc[10, "rolling_mean_4"]
    modified.loc[10, "total"] = 999_999_999
    # Re-run features on modified single-state data
    single = weekly_df[weekly_df["state"] == "California"].copy()
    single = single.sort_values("date").reset_index(drop=True)
    single.loc[10, "total"] = 999_999_999
    feat_mod = create_features(single, config)
    feat_mod_ca = feat_mod.sort_values("date").reset_index(drop=True)
    # rolling_mean_4 at row 10 should be unchanged (it uses rows 6–9 via shift(1))
    assert feat_mod_ca.loc[10, "rolling_mean_4"] == pytest.approx(
        original_rolling, rel=1e-4
    )


# ── Warmup rows ───────────────────────────────────────────────────────────


def test_drop_warmup_rows_removes_nan_rows(weekly_df, config):
    result = create_features(weekly_df, config)
    clean = drop_warmup_rows(result)
    feat_cols = get_feature_columns(result)
    assert clean[feat_cols].isna().sum().sum() == 0


def test_drop_warmup_rows_preserves_non_nan_rows(weekly_df, config):
    result = create_features(weekly_df, config)
    clean = drop_warmup_rows(result)
    assert len(clean) < len(result)
    assert len(clean) > 0


# ── Calendar features ─────────────────────────────────────────────────────


def test_calendar_features_present(weekly_df, config):
    result = create_features(weekly_df, config)
    for col in ["week_of_year", "month", "quarter", "year"]:
        assert col in result.columns, f"Missing calendar feature: {col}"


def test_week_of_year_in_valid_range(weekly_df, config):
    result = create_features(weekly_df, config)
    woy = result["week_of_year"].dropna()
    assert woy.between(1, 53).all()


def test_month_in_valid_range(weekly_df, config):
    result = create_features(weekly_df, config)
    assert result["month"].between(1, 12).all()


# ── Holiday features ──────────────────────────────────────────────────────


def test_is_holiday_is_binary(weekly_df, config):
    result = create_features(weekly_df, config)
    assert set(result["is_holiday"].dropna().unique()).issubset({0, 1})


def test_days_to_next_holiday_non_negative(weekly_df, config):
    result = create_features(weekly_df, config)
    assert (result["days_to_next_holiday"].dropna() >= 0).all()
