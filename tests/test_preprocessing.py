"""Tests for src/preprocessing/ — cleaner, validator, pipeline."""

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from src.preprocessing.cleaner import (
    aggregate_duplicate_dates,
    aggregate_to_weekly,
    fill_missing_dates,
    handle_outliers,
    impute_missing,
    remove_duplicates,
)
from src.preprocessing.validator import clean_schema, raw_schema

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_daily(state="Alabama", n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "state": state,
            "date": dates,
            "total": rng.uniform(1000, 5000, n),
            "category": "Beverages",
        }
    )


# ── Duplicate removal ──────────────────────────────────────────────────────


def test_remove_duplicates_drops_exact_rows():
    df = _make_daily(n=10)
    df_dup = pd.concat([df, df.iloc[:3]], ignore_index=True)
    result = remove_duplicates(df_dup)
    assert len(result) == 10


def test_remove_duplicates_preserves_distinct_rows():
    df = _make_daily(n=10)
    assert len(remove_duplicates(df)) == 10


# ── Aggregate duplicate dates ──────────────────────────────────────────────


def test_aggregate_duplicate_dates_sums_total():
    df = pd.DataFrame(
        {
            "state": ["Alabama", "Alabama"],
            "date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01")],
            "total": [1000.0, 500.0],
            "category": ["Beverages", "Beverages"],
        }
    )
    result = aggregate_duplicate_dates(df).reset_index(drop=True)
    assert len(result) == 1
    assert result.loc[0, "total"] == pytest.approx(1500.0)


def test_aggregate_duplicate_dates_different_states_kept_separate():
    df = pd.DataFrame(
        {
            "state": ["Alabama", "Arizona"],
            "date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01")],
            "total": [1000.0, 2000.0],
            "category": ["Beverages", "Beverages"],
        }
    )
    result = aggregate_duplicate_dates(df)
    assert len(result) == 2


# ── Fill missing dates ─────────────────────────────────────────────────────


def test_fill_missing_dates_completes_range():
    df = pd.DataFrame(
        {
            "state": ["Alabama"] * 3,
            "date": pd.to_datetime(["2022-01-01", "2022-01-03", "2022-01-05"]),
            "total": [100.0, 300.0, 500.0],
            "category": "Beverages",
        }
    )
    result = fill_missing_dates(df)
    alabama = result[result["state"] == "Alabama"].sort_values("date")
    dates = alabama["date"].tolist()
    # Should have every day from Jan 1 to Jan 5
    assert pd.Timestamp("2022-01-02") in dates
    assert pd.Timestamp("2022-01-04") in dates


def test_fill_missing_dates_introduces_nan_for_gaps():
    df = pd.DataFrame(
        {
            "state": ["Alabama", "Alabama"],
            "date": pd.to_datetime(["2022-01-01", "2022-01-05"]),
            "total": [100.0, 500.0],
            "category": "Beverages",
        }
    )
    result = fill_missing_dates(df)
    gap_rows = result[(result["state"] == "Alabama") & (result["total"].isna())]
    assert len(gap_rows) >= 3  # Jan 2, 3, 4


# ── Imputation ─────────────────────────────────────────────────────────────


def test_impute_missing_fills_all_nulls():
    df = _make_daily(n=10)
    df.loc[3:5, "total"] = np.nan
    result = impute_missing(df)
    assert result["total"].isna().sum() == 0


def test_impute_missing_interpolates_correctly():
    df = pd.DataFrame(
        {
            "state": ["Alabama"] * 5,
            "date": pd.date_range("2022-01-01", periods=5, freq="D"),
            "total": [100.0, np.nan, np.nan, np.nan, 500.0],
            "category": "Beverages",
        }
    )
    result = impute_missing(df).sort_values("date").reset_index(drop=True)
    # Middle value should be interpolated between 100 and 500
    mid = result.loc[2, "total"]
    assert 100 < mid < 500


# ── Outlier handling ───────────────────────────────────────────────────────


def test_handle_outliers_caps_extreme_values():
    df = _make_daily(n=50)
    df.loc[0, "total"] = 1_000_000_000  # extreme high outlier
    result = handle_outliers(df, method="iqr", threshold=1.5)
    assert result["total"].max() < 1_000_000_000


def test_handle_outliers_no_negative_values():
    df = _make_daily(n=50)
    df.loc[0, "total"] = -999_999
    result = handle_outliers(df, method="iqr", threshold=1.5)
    assert (result["total"] >= 0).all()


# ── Weekly aggregation ─────────────────────────────────────────────────────


def test_aggregate_to_weekly_reduces_rows():
    df = _make_daily(n=70)
    result = aggregate_to_weekly(df)
    # 70 days ≈ 10 weeks — result should have far fewer rows than 70
    assert len(result) < 70


def test_aggregate_to_weekly_sums_total():
    # 7 daily rows for one week should sum to their total
    dates = pd.date_range("2022-01-03", periods=7, freq="D")  # Mon–Sun
    totals = [1000.0] * 7
    df = pd.DataFrame(
        {"state": "Alabama", "date": dates, "total": totals, "category": "Beverages"}
    )
    weekly = aggregate_to_weekly(df)
    assert weekly["total"].sum() == pytest.approx(7000.0)


def test_aggregate_to_weekly_dates_are_mondays():
    df = _make_daily(n=28)
    result = aggregate_to_weekly(df)
    for dt in result["date"]:
        assert pd.Timestamp(dt).dayofweek == 0  # 0 = Monday


# ── Pandera schema validation ──────────────────────────────────────────────


def test_raw_schema_accepts_valid_data():
    df = pd.DataFrame(
        {
            "state": ["Alabama"],
            "date": [pd.Timestamp("2022-01-01")],
            "total": [1000.0],
            "category": ["Beverages"],
        }
    )
    validated = raw_schema.validate(df)
    assert len(validated) == 1


def test_raw_schema_rejects_negative_total():
    df = pd.DataFrame(
        {
            "state": ["Alabama"],
            "date": [pd.Timestamp("2022-01-01")],
            "total": [-500.0],
            "category": ["Beverages"],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        raw_schema.validate(df)


def test_clean_schema_rejects_null_state():
    df = pd.DataFrame(
        {
            "state": [None],
            "date": [pd.Timestamp("2022-01-01")],
            "total": [1000.0],
            "category": ["Beverages"],
        }
    )
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        clean_schema.validate(df, lazy=True)
