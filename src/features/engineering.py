"""
Feature engineering for the sales forecasting system.

All features are created PER STATE — never leaking across state boundaries.
Lag and rolling features use only past data (shift(n) / rolling(n).shift(1)).
"""

import holidays as hol
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Individual feature groups
# ---------------------------------------------------------------------------


def _add_lag_features(grp: pd.DataFrame, lag_periods: list[int]) -> pd.DataFrame:
    """Lag the target column by n weeks. lag_1 at row t = total at row t-1."""
    for n in lag_periods:
        grp[f"lag_{n}"] = grp["total"].shift(n)
    return grp


def _add_rolling_features(
    grp: pd.DataFrame,
    windows: list[int],
    stats: list[str],
) -> pd.DataFrame:
    """
    Rolling stats computed BEFORE shifting so they use only past data.
    rolling(w).shift(1) gives a window ending at t-1 for the value at t.
    min_periods=1 avoids NaN for the warm-up rows.
    """
    for w in windows:
        for stat in stats:
            col = f"rolling_{stat}_{w}"
            if stat == "mean":
                grp[col] = grp["total"].rolling(w, min_periods=1).mean().shift(1)
            elif stat == "std":
                grp[col] = grp["total"].rolling(w, min_periods=1).std().shift(1)
    return grp


def _add_calendar_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month, quarter, week-of-year, month boundaries, year."""
    dt = grp["date"]
    grp["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    grp["month"] = dt.dt.month
    grp["quarter"] = dt.dt.quarter
    grp["year"] = dt.dt.year
    grp["dayofweek"] = dt.dt.dayofweek  # 0=Mon … 6=Sun
    grp["is_month_start"] = dt.dt.is_month_start.astype(int)
    grp["is_month_end"] = dt.dt.is_month_end.astype(int)
    return grp


def _add_holiday_features(grp: pd.DataFrame, country: str = "US") -> pd.DataFrame:
    """
    US holiday flags and days-to/from nearest holiday.
    Computed once for the full date range in the group.
    """
    years = grp["date"].dt.year.unique().tolist()
    us_holidays = hol.country_holidays(country, years=years)
    holiday_dates = pd.to_datetime(sorted(us_holidays.keys()))

    def _days_to_next(d: pd.Timestamp) -> int:
        future = holiday_dates[holiday_dates >= d]
        return int((future[0] - d).days) if len(future) else 365

    def _days_from_last(d: pd.Timestamp) -> int:
        past = holiday_dates[holiday_dates <= d]
        return int((d - past[-1]).days) if len(past) else 365

    grp["is_holiday"] = grp["date"].isin(holiday_dates).astype(int)
    grp["days_to_next_holiday"] = grp["date"].apply(_days_to_next)
    grp["days_from_last_holiday"] = grp["date"].apply(_days_from_last)
    return grp


def _add_trend_feature(grp: pd.DataFrame) -> pd.DataFrame:
    """Integer index incrementing per week — captures overall linear trend."""
    grp["linear_trend"] = np.arange(len(grp))
    return grp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build all features for the weekly-aggregated DataFrame.

    Input:  DataFrame with columns [state, date, total, category] (weekly)
    Output: DataFrame with all features appended, sorted by state → date.

    Leakage prevention:
      - Lag features: shift(n) — value at t uses only data from t-n
      - Rolling features: rolling(w).shift(1) — window ends at t-1
      - No future information used anywhere
      - NaN rows from warm-up period are NOT dropped here; training code
        drops them only from the training split, keeping test rows intact
    """
    lag_periods: list[int] = config.get("lag_periods", [1, 7, 14, 30])
    rolling_windows: list[int] = config.get("rolling_windows", [7, 14, 30])
    rolling_stats: list[str] = config.get("rolling_stats", ["mean", "std"])
    holiday_country: str = config.get("holiday_country", "US")

    logger.info(
        f"[Features] Building features | lags={lag_periods} | "
        f"rolling_windows={rolling_windows} | stats={rolling_stats}"
    )

    result_groups: list[pd.DataFrame] = []

    for state, grp in df.groupby("state"):
        grp = grp.sort_values("date").copy()
        grp = _add_lag_features(grp, lag_periods)
        grp = _add_rolling_features(grp, rolling_windows, rolling_stats)
        grp = _add_calendar_features(grp)
        grp = _add_holiday_features(grp, country=holiday_country)
        grp = _add_trend_feature(grp)
        result_groups.append(grp)

    out = pd.concat(result_groups, ignore_index=True)
    out = out.sort_values(["state", "date"]).reset_index(drop=True)

    n_features = len(out.columns) - 4  # exclude state, date, total, category
    n_nan_rows = out[out.isnull().any(axis=1)].shape[0]
    logger.info(
        f"[Features] Done | {n_features} features added | "
        f"{n_nan_rows} warm-up rows have NaN (will be dropped in training split only)"
    )

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of engineered feature column names (excludes state, date, total, category)."""  # noqa: E501
    exclude = {"state", "date", "total", "category"}
    return [c for c in df.columns if c not in exclude]


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any feature is NaN. Call only on the TRAINING split."""
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.debug(f"[Features] Dropped {dropped} warm-up rows with NaN features")
    return df
