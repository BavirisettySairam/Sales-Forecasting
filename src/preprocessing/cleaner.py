"""
Data cleaning pipeline for the sales forecasting dataset.

Raw CSV quirks this module handles:
  - Total column has comma thousands separators and surrounding whitespace: "  109,574,036 "
  - Date format is M/D/YYYY (e.g. 1/12/2019), not ISO 8601
  - Multiple categories per state/date combination
"""

import pandas as pd
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Step 1: Load & normalise columns
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    """Read the CSV and return a DataFrame with normalised column names + types."""
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {len(df)} rows, {list(df.columns)}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalise Total: remove whitespace and commas, cast to float
    df["Total"] = (
        df["Total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

    # Parse dates — dataset has mixed formats: M/D/YYYY and D-MM-YYYY
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)

    # Lowercase column names for internal consistency
    df = df.rename(columns={
        "State": "state",
        "Date": "date",
        "Total": "total",
        "Category": "category",
    })

    # Strip whitespace in string columns
    df["state"] = df["state"].str.strip()
    df["category"] = df["category"].str.strip()

    return df


# ---------------------------------------------------------------------------
# Step 2: Remove exact duplicates
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info(f"Removed {dropped} exact duplicate rows")
    return df


# ---------------------------------------------------------------------------
# Step 3: Aggregate duplicate (state, date, category) combinations
# ---------------------------------------------------------------------------

def aggregate_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Sum Total for rows sharing the same state + date + category."""
    before = len(df)
    df = (
        df.groupby(["state", "date", "category"], as_index=False)
        .agg({"total": "sum"})
    )
    dropped = before - len(df)
    if dropped:
        logger.info(f"Aggregated {dropped} duplicate (state, date, category) rows")
    return df


# ---------------------------------------------------------------------------
# Step 4: Fill missing dates per state+category
# ---------------------------------------------------------------------------

def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (state, category) group, create a complete daily date range
    from the group's min to max date, then merge to expose gaps.
    """
    groups = []
    for (state, category), grp in df.groupby(["state", "category"]):
        full_range = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        scaffold = pd.DataFrame({"date": full_range})
        scaffold["state"] = state
        scaffold["category"] = category
        merged = scaffold.merge(grp[["date", "total"]], on="date", how="left")
        groups.append(merged)

    result = pd.concat(groups, ignore_index=True)
    gaps = result["total"].isna().sum()
    if gaps:
        logger.info(f"Identified {gaps} missing date gaps across all state/category groups")
    return result


# ---------------------------------------------------------------------------
# Step 5: Impute missing values
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    """
    Fill missing Total values per (state, category) group.
    method: 'interpolate' (linear), 'ffill', or 'bfill'
    """
    def _fill(grp: pd.DataFrame) -> pd.DataFrame:
        if method == "interpolate":
            grp["total"] = grp["total"].interpolate(method="linear", limit_direction="both")
        elif method == "ffill":
            grp["total"] = grp["total"].ffill().bfill()
        elif method == "bfill":
            grp["total"] = grp["total"].bfill().ffill()
        return grp

    df = df.groupby(["state", "category"], group_keys=False).apply(_fill)
    remaining_nulls = df["total"].isna().sum()
    if remaining_nulls:
        logger.warning(f"{remaining_nulls} nulls remain after imputation — filling with 0")
        df["total"] = df["total"].fillna(0)
    logger.info(f"Missing values imputed using method='{method}'")
    return df


# ---------------------------------------------------------------------------
# Step 6: Outlier detection (IQR) — flag and cap, never silently drop
# ---------------------------------------------------------------------------

def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Flag outliers per (state, category) group using IQR.
    Caps extreme outliers at fence values; logs counts but does not drop rows.
    """
    df = df.copy()
    df["is_outlier"] = False
    total_flagged = 0

    for (state, category), idx in df.groupby(["state", "category"]).groups.items():
        grp_total = df.loc[idx, "total"]
        q1 = grp_total.quantile(0.25)
        q3 = grp_total.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        mask = (grp_total < lower) | (grp_total > upper)
        df.loc[idx[mask], "is_outlier"] = True
        total_flagged += mask.sum()

        # Cap extreme values at the fences
        df.loc[idx, "total"] = grp_total.clip(lower=max(0, lower), upper=upper)

    logger.info(f"Outlier detection ({method}, threshold={threshold}): {total_flagged} values flagged and capped")
    return df


# ---------------------------------------------------------------------------
# Step 7: Aggregate to weekly (Monday-anchored, sum of Total)
# ---------------------------------------------------------------------------

def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to weekly (Monday as week start) by summing Total
    per (state, category). Drops the outlier flag column before aggregation.
    """
    df = df.copy()
    df = df.drop(columns=["is_outlier"], errors="ignore")
    df["date"] = pd.to_datetime(df["date"])

    weekly_groups = []
    for (state, category), grp in df.groupby(["state", "category"]):
        grp = grp.set_index("date").sort_index()
        weekly = grp["total"].resample("W-MON", label="left", closed="left").sum()
        weekly_df = weekly.reset_index()
        weekly_df.columns = ["date", "total"]
        weekly_df["state"] = state
        weekly_df["category"] = category
        weekly_groups.append(weekly_df)

    result = pd.concat(weekly_groups, ignore_index=True)
    result = result[["state", "date", "total", "category"]]
    logger.info(f"Weekly aggregation complete: {len(result)} rows across {result['state'].nunique()} states")
    return result


# ---------------------------------------------------------------------------
# Step 8: Sort
# ---------------------------------------------------------------------------

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["state", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point used by pipeline.py
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    fill_method: str = "interpolate",
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Run all cleaning steps in order and return the weekly-aggregated DataFrame.

    Steps:
      1. remove_duplicates
      2. aggregate_duplicate_dates
      3. fill_missing_dates
      4. impute_missing
      5. handle_outliers
      6. aggregate_to_weekly
      7. sort_data
    """
    logger.info("=== Starting data cleaning ===")
    before = len(df)

    df = remove_duplicates(df)
    df = aggregate_duplicate_dates(df)
    df = fill_missing_dates(df)
    df = impute_missing(df, method=fill_method)
    df = handle_outliers(df, method=outlier_method, threshold=outlier_threshold)
    df = aggregate_to_weekly(df)
    df = sort_data(df)

    logger.info(f"=== Cleaning complete: {before} raw rows → {len(df)} weekly rows ===")
    return df
