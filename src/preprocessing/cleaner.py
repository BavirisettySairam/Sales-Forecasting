"""
Data cleaning pipeline for the sales forecasting dataset.

Raw CSV quirks handled here:
  - Total column has comma-separated thousands and whitespace: "  109,574,036 "
  - Date format is M/D/YYYY (e.g. 1/12/2019), not ISO 8601
  - Category column is dropped on load (single constant value, unused)
"""

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Step 1: Load & normalise columns
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    """Read the CSV, drop the Category column, and normalise remaining columns."""
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {len(df)} rows, columns={list(df.columns)}")

    df.columns = df.columns.str.strip()

    # Drop Category — single constant value ("Beverages"), adds no signal
    if "Category" in df.columns:
        df = df.drop(columns=["Category"])
        logger.info("Dropped 'Category' column (constant, unused)")

    # Normalise Total: strip commas and whitespace, cast to float
    df["Total"] = (
        df["Total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

    # Parse dates — dataset uses mixed formats (M/D/YYYY and D-MM-YYYY)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)

    df = df.rename(columns={"State": "state", "Date": "date", "Total": "total"})

    df["state"] = df["state"].str.strip()
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
# Step 3: Aggregate duplicate (state, date) combinations
# ---------------------------------------------------------------------------

def aggregate_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Sum Total for rows sharing the same state + date."""
    before = len(df)
    df = df.groupby(["state", "date"], as_index=False).agg({"total": "sum"})
    dropped = before - len(df)
    if dropped:
        logger.info(f"Aggregated {dropped} duplicate (state, date) rows")
    return df


# ---------------------------------------------------------------------------
# Step 4: Fill missing dates per state
# ---------------------------------------------------------------------------

def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """For each state, create a complete daily date range and expose gaps."""
    groups = []
    for state, grp in df.groupby("state"):
        full_range = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        scaffold = pd.DataFrame({"date": full_range, "state": state})
        merged = scaffold.merge(grp[["date", "total"]], on="date", how="left")
        groups.append(merged)

    result = pd.concat(groups, ignore_index=True)
    gaps = result["total"].isna().sum()
    if gaps:
        logger.info(f"Identified {gaps} missing date gaps across all states")
    return result


# ---------------------------------------------------------------------------
# Step 5: Impute missing values
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    """Fill missing Total values per state."""

    def _fill(series: pd.Series) -> pd.Series:
        if method == "interpolate":
            return series.interpolate(method="linear", limit_direction="both")
        elif method == "ffill":
            return series.ffill().bfill()
        elif method == "bfill":
            return series.bfill().ffill()
        return series

    df = df.copy()
    df["total"] = df.groupby("state")["total"].transform(_fill)
    remaining = df["total"].isna().sum()
    if remaining:
        logger.warning(f"{remaining} nulls remain after imputation — filling with 0")
        df["total"] = df["total"].fillna(0)
    logger.info(f"Missing values imputed (method='{method}')")
    return df


# ---------------------------------------------------------------------------
# Step 6: Outlier detection — cap at IQR fences per state
# ---------------------------------------------------------------------------

def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Flag and cap outliers per state using IQR. Never silently drops rows."""
    df = df.copy()
    df["is_outlier"] = False
    total_flagged = 0

    for state, idx in df.groupby("state").groups.items():
        vals = df.loc[idx, "total"]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (vals < lower) | (vals > upper)
        df.loc[idx[mask], "is_outlier"] = True
        total_flagged += mask.sum()
        df.loc[idx, "total"] = vals.clip(lower=max(0, lower), upper=upper)

    logger.info(
        f"Outlier detection ({method}, threshold={threshold}): "
        f"{total_flagged} values flagged and capped"
    )
    return df


# ---------------------------------------------------------------------------
# Step 7: Aggregate to weekly (Monday-anchored, sum of Total)
# ---------------------------------------------------------------------------

def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly (W-MON) by summing Total per state."""
    df = df.copy().drop(columns=["is_outlier"], errors="ignore")
    df["date"] = pd.to_datetime(df["date"])

    weekly_groups = []
    for state, grp in df.groupby("state"):
        grp = grp.set_index("date").sort_index()
        weekly = grp["total"].resample("W-MON", label="left", closed="left").sum()
        wdf = weekly.reset_index()
        wdf.columns = ["date", "total"]
        wdf["state"] = state
        weekly_groups.append(wdf)

    result = pd.concat(weekly_groups, ignore_index=True)
    result = result[["state", "date", "total"]]
    logger.info(
        f"Weekly aggregation complete: {len(result)} rows, "
        f"{result['state'].nunique()} states"
    )
    return result


# ---------------------------------------------------------------------------
# Step 8: Sort
# ---------------------------------------------------------------------------

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["state", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    fill_method: str = "interpolate",
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
) -> pd.DataFrame:
    """Run all cleaning steps and return the weekly-aggregated DataFrame."""
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
