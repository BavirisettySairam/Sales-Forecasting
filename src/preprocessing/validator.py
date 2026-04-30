import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Raw schema — validated after load_raw() normalises column names
raw_schema = DataFrameSchema(
    columns={
        "state": Column(str, nullable=False),
        "date": Column(pa.DateTime, nullable=False),
        "total": Column(float, checks=Check.ge(0), nullable=False),
    },
    coerce=True,
    strict=False,  # allow extra columns (e.g. is_outlier) to pass through
)

# Post-cleaning schema — enforces no nulls, non-negative totals
clean_schema = DataFrameSchema(
    columns={
        "state": Column(str, nullable=False, checks=Check.str_length(min_value=2)),
        "date": Column(pa.DateTime, nullable=False),
        "total": Column(float, nullable=False, checks=[Check.ge(0)]),
    },
    coerce=True,
    strict=False,
)


def validate_raw(df) -> None:
    raw_schema.validate(df, lazy=True)


def validate_clean(df) -> None:
    clean_schema.validate(df, lazy=True)
