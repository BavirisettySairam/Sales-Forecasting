import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Raw schema: validates after load_raw() has normalised column names to lowercase
raw_schema = DataFrameSchema(
    columns={
        "state": Column(str, nullable=False),
        "date": Column(pa.DateTime, nullable=False),
        "total": Column(float, checks=Check.ge(0), nullable=False),
        "category": Column(str, nullable=False),
    },
    coerce=True,
    strict=False,  # allow extra columns (e.g. is_outlier) to pass through
)

# Post-cleaning schema: enforces no nulls, lowercase names, non-negative totals
clean_schema = DataFrameSchema(
    columns={
        "state": Column(str, nullable=False, checks=Check.str_length(min_value=2)),
        "date": Column(pa.DateTime, nullable=False),
        "total": Column(float, nullable=False, checks=[Check.ge(0)]),
        "category": Column(str, nullable=False),
    },
    coerce=True,
    strict=False,
)


def validate_raw(df) -> None:
    """Raise SchemaError if the loaded DataFrame violates the expected raw schema."""
    raw_schema.validate(df, lazy=True)


def validate_clean(df) -> None:
    """Raise SchemaError if the cleaned weekly DataFrame violates the post-clean schema."""  # noqa: E501
    clean_schema.validate(df, lazy=True)
