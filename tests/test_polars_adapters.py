import pandas as pd
import polars as pl
import pytest

from hoodini.models.schemas import TableSchema
from hoodini.utils.polars_adapters import ensure_required, rename_if_present, to_pandas, to_polars


@pytest.fixture()
def sample_schema() -> TableSchema:
    return TableSchema(name="dummy", required={"a": pl.Int64, "b": pl.Utf8})


def test_to_polars_from_pandas_enforces_schema(sample_schema: TableSchema) -> None:
    pdf = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df = to_polars(pdf, schema=sample_schema)

    assert isinstance(df, pl.DataFrame)
    assert df.schema == {"a": pl.Int64, "b": pl.Utf8}
    assert df.height == 2


def test_to_polars_raises_on_unsupported_type(sample_schema: TableSchema) -> None:
    with pytest.raises(TypeError):
        to_polars([1, 2, 3], schema=sample_schema)  # type: ignore[arg-type]


def test_to_pandas_from_polars_roundtrip() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    pdf = to_pandas(df)

    assert isinstance(pdf, pd.DataFrame)
    assert list(pdf.columns) == ["a", "b"]
    assert pdf.shape == (2, 2)


def test_ensure_required_detects_missing_columns() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"Missing required columns: \['b'\]"):
        ensure_required(df, ["a", "b"])


def test_rename_if_present_only_renames_existing_columns() -> None:
    df = pl.DataFrame({"old": [1], "keep": [2]})
    renamed = rename_if_present(df, {"old": "new", "missing": "ignored"})

    assert set(renamed.columns) == {"new", "keep"}
    assert renamed.select("new").item() == 1
