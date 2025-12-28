import pandas as pd
import polars as pl

import hoodini.initialize as initialize


def test_initialize_inputs_deduplicates_and_marks_premade(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(initialize, "check_assembly_db", lambda: None)

    def fake_read_input_list(path):
        return pd.DataFrame(
            [
                {
                    "og_index": 0,
                    "input_type": "protein",
                    "protein_id": "protA",
                    "gff_path": "/tmp/a.gff",
                    "faa_path": "/tmp/a.faa",
                },
                {
                    "og_index": 0,
                    "input_type": "protein",
                    "protein_id": "protB",
                    "gbf_path": "/tmp/b.gbk",
                },
            ]
        )

    monkeypatch.setattr(initialize, "read_input_list", fake_read_input_list)
    monkeypatch.setattr(initialize, "read_input_sheet", lambda p: pd.DataFrame())
    monkeypatch.setattr(initialize, "uniprot2ncbi", lambda df: df)

    output_dir = tmp_path / "out"

    df = initialize.initialize_inputs(input_path="input.txt", output=output_dir)

    assert output_dir.exists()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert df.select("protein_id").item() == "protA"
    assert df.select("premade").item() is True
