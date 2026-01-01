import os
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

import hoodini.cli as cli


@pytest.mark.skipif(
    os.environ.get("HOODINI_RUN_SLOW") != "1",
    reason="Set HOODINI_RUN_SLOW=1 to run integration smoke test",
)
def test_cli_run_smoke(monkeypatch, tmp_path) -> None:
    input_path = Path(__file__).resolve().parents[1] / "example" / "cas9.txt"
    output_dir = tmp_path / "out"

    fake_records = pl.DataFrame(
        {
            "og_index": [0],
            "input_type": ["protein"],
            "protein_id": ["P12345"],
        }
    )

    def fake_initialize_inputs(**kwargs):
        if kwargs.get("output"):
            Path(kwargs["output"]).mkdir(parents=True, exist_ok=True)
        return fake_records

    monkeypatch.setattr("hoodini.pipeline.initialize.initialize_inputs", fake_initialize_inputs)
    monkeypatch.setattr(
        "hoodini.pipeline.parse_ipg.run_ipg", lambda records_df, cand_mode: records_df
    )

    def fake_run_assembly_parser(**kwargs):
        return {
            "records": fake_records,
            "all_gff": pl.DataFrame(),
            "all_prots": pl.DataFrame({"id": ["protA"], "sequence": ["M"]}),
            "all_neigh": pl.DataFrame(
                {
                    "seqid": ["contig1"],
                    "start_win": [0],
                    "end_win": [10],
                    "sequence": ["AAAA"],
                    "unique_id": ["0"],
                }
            ),
            "valid_uids": ["0"],
        }

    monkeypatch.setattr(
        "hoodini.pipeline.parse_assemblies.run_assembly_parser", fake_run_assembly_parser
    )
    monkeypatch.setattr(
        "hoodini.pipeline.cluster_proteins.cluster_proteins", lambda all_prots, **kwargs: all_prots
    )
    monkeypatch.setattr(
        "hoodini.pipeline.taxonomy.parse_taxonomy_and_build_tree",
        lambda **kwargs: ("(A);", pl.DataFrame()),
    )
    monkeypatch.setattr("hoodini.pipeline.write_data.write_viz_outputs", lambda **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--force",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert output_dir.exists()
