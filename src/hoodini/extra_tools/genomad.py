import os
import polars as pl
import subprocess
from importlib.resources import files
from pathlib import Path

from hoodini.utils.logging_utils import console


def _resolve_genomad_db() -> Path:
    """Return path to geNomad DB in packaged data (no env override).

    Some distributions ship the DB as data/genomad_db/genomad_db, so we check both.
    """
    base = files("hoodini").joinpath("data", "genomad_db")
    nested = base.joinpath("genomad_db")
    return nested if nested.exists() else base


def _ensure_db_exists(db_path: Path):
    version_file = db_path / "version.txt"
    if not version_file.exists():
        msg = (
            "geNomad database not found at "
            f"'{db_path}'. Please download it with:\n"
            "  genomad download-database /path/to/genomad_db\n"
            "and place the resulting folder at that path (packaged data location)."
        )
        raise RuntimeError(msg)


def run_genomad(all_neigh, output, num_threads, valid_unique_ids):
    console.print("🧬\tRunning geNomad...")
    genomad_df = pl.DataFrame()
    genomad_dir = f"{output}/genomad"
    os.makedirs(genomad_dir, exist_ok=True)

    db_path = _resolve_genomad_db()
    _ensure_db_exists(db_path)

    neighborhood_fasta = f"{output}/neighborhood/neighborhoods.fasta"
    genomad_command = [
        "genomad",
        "end-to-end",
        "--cleanup",
        "--splits",
        str(max(1, num_threads or 1)),
        neighborhood_fasta,
        f"{genomad_dir}/output",
        str(db_path),
    ]
    subprocess.run(genomad_command, check=True)
    plasmid_file = f"{genomad_dir}/output/neighborhoods_summary/neighborhoods_plasmid_genes.tsv"
    virus_file = f"{genomad_dir}/output/neighborhoods_summary/neighborhoods_virus_genes.tsv"
    plasmid_prots = (
        pl.read_csv(plasmid_file, separator="\t")
        if os.path.exists(plasmid_file)
        else pl.DataFrame()
    )
    virus_prots = (
        pl.read_csv(virus_file, separator="\t") if os.path.exists(virus_file) else pl.DataFrame()
    )
    # If neither plasmid nor virus result files contain rows, return empty DataFrame
    if plasmid_prots.height == 0 and virus_prots.height == 0:
        return pl.DataFrame()

    if len(plasmid_prots) > 0 or len(virus_prots) > 0:
        plasmid_clean = plasmid_prots.with_columns(
            pl.col("gene").cast(pl.Utf8).str.replace(r"_[^_]+$", "", literal=False).alias("id")
        )
        virus_clean = virus_prots.with_columns(
            pl.col("gene").cast(pl.Utf8).str.replace(r"_[^_]+$", "", literal=False).alias("id")
        )
        # If id contains "|provirus", strip at the pipe
        virus_clean = virus_clean.with_columns(
            pl.when(pl.col("id").str.contains(r"\|provirus", literal=False))
            .then(pl.col("id").str.split("|").list.first())
            .otherwise(pl.col("id"))
            .alias("id")
        )

        def _normalize(df: pl.DataFrame, label: str) -> pl.DataFrame:
            if df.height == 0:
                return pl.DataFrame(
                    schema={"id": pl.Utf8, "start": pl.Int64, "end": pl.Int64, "mge_type": pl.Utf8}
                ).with_columns(pl.lit(label).alias("mge_type"))

            return (
                df.with_columns(
                    [
                        pl.col("id").cast(pl.Utf8),
                        pl.col("start").cast(pl.Int64),
                        pl.col("end").cast(pl.Int64),
                    ]
                )
                .group_by("id")
                .agg(
                    [
                        pl.col("start").min().alias("start"),
                        pl.col("end").max().alias("end"),
                    ]
                )
                .with_columns(pl.lit(label).alias("mge_type"))
            )

        plasmid_ag = _normalize(plasmid_clean, "plasmid")
        virus_ag = _normalize(virus_clean, "virus")
        genomad_prots = pl.concat([plasmid_ag, virus_ag], how="vertical")

        valid_ids = [str(n) for n in valid_unique_ids]
        valid = all_neigh.filter(pl.col("unique_id").cast(pl.Utf8).is_in(valid_ids))[
            [
                "seqid",
                "start_target",
                "end_target",
                "start_win",
                "end_win",
                "strand_win",
                "unique_id",
                "length",
                "temp_seqid",
            ]
        ]
        genomad_prots = genomad_prots.join(valid, left_on="id", right_on="temp_seqid", how="left")
        genomad_prots = genomad_prots.with_columns(
            [
                (pl.col("start") + pl.col("start_win").fill_null(0)).alias("start"),
                (pl.col("end") + pl.col("start_win").fill_null(0)).alias("end"),
                pl.col("unique_id").cast(pl.Utf8).alias("unique_id"),
            ]
        )

        genomad_df = genomad_prots.select(
            [
                "seqid",
                "start",
                "end",
                "mge_type",
                "start_target",
                "end_target",
                "start_win",
                "end_win",
                "strand_win",
                "unique_id",
            ]
        )
    return genomad_df
