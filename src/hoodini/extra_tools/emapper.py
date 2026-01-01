import subprocess
import polars as pl
from hoodini.utils.logging_utils import console
from importlib.resources import files
from pathlib import Path
from shutil import copyfile
import polars as pl


def run_emapper(all_prots: pl.DataFrame, output: str | Path, num_threads: int = 1) -> pl.DataFrame:
    """
    Run mmseqs easy-search, pick best hit per query directly in Polars,
    join to eggNOG metadata, pick the deepest OG per query,
    and return one row per input protein as a pandas DataFrame.
    """

    console.print("🧾\tRunning eggNOG-mapper (mmseqs + eggNOG, best+deepest OG via Polars) ...")

    output = Path(output)
    emapper_dir = output / "emapper"
    emapper_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output / "results.faa"
    fasta_fallback = output / "results.fasta"

    # write FASTA if missing
    if not fasta_path.exists():
        if fasta_fallback.exists():
            copyfile(fasta_fallback, fasta_path)
            console.print(f"[dim]Copied {fasta_fallback} -> {fasta_path}[/dim]")
        else:
            seq_df = all_prots[["id", "sequence"]].drop_nulls().drop_duplicates("id")
            seq_df.to_fasta("id", "sequence", fasta_path)
            console.print(f"[green]✔ Generated {fasta_path}[/green]")

    # locate mmseqs DB
    mmseqs_dir = files("hoodini").joinpath("data", "emapper", "mmseqs")
    mmseqs_db_padded = str(mmseqs_dir.joinpath("mmseqs.db_pad"))
    mmseqs_db_unpadded = str(mmseqs_dir.joinpath("mmseqs.db"))

    results_m8 = emapper_dir / "results.m8"
    tmpdir = emapper_dir / "mmseqs_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    def run_mmseqs(db_prefix: str, use_gpu: bool):
        cmd = [
            "mmseqs",
            "easy-search",
            str(fasta_path),
            db_prefix,
            str(results_m8),
            str(tmpdir),
            "--threads",
            str(max(1, int(num_threads or 1))),
        ]
        if use_gpu:
            cmd += ["--gpu", "1"]
        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
        subprocess.run(cmd, check=True)

    # try GPU first, fallback CPU
    try:
        run_mmseqs(mmseqs_db_padded, use_gpu=True)
    except subprocess.CalledProcessError:
        console.print("[yellow]GPU search failed — retrying CPU (padded DB)...[/yellow]")
        try:
            run_mmseqs(mmseqs_db_padded, use_gpu=False)
        except subprocess.CalledProcessError:
            console.print("[yellow]CPU (padded DB) failed — trying unpadded DB GPU...[/yellow]")
            try:
                run_mmseqs(mmseqs_db_unpadded, use_gpu=True)
            except subprocess.CalledProcessError:
                console.print("[yellow]GPU (unpadded DB) failed — CPU unpadded DB...[/yellow]")
                run_mmseqs(mmseqs_db_unpadded, use_gpu=False)

    if not results_m8.exists():
        console.print(
            f"[bold yellow]Warning: mmseqs results not found at {results_m8}[/bold yellow]"
        )
        return pl.DataFrame()

    # --- Polars pipeline: read full m8 and pick best hit per query ---
    hits_all = pl.read_csv(
        results_m8,
        has_header=False,
        separator="\t",
        new_columns=[
            "qseqid",
            "sseqid",
            "pident",
            "alnlen",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "sstart",
            "send",
            "evalue",
            "bitscore",
        ],
    )

    # best = max bitscore per qseqid, drop mmseqs columns right away
    hits_best = (
        hits_all.sort(["qseqid", "bitscore"], descending=[False, True])
        .group_by("qseqid")
        .head(1)
        .select(["qseqid", "sseqid"])
    )

    eggnog_prots_path = str(files("hoodini").joinpath("data", "emapper", "eggnog_prots.parquet"))
    eggnog_og_path = str(files("hoodini").joinpath("data", "emapper", "eggnog_og.parquet"))

    prots = (
        pl.scan_parquet(eggnog_prots_path)
        .with_columns(pl.col("ogs").fill_null("").str.split(",").alias("ogs_list"))
        .explode("ogs_list")
        .filter((pl.col("ogs_list") != "") & pl.col("ogs_list").str.contains("@"))
        .with_columns(pl.col("ogs_list").str.split_exact("@", 1).alias("og_split"))
        .with_columns(
            [
                pl.col("og_split").struct.field("field_0").alias("og"),
                pl.col("og_split").struct.field("field_1").cast(pl.Utf8).alias("level"),
            ]
        )
        .drop(["ogs_list", "og_split"])
    )

    hits_prots = hits_best.lazy().join(prots, left_on="sseqid", right_on="name", how="left")

    og = pl.scan_parquet(eggnog_og_path).with_columns(pl.col("level").cast(pl.Utf8))

    annotated = hits_prots.join(og, on=["og", "level"], how="left", suffix="_og").collect()

    # --- Pick deepest OG per query (max taxid) ---
    annotated = annotated.with_columns(pl.col("level").cast(pl.Int64, strict=False))
    annotated = annotated.sort(["qseqid", "level"], descending=[False, True])
    deepest = annotated.group_by("qseqid").head(1)
    deepest = deepest.with_columns(pl.col("level").cast(pl.Utf8))

    # --- Reorder columns ---
    deepest = deepest.rename({"qseqid": "id"})
    exclude_cols = {"level", "nm", "og", "ogs", "orthoindex", "sseqid", "name"}
    lead = ["id", "pname", "description", "COG_categories", "pfam"]
    lead_present = [c for c in lead if c in deepest.columns]
    rest = [c for c in deepest.columns if c not in lead_present and c not in exclude_cols]
    deepest = deepest.select(lead_present + rest)

    console.print("🔎 Head of annotated Polars DF (deepest OG per best hit):")
    console.print(deepest.head(10))
    console.print(f"shape: {deepest.shape}")

    console.print(f"[green]✔ mmseqs annotations ready: {deepest.height} queries annotated[/green]")
    return deepest
