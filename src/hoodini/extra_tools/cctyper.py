import subprocess
from ast import literal_eval
from pathlib import Path

import polars as pl
from hoodini.utils.logging_utils import info


def run_cctyper(all_gff, all_prots, all_neigh, output, num_threads, valid_unique_ids):
    info("🧬\tRunning CCTyper...")
    output = Path(output)

    temp_gff = all_gff.clone()
    temp_gff = temp_gff.with_columns(
        pl.col("attributes").str.extract(r"ID=([^;]+)", 1).alias("id")
    )

    temp_gff = temp_gff.join(
        all_prots.select([c for c in ["id", "unique_id", "sequence"] if c in all_prots.columns]),
        on="id",
        how="left",
    )

    cols = [
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "unique_id",
        "attributes",
        "start_win",
        "end_win",
        "temp_seqid",
    ]

    valid = all_neigh.filter(pl.col("unique_id").is_in([str(n) for n in valid_unique_ids]))[
        ["start_win", "end_win", "temp_seqid", "unique_id"]
    ]
    temp_gff = temp_gff.join(valid, on="unique_id", how="left")

    temp_gff = temp_gff.with_columns(
        (pl.col("start") - pl.col("start_win")).alias("start"),
        (pl.col("end") - pl.col("start_win")).alias("end"),
        pl.col("temp_seqid").alias("seqid"),
    )

    temp_gff = temp_gff.select([c for c in cols if c in temp_gff.columns])
    temp_gff = temp_gff.unique(subset=["attributes", "seqid"])

    temp_gff.write_csv(output / "temp.gff", separator="\t", include_header=False)
    (output / "cctyper").mkdir(parents=True, exist_ok=True)
    command = [
        "cctyper",
        "--gff",
        str(output / "temp.gff"),
        "--prot",
        str(output / "results.fasta"),
        "-t",
        str(num_threads),
        str(output / "neighborhood" / "neighborhoods.fasta"),
        str(output / "cctyper"),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        hint = ""
        if "drawSvg" in err or "drawsvg" in err:
            hint = " (drawSvg may be missing)"
        raise RuntimeError(f"cctyper failed: {err}{hint}") from e

    operon_file = output / "cctyper" / "cas_operons.tab"
    if operon_file.exists():
        cctyper_df = pl.read_csv(operon_file, separator="\t")
        cctyper_df = cctyper_df.with_columns(
            pl.col("Genes").map_elements(literal_eval),
            pl.col("Prot_IDs").map_elements(literal_eval),
        )
        exploded = {"Genes": [], "Prot_IDs": [], "Best_type": []}
        for row in cctyper_df.iter_rows(named=True):
            for gene, prot in zip(row["Genes"], row["Prot_IDs"]):
                exploded["Genes"].append(gene)
                exploded["Prot_IDs"].append(prot)
                exploded["Best_type"].append(row["Best_type"])
        cctyper_df = pl.DataFrame(exploded).rename(
            {"Best_type": "cctyper_system", "Genes": "cctyper_gene", "Prot_IDs": "id"}
        )
    else:
        cctyper_df = pl.DataFrame()

    crispr_path = output / "cctyper" / "crisprs_all.tab"
    if crispr_path.exists():
        crispr_df = pl.read_csv(crispr_path, separator="\t", engine="python")
        valid = all_neigh.filter(pl.col("unique_id").is_in([str(n) for n in valid_unique_ids]))[
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
        crispr_df = crispr_df.join(valid, left_on="Contig", right_on="temp_seqid", how="left")
        crispr_df = crispr_df.with_columns(
            (pl.col("Start") + pl.col("start_win")).alias("start"),
            (pl.col("End") + pl.col("start_win")).alias("end"),
            pl.col("Contig").replace(valid["temp_seqid"].to_list(), valid["seqid"].to_list()),
            pl.col("CRISPR").replace(valid["temp_seqid"].to_list(), valid["seqid"].to_list()),
            (pl.lit("CRISPR array ") + pl.col("Subtype")).alias("nc_feature"),
            pl.col("unique_id").cast(pl.Utf8),
        )
    else:
        crispr_df = pl.DataFrame()

    return cctyper_df, crispr_df
