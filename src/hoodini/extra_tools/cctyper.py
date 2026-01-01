import os
import polars as pl
import subprocess
from ast import literal_eval
from hoodini.utils.logging_utils import console


def run_cctyper(all_gff, all_prots, all_neigh, output, num_threads, valid_unique_ids):
    console.print("🧬\tRunning CCTyper...")
    temp_gff = all_gff.copy()
    temp_gff["id"] = temp_gff["attributes"].str.extract(r"ID=([^;]+)")[0]
    temp_gff = temp_gff.join(all_prots[["id", "unique_id", "sequence"]], on="id", how="left")
    temp_gff = temp_gff[
        [
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
        ]
    ].copy()
    valid = all_neigh[all_neigh["unique_id"].isin([str(n) for n in valid_unique_ids])][
        ["start_win", "end_win", "temp_seqid", "unique_id"]
    ]
    temp_gff = temp_gff.join(valid, on="unique_id", how="left")
    temp_gff["start"] -= temp_gff["start_win"]
    temp_gff["end"] -= temp_gff["start_win"]
    temp_gff["seqid"] = temp_gff["temp_seqid"]
    temp_gff = temp_gff.drop(["start_win", "end_win", "temp_seqid", "unique_id"])
    temp_gff = temp_gff.drop_duplicates(subset=["attributes", "seqid"])
    temp_gff.write_csv(f"{output}/temp.gff", separator="\t", include_header=False)
    if not os.path.exists(f"{output}/cctyper"):
        os.makedirs(f"{output}/cctyper")
    command = [
        "cctyper",
        "--gff",
        f"{output}/temp.gff",
        "--prot",
        f"{output}/results.fasta",
        "-t",
        str(num_threads),
        f"{output}/neighborhood/neighborhoods.fasta",
        f"{output}/cctyper",
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    # Parse Cas operons
    operon_file = f"{output}/cctyper/cas_operons.tab"
    if os.path.exists(operon_file):
        cctyper_df = pl.read_csv(operon_file, separator="\t")
        cctyper_df["Genes"] = cctyper_df["Genes"].apply(literal_eval)
        cctyper_df["Prot_IDs"] = cctyper_df["Prot_IDs"].apply(literal_eval)
        exploded = {"Genes": [], "Prot_IDs": [], "Best_type": []}
        for row in cctyper_df.iter_rows(named=True):
            for gene, prot in zip(row["Genes"], row["Prot_IDs"]):
                exploded["Genes"].append(gene)
                exploded["Prot_IDs"].append(prot)
                exploded["Best_type"].append(row["Best_type"])
        cctyper_df = pl.DataFrame(exploded).rename(
            columns={"Best_type": "cctyper_system", "Genes": "cctyper_gene", "Prot_IDs": "id"}
        )
    else:
        cctyper_df = None
    # Parse CRISPR arrays
    crispr_path = f"{output}/cctyper/crisprs_all.tab"
    if os.path.exists(crispr_path):
        crispr_df = pl.read_csv(crispr_path, separator="\t", engine="python")
        valid = all_neigh[all_neigh["unique_id"].isin([str(n) for n in valid_unique_ids])][
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
        crispr_df["start"] = crispr_df["Start"] + crispr_df["start_win"]
        crispr_df["end"] = crispr_df["End"] + crispr_df["start_win"]
        crispr_df["Contig"] = crispr_df["Contig"].replace(
            valid["temp_seqid"].to_list(), valid["seqid"].to_list()
        )
        crispr_df["CRISPR"] = crispr_df["CRISPR"].replace(
            valid["temp_seqid"].to_list(), valid["seqid"].to_list()
        )
        crispr_df["nc_feature"] = "CRISPR array " + crispr_df["Subtype"]
        crispr_df["unique_id"] = crispr_df["unique_id"].astype(str)
    else:
        crispr_df = None
    return cctyper_df, crispr_df
