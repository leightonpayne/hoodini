import os
import pandas as pd
import subprocess
from hoodini.utils.core import console
from importlib.resources import files

genomad_db = files("hoodini").joinpath("data", "genomad_db")

def run_genomad(all_neigh, output, num_threads, valid_unique_ids):
    console.print("🧬\tRunning GenoMAD...")
    genomad_df = pd.DataFrame()
    genomad_dir = f"{output}/genomad"
    if not os.path.exists(genomad_dir):
        os.makedirs(genomad_dir)
    neighborhood_fasta = f"{output}/neighborhood/neighborhoods.fasta"
    genomad_command = [
        "genomad", "end-to-end", "--cleanup",
        "--splits", "12",
        neighborhood_fasta,
        f"{genomad_dir}/output",
        str(genomad_db),
    ]
    subprocess.run(genomad_command, check=True)
    plasmid_file = f"{genomad_dir}/output/neighborhoods_summary/neighborhoods_plasmid_genes.tsv"
    virus_file = f"{genomad_dir}/output/neighborhoods_summary/neighborhoods_virus_genes.tsv"
    plasmid_prots = pd.read_csv(plasmid_file, sep="\t") if os.path.exists(plasmid_file) else pd.DataFrame()
    virus_prots = pd.read_csv(virus_file, sep="\t") if os.path.exists(virus_file) else pd.DataFrame()
    # If neither plasmid nor virus result files contain rows, return empty DataFrame
    if plasmid_prots.empty and virus_prots.empty:
        return pd.DataFrame()

    if len(plasmid_prots) > 0 or len(virus_prots) > 0:
        plasmid_prots["id"] = plasmid_prots["gene"].str.rsplit("_", n=1).str[0]
        virus_prots["id"] = virus_prots["gene"].str.rsplit("_", n=1).str[0]
        virus_prots.loc[virus_prots["id"].str.contains(r"\\|provirus"), "id"] = (
            virus_prots["id"].str.rsplit("|").str[0]
        )
        plasmid_prots = plasmid_prots.groupby("id").agg({"start": "min", "end": "max"}).reset_index()
        virus_prots = virus_prots.groupby("id").agg({"start": "min", "end": "max"}).reset_index()
        plasmid_prots["mge_type"] = "plasmid"
        virus_prots["mge_type"] = "virus"
        genomad_prots = pd.concat([plasmid_prots, virus_prots])
        valid = all_neigh[all_neigh["unique_id"].isin([str(n) for n in valid_unique_ids])][
            ["seqid", "start_target", "end_target", "start_win", "end_win", "strand_win", "unique_id", "length", "temp_seqid"]
        ]
        genomad_prots = genomad_prots.merge(valid, left_on="id", right_on="temp_seqid", how="left")
        genomad_prots["start"] += genomad_prots["start_win"]
        genomad_prots["end"] += genomad_prots["start_win"]
        genomad_prots["unique_id"] = genomad_prots["unique_id"].astype(str)
        genomad_df = genomad_prots[
            ['seqid', 'start', 'end', 'mge_type', 'start_target', 'end_target', 'start_win', 'end_win', 'strand_win', 'unique_id']
        ]
        print(genomad_df)
    return genomad_df
