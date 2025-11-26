import os
import pandas as pd
import subprocess
from hoodini.utils.core import console, to_fasta

def run_defensefinder(all_gff, all_prots, output):
    console.print("🛡️\tRunning DefenseFinder...")
    try:
        temp_gff = all_gff.copy()
        temp_gff["id"] = temp_gff["attributes"].str.extract(r'ID=([^;]+)')[0]
        temp_gff = temp_gff.merge(
            all_prots[["id", "unique_id", "sequence"]],
            on="id", 
            how="left"
        )
        temp_gff = temp_gff[
            ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "id", "sequence"]
        ].reset_index()
        temp_gff["temp_index"] = temp_gff.index.astype(str)
        temp_gff["attributes"] = temp_gff["seqid"] + "_" + temp_gff["temp_index"]
        temp_gff.to_fasta("attributes", "sequence", f"{output}/proteome.fasta")
        command = [
            "defense-finder",
            "run", f"{output}/proteome.fasta",
            "-a",
            "--db-type", "gembase",
            "-o", f"{output}/defense_finder",
        ]
        subprocess.run(command, check=True)
        result_file = f"{output}/defense_finder/proteome_defense_finder_genes.tsv"
        deffinder_df = None
        if os.path.exists(result_file):
            deffinder_df = pd.read_csv(result_file, sep="\t")
            if not deffinder_df.empty:
                deffinder_df["temp_index"] = deffinder_df["hit_id"].str.split("_").str[-1]
                deffinder_df["deffinder_type"] = deffinder_df["type"]
                deffinder_df["deffinder_subtype"] = deffinder_df["subtype"]
                deffinder_df.loc[deffinder_df["deffinder_type"] == "Cas", "deffinder_gene"] = deffinder_df["gene_name"].str.split("_").str[0]
                deffinder_df.loc[deffinder_df["deffinder_type"] != "Cas", "deffinder_gene"] = deffinder_df["gene_name"].str.split("__").str[1]
                temp_gff = temp_gff.merge(deffinder_df, on="temp_index", how="left")
                deffinder_df = temp_gff[["id", "deffinder_subtype", "deffinder_type", "deffinder_gene"]].dropna()
        else:
            print("DefenseFinder output not found")
    except Exception as e:
        print(f"DefenseFinder failed: {e}")
    if deffinder_df is not None:
        return deffinder_df
