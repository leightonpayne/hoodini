import os
import pandas as pd
import subprocess
from hoodini.utils.core import console

def run_padloc(all_gff, all_prots, output, num_threads):
    console.print("🧬\tRunning PADLOC...")


    temp_gff = all_gff.copy()
    temp_gff["id"] = temp_gff["attributes"].str.extract(r'ID=([^;]+)')[0]
    temp_gff = temp_gff.merge(
        all_prots[["id", "unique_id", "sequence"]],
        on="id", 
        how="left"
    )
    temp_gff["temp_id"] = (
        temp_gff["id"] + "-" + temp_gff["unique_id"].astype(str) + "-" + temp_gff["start"].astype(str)
    )
    temp_gff["attributes"] = "ID=" + temp_gff["temp_id"]
    temp_gff["seqid"] = temp_gff["seqid"] + "-" + temp_gff["unique_id"]
    temp_gff = temp_gff.drop_duplicates(subset=["attributes", "seqid"])
    temp_gff = temp_gff.drop_duplicates(subset=["seqid", "start", "end"])
    temp_gff[["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]].to_csv(
        f"{output}/temp.gff", sep="\t", index=False, header=False
    )
    temp_gff[["temp_id", "sequence"]].dropna().drop_duplicates(subset=["temp_id"]).to_fasta(
        "temp_id", "sequence", f"{output}/temp.fasta"
    )
    padloc_dir = f"{output}/padloc"
    if not os.path.exists(padloc_dir):
        os.makedirs(padloc_dir)
    command = [
        "padloc",
        "--faa", f"{output}/temp.fasta",
        "--gff", f"{output}/temp.gff",
        "--cpu", str(num_threads),
        "--outdir", padloc_dir,
    ]
    subprocess.run(command, check=True)
    result_path = f"{padloc_dir}/temp.fasta_padloc.csv"
    if os.path.exists(result_path):
        padloc_df = pd.read_csv(result_path)
        if not padloc_df.empty:
            padloc_df = padloc_df.rename(columns={"system": "padloc_system", "protein.name": "padloc_gene"})
            padloc_df["target.name"] = padloc_df["target.name"].str.rsplit("-", n=2).str[0]
            padloc_df["id"] = padloc_df["target.name"]
            padloc_df = padloc_df[["id", "padloc_system", "padloc_gene"]]
            padloc_df = padloc_df.drop_duplicates(subset=["id"])
    else:
        padloc_df = pd.DataFrame()
       
    return padloc_df
