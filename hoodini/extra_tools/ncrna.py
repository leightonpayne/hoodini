import os
import pandas as pd
import subprocess
from Bio import SeqIO
from importlib.resources import files
from hoodini.utils.core import console

def run_ncrna(all_neigh, den_data, output, num_threads, valid_unique_ids):
    console.print("🔬\tRunning Infernal for ncRNA annotation...")
    ncrna_dir = f"{output}/ncrna"
    if not os.path.exists(ncrna_dir):
        os.makedirs(ncrna_dir)
    cm_path = files('hoodini').joinpath('data', 'all.cm')
    stockholm_file = f"{ncrna_dir}/results.sto"
    tblout_file = f"{ncrna_dir}/results.txt"
    command = [
        "cmsearch",
        "--tblout", tblout_file,
        "-A", stockholm_file,
        "-E", "0.1", "--incE", "0.1",
        "--cpu", str(num_threads),
        cm_path, f"{output}/neighborhood/neighborhoods.fasta"
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    column_names = [
        'nucid', '-', 'nc_feature', "--", "cm", "mdlfrom", "mdlto", "seqfrom",
        "seqto", "strand_ncrna", "trunc", "pass", "gc", "bias", "score",
        "E-value", "inc", "desc"
    ]
    if os.path.getsize(stockholm_file) > 0:
        cmdf = pd.read_csv(
            tblout_file, sep=r'\s+', engine='python', comment="#",
            header=None, names=column_names
        )
        for record in SeqIO.parse(stockholm_file, "stockholm"):
            seqfrom = int(record.id.split("/")[1].split("-")[0])
            seqto = int(record.id.split("/")[1].split("-")[1])
            seqid = record.id.split("/")[0]
            sequence = str(record.seq).replace(".", "").replace("-", "")
            mask = (cmdf['nucid'] == seqid) & (cmdf['seqfrom'] == seqfrom) & (cmdf['seqto'] == seqto)
            cmdf.loc[mask, "sequence"] = sequence
        valid = all_neigh[all_neigh["unique_id"].isin([str(n) for n in valid_unique_ids])][
            ["seqid", "start_target", "end_target", "start_win", "end_win", "strand_win", "unique_id", "length", "temp_seqid"]
        ]
        print(cmdf)
        cmdf = cmdf.merge(valid, left_on="nucid", right_on="temp_seqid", how="left")
        cmdf["start"] = cmdf["seqfrom"] + cmdf["start_win"]
        cmdf["end"] = cmdf["seqto"] + cmdf["start_win"]
        cmdf["nucid"] = cmdf["nucid"].replace(valid["temp_seqid"].tolist(), valid["seqid"].tolist())
        cmdf["nc_feature"] = cmdf["nc_feature"]
        cmdf["unique_id"] = cmdf["unique_id"].astype(str)
        cmdf.to_csv(f"{ncrna_dir}/ncrna_results.tsv", sep="\t", index=False)
        return cmdf
    
    else:
        console.print(f"[yellow]⚠️  No ncRNA found by Infernal (empty {stockholm_file})[/yellow]")
        empty_df = pd.DataFrame()
        empty_df.to_csv(f"{ncrna_dir}/ncrna_results.tsv", sep="\t", index=False)
        return empty_df
