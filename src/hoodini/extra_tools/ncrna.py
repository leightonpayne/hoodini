import subprocess
from importlib.resources import files
from pathlib import Path

import polars as pl
from Bio import SeqIO

from hoodini.utils.logging_utils import info, warn

def run_ncrna(all_neigh, den_data, output, num_threads, valid_unique_ids):
    info("🔬\tRunning Infernal for ncRNA annotation...")
    output = Path(output)
    ncrna_dir = output / "ncrna"
    ncrna_dir.mkdir(parents=True, exist_ok=True)
    cm_path = files("hoodini").joinpath("data", "all.cm")
    stockholm_file = ncrna_dir / "results.sto"
    tblout_file = ncrna_dir / "results.txt"
    command = [
        "cmsearch",
        "--tblout",
        str(tblout_file),
        "-A",
        str(stockholm_file),
        "-E",
        "0.1",
        "--incE",
        "0.1",
        "--cpu",
        str(num_threads),
        cm_path,
        str(output / "neighborhood" / "neighborhoods.fasta"),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    column_names = [
        "nucid",
        "-",
        "nc_feature",
        "--",
        "cm",
        "mdlfrom",
        "mdlto",
        "seqfrom",
        "seqto",
        "strand_ncrna",
        "trunc",
        "pass",
        "gc",
        "bias",
        "score",
        "E-value",
        "inc",
        "desc",
    ]
    if stockholm_file.stat().st_size > 0:
        cmdf = pl.read_csv(
            tblout_file,
            separator=r"\s+",
            engine="python",
            comment="#",
            header=None,
            names=column_names,
        )
        for record in SeqIO.parse(stockholm_file, "stockholm"):
            seqfrom = int(record.id.split("/")[1].split("-")[0])
            seqto = int(record.id.split("/")[1].split("-")[1])
            seqid = record.id.split("/")[0]
            sequence = str(record.seq).replace(".", "").replace("-", "")
            mask = (
                (cmdf["nucid"] == seqid) & (cmdf["seqfrom"] == seqfrom) & (cmdf["seqto"] == seqto)
            )
            cmdf.loc[mask, "sequence"] = sequence
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
        info(f"Parsed {len(cmdf)} ncRNA hits from Infernal.")
        cmdf = cmdf.join(valid, left_on="nucid", right_on="temp_seqid", how="left")
        cmdf["start"] = cmdf["seqfrom"] + cmdf["start_win"]
        cmdf["end"] = cmdf["seqto"] + cmdf["start_win"]
        cmdf["nucid"] = cmdf["nucid"].replace(
            valid["temp_seqid"].to_list(), valid["seqid"].to_list()
        )
        cmdf["nc_feature"] = cmdf["nc_feature"]
        cmdf["unique_id"] = cmdf["unique_id"].astype(str)
        cmdf.write_csv(ncrna_dir / "ncrna_results.tsv", separator="\t", include_header=False)
        return cmdf

    else:
        warn(f"No ncRNA found by Infernal (empty {stockholm_file})")
        empty_df = pl.DataFrame()
        empty_df.write_csv(ncrna_dir / "ncrna_results.tsv", separator="\t", include_header=False)
        return empty_df
