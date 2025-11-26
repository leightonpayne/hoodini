import pandas as pd
from hoodini.utils.core import console
from pyblast import BCLine6

def run_blast(all_neigh, output, blast, num_threads, valid_unique_ids):
    if blast:
        console.print("🔍\tRunning BLAST annotation...")
        neighborhood_fasta = f"{output}/neighborhood/neighborhoods.fasta"
        query = blast
        bcl = BCLine6(
            "blastn",
            query=query,
            subject=neighborhood_fasta,
            outfmt="evalue",
            word_size=8,
            evalue=1e-5,
            dust="no",
            reward=1,
            penalty=-2,
            gapopen=6,
            gapextend=2,
        )
        print(bcl)
        results_blast = bcl.run(ncore=num_threads, chunksize=20, quiet=True)
        if not results_blast.empty:
            valid = all_neigh[
                all_neigh["unique_id"].isin([str(n) for n in valid_unique_ids])
            ][
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
            results_blast = results_blast.merge(
                valid, left_on="sseqid", right_on="temp_seqid", how="left"
            )
            print(results_blast.columns)
            results_blast["start"] = results_blast["sstart"] + results_blast["start_win"]
            results_blast["end"] = results_blast["send"] + results_blast["start_win"]
            results_blast.rename(columns={"qseqid": "nc_feature"}, inplace=True)
            results_blast["nc_feature"] = "BLAST " + results_blast["nc_feature"]
            results_blast["unique_id"] = results_blast["unique_id"].astype(str)

    return results_blast
