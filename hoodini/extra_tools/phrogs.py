import os
import collections
import polars as pl
import pyhmmer
from pyhmmer import easel
from pyhmmer.plan7 import HMMFile

def run_phrogs(all_gff, output, num_threads):
    hmm_path = os.path.join(os.environ.get("CONDA_PREFIX"), "db/metacerberus/PHROG.hmm")
    metadata_phrogs = os.path.join(os.environ.get("CONDA_PREFIX"), "db/metacerberus/PHROG.tsv")
    alphabet = pyhmmer.easel.Alphabet.amino()
    with HMMFile(hmm_path) as hmm_file:
        hmms = list(hmm_file)
    with easel.SequenceFile(f"{output}/results.fasta", digital=True) as seq_file:
        sequences = list(seq_file)
    Result = collections.namedtuple("Result", ["id", "phrog", "phrog_bitscore", "phrog_evalue"])
    results_phrogs = []
    for hits in pyhmmer.hmmsearch(hmms, sequences, cpus=num_threads, E=1e-5):
        hmm = hits.query_name.decode()
        for hit in hits:
            if hit.included:
                results_phrogs.append(Result(hit.name.decode(), hmm, hit.score, hit.evalue))
    if results_phrogs:
        df_phrogs = pl.DataFrame(results_phrogs)
        df_phrogs = df_phrogs.sort_values(by=["id", "phrog_bitscore"], ascending=False)
        metadata_df = pl.read_csv(metadata_phrogs, separator="\t", names=["phrog_cat", "phrog", "phrog_gene", "EC"])
        df_phrogs = df_phrogs.join(metadata_df[["phrog_cat", "phrog", "phrog_gene"]], on="phrog", how="left")
        color_map = {
            "DNA, RNA and nucleotide metabolism": [76, 114, 176, 255],
            "connector": [221, 132, 82, 255],
            "head and packaging": [204, 185, 116, 255],
            "integration and excision": [85, 168, 104, 255],
            "lysis": [196, 78, 82, 255],
            "moron, auxiliary metabolic gene and host takeover": [129, 114, 179, 255],
            "tail": [147, 120, 96, 255],
            "transcription regulation": [218, 139, 195, 255],
            "other": [0, 0, 0, 100],
            "unknown function": [0, 0, 0, 100]
        }
        all_gff = all_gff.join(df_phrogs, on="id", how="left")
        all_gff["linecolor"] = all_gff.apply(
            lambda x: color_map.get(x["phrog_cat"], x["linecolor"]) if pl.notna(x["phrog_cat"]) else x["linecolor"],
            axis=1,
        )
    return all_gff
