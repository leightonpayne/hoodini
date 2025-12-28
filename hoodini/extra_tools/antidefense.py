import collections
import polars as pl
import pyhmmer
from pyhmmer import easel
from pyhmmer.plan7 import HMMFile
from importlib.resources import files

def run_antidefense(all_gff, output, num_threads):
    hmm_path = files('flagcsnap').joinpath('data', 'acr_antidefense.hmm')
    database_path = f"{output}/results.fasta"
    alphabet = pyhmmer.easel.Alphabet.amino()
    with HMMFile(hmm_path) as hmm_file:
        hmms = list(hmm_file)
    with easel.SequenceFile(database_path, digital=True) as seq_file:
        sequences = list(seq_file)
    Result = collections.namedtuple("Result", ["id", "anti_name", "anti_bitscore", "anti_evalue"])
    results_anti = []
    for hits in pyhmmer.hmmsearch(hmms, sequences, cpus=num_threads, E=1e-5):
        hmm = hits.query_name.decode()
        for hit in hits:
            if hit.included:
                results_anti.append(Result(hit.name.decode(), hmm, hit.score, hit.evalue))
    if results_anti:
        df_anti = pl.DataFrame(results_anti)
        df_anti = df_anti.sort(["id", "anti_bitscore"], descending=[False, True])
        # left join on id
        all_gff = all_gff.join(df_anti, on="id", how="left")
        # set linecolor to red when anti_name is present
        all_gff = all_gff.with_columns(
            pl.when(pl.col("anti_name").is_not_null())
            .then(pl.lit([255, 0, 0, 255]))
            .otherwise(pl.col("linecolor"))
            .alias("linecolor")
        )
    return all_gff
