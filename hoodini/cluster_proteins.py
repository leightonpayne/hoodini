import os
import subprocess
from io import StringIO
import pandas as pd
from hoodini.utils.core import console, merge_cluster_result
from hoodini.deep_mmseqs import run_mmseqs_clustering
from hoodini.deep_jackhmmer import parallel_jackhmmer


def cluster_proteins(
    all_prots: pd.DataFrame,
    output_dir: str,
    clust_method: str = "deepmmseqs",
    sorfs: bool = False,
) -> pd.DataFrame:
    """
    Cluster neighbor proteins and annotate GFF with fam_cluster.

    Parameters:
    - all_prots: pd.DataFrame with at least columns: 'id', 'sequence'
    - output_dir: path to output folder
    - clust_method: one of ['diamond_deepclust', 'deepmmseqs', 'jackhmmer']
    - sorfs: whether to filter based on sORF/orfipy rules

    Returns:
    - pd.DataFrame with fam_cluster annotations
    """
    os.makedirs(output_dir, exist_ok=True)

    faa_path = os.path.join(output_dir, "results.fasta")
    
    # Generate FASTA file from all_prots using pandas extension
    all_prots[["id", "sequence"]].dropna().drop_duplicates("id").to_fasta("id", "sequence", faa_path)

    if clust_method == "diamond_deepclust":
        cmd = ["diamond", "deepclust", "-d", faa_path, "--member-cover", "0.8"]
        result = subprocess.run(cmd, check=True, capture_output=True)
        clusterdf = pd.read_csv(
            StringIO(result.stdout.decode("utf-8")),
            sep="\t", names=["clu_rep_seq", "member"], header=None
        )

    elif clust_method == "deepmmseqs":
        tmp_dir = os.path.join(output_dir, "tmp_mmseqs")
        output_file = os.path.join(output_dir, "deepmmseqs_results.tsv")
        os.makedirs(tmp_dir, exist_ok=True)
        run_mmseqs_clustering(
            faa_path, tmp_dir,
            max_steps=5, sensitivity=15,
            cluster_mode=1, cluster_steps=9,
            cov_mode=0, coverage=0.7,
            output=output_file
        )
        clusterdf = pd.read_csv(
            output_file, sep="\t", names=["clu_rep_seq", "member"], header=None
        )

    elif clust_method == "jackhmmer":
        clusterdf = parallel_jackhmmer(faa_path)

    elif clust_method == "blastp":
        # make blast db
        db_prefix = os.path.join(output_dir, "blastdb")
        subprocess.run(["makeblastdb", "-in", faa_path, "-dbtype", "prot", "-out", db_prefix], check=True)
        # run blastp all-vs-all
        blast_out = os.path.join(output_dir, "blastp_results.tsv")
        cmd = ["blastp", "-query", faa_path, "-db", db_prefix, "-outfmt", "6 qseqid sseqid bitscore evalue pident length qcovs", "-out", blast_out]
        subprocess.run(cmd, check=True)
        # parse results
        df_blast = pd.read_csv(blast_out, sep="\t", names=["qseqid","sseqid","bitscore","evalue","pident","length","qcovs"])
        # remove self-hits
        df_blast = df_blast[df_blast["qseqid"] != df_blast["sseqid"]]
        # build clusters via union-find
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        ids = set(df_blast["qseqid"]).union(df_blast["sseqid"])
        for i in ids:
            parent[i] = i
        for _, row in df_blast.iterrows():
            union(row["qseqid"], row["sseqid"])
        comps = {}
        for i in ids:
            r = find(i)
            comps.setdefault(r, []).append(i)
        records = []
        for members in comps.values():
            rep = sorted(members)[0]
            for m in members:
                records.append((rep, m))
        clusterdf = pd.DataFrame(records, columns=["clu_rep_seq", "member"])

    else:
        raise ValueError(f"Unsupported clustering method: {clust_method}")
    
    clusterdf['clu_size'] = clusterdf['clu_rep_seq'].map(clusterdf['clu_rep_seq'].value_counts())
    rep_to_fam = (
        clusterdf.loc[clusterdf['clu_size'] >= 2]
        .drop_duplicates(subset='clu_rep_seq')
        .assign(fam_cluster=lambda df: df['clu_size'].rank(method='first', ascending=False).astype(int).astype(str))
        .loc[:, ['clu_rep_seq', 'fam_cluster']]
    )
    clusterdf = clusterdf.merge(rep_to_fam, on='clu_rep_seq', how='left')  
    if sorfs:
        clusterdf = clusterdf.loc[~(
            clusterdf["clu_rep_seq"].str.contains("sORF") & clusterdf["fam_cluster"].isnull()
        )]
        clusterdf = clusterdf.groupby("fam_cluster").filter(
            lambda x: not all(x["member"].str.contains("orfipy"))
        )
    all_prots = all_prots.merge(
        clusterdf[["member","fam_cluster"]],
        left_on="id",
        right_on="member",
        how="left"
    ).drop(columns=["member"])

    console.print("✔️\tProtein clustering complete")
    
    return all_prots
