import os
import sys
import pandas as pd
import numpy as np
import subprocess
from itertools import combinations_with_replacement
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
import taxoniq
from hoodini.utils.core import console, read_fasta
from UniProtMapper import ProtMapper
from alphafetcher import AlphaFetcher
import itertools
from typing import Optional, Iterable
import ete3

sys.setrecursionlimit(3000000)

def parse_taxonomy_and_build_tree(records, all_gff, all_prots, all_neigh, output_dir, tree_mode, tree_file=None, num_threads=4,
                                  pairwise_aai=None, pairwise_ani=None, valid_uids=None,
                                  aai_mode: Optional[str] = None, ani_mode: Optional[str] = None,
                                  aai_subset_mode: Optional[str] = None, nj_algorithm: Optional[str] = None):
    os.makedirs(output_dir, exist_ok=True)

    if tree_mode == "taxonomy":
        tree_str = _make_taxonomic_tree(records)
    elif tree_mode == "fast_nj":
        tree_str = _make_fast_phylo_tree(records, all_prots, output_dir, nj_algorithm)
    elif tree_mode == "aai_tree":
        # build tree from AAI pairwise table
        pairwise_aai
        tree_str = aai_tree(pairwise_aai, valid_uids=valid_uids, algorithm=nj_algorithm, threads=num_threads,
                            mode=aai_mode, subset_mode=aai_subset_mode)
    elif tree_mode == "ani_tree":
        # build tree from ANI pairwise table
        tree_str = ani_tree(pairwise_ani, valid_uids=valid_uids, algorithm=nj_algorithm, threads=num_threads,
                            mode=ani_mode)
    elif tree_mode == "fast_ml":
        tree_str = _make_tree(records, all_prots, output_dir, num_threads)
    elif tree_mode == "use_input_tree":
        with open(tree_file) as f:
            tree_str = f.read()
    elif tree_mode == "foldmason_tree":
        tree_str = _make_foldmason_tree(records, all_prots, output_dir, num_threads)
    elif tree_mode == "neigh_similarity_tree":
        tree_str = _make_neigh_similarity_tree(all_prots)
    elif tree_mode == "neigh_phylo_tree":
        tree_str = _make_neigh_phylo_tree(records, all_prots)
    else:
        raise ValueError(f"Unsupported tree mode: {tree_mode}")

    with open(f"{output_dir}/tree.nwk", "w") as out:
        out.write(tree_str)

    den_data = _build_leaf_metadata(records, all_neigh)
    console.print("\u2714\ufe0f\tTree saved as Newick\n")
    return tree_str, den_data

def _build_leaf_metadata(records, all_neigh):
    
    dicc_tax = {}
    records["taxid"].fillna("32644", inplace=True)
    for taxid in records["taxid"].unique():
        try:
            t = taxoniq.Taxon(taxid)
        except Exception as e:
            t = taxoniq.Taxon(32644)  # Default to Bacteria if error
        dicc_tax[taxid] = {t.rank.name: t.scientific_name for t in t.ranked_lineage}

    taxdf = pd.DataFrame(dicc_tax).T
    records = records.merge(taxdf, left_on="taxid", right_index=True, how="left")
    
    taxcols = ["superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
    for col in taxcols:
        if col not in records.columns:
            records[col] = "unclassified"
    records["unique_id"] = records["unique_id"].astype(str)
    all_neigh["unique_id"] = all_neigh["unique_id"].astype(str)

    den_data = records[["unique_id", "og_index"] + taxcols].copy()
    den_data = den_data.merge(all_neigh[["unique_id", "start_win", "end_win", "strand_win", "start_target", "end_target"]],
                              on="unique_id", how="left")
    return den_data

def _make_tree(records, all_prots, output_dir, threads):
    prots = records.loc[records["failed"].isnull(), "protein_id"].dropna().unique()
    faa = all_prots[all_prots["id"].isin(prots)]
    faa = faa[faa["target_prot"] == faa["id"]].drop_duplicates(subset=["id", "target_prot"])
    faa.to_fasta("unique_id", "sequence", f"{output_dir}/target_prots.fasta")
    subprocess.run(["famsa", f"{output_dir}/target_prots.fasta", f"{output_dir}/target_prots.aln"], check=True)
    result = subprocess.run(["VeryFastTree", "-threads", str(threads), f"{output_dir}/target_prots.aln"], capture_output=True, text=True)
    return result.stdout

def _make_fast_phylo_tree(records, all_prots, output_dir, nj_algorithm):
    from hoodini.decenttree_runner import run_decenttree_from_matrix

    prots = records.loc[records["failed"].isnull(), "protein_id"].dropna().unique()
    faa = all_prots[all_prots["id"].isin(prots)]
    faa = faa[faa["target_prot"] == faa["id"]].drop_duplicates(subset=["id", "target_prot"])
    faa.to_fasta("unique_id", "sequence", f"{output_dir}/target_prots.fasta")
    subprocess.run(["famsa", "-dist_export", "-square_matrix", f"{output_dir}/target_prots.fasta", f"{output_dir}/distance_matrix.csv"], check=True)
    dispd = pd.read_csv(f"{output_dir}/distance_matrix.csv", index_col=0)
    np.fill_diagonal(dispd.values, 0)
    # Use DecentTree for fast phylogenetic tree building from the distance matrix
    newick = run_decenttree_from_matrix(dispd, algorithm=nj_algorithm, threads=1, )
    return newick

def _make_taxonomic_tree(records):
    valid = records[records["failed"].isnull()].reset_index(drop=True)
    taxids = valid["taxid"].dropna().unique().tolist()
    distances = calculate_taxid_distances(taxids, update_db=False)
    dispd = pd.DataFrame(index=valid["unique_id"], columns=valid["unique_id"])
    for i, j in combinations_with_replacement(valid["unique_id"], 2):
        taxid_i = int(valid.loc[valid["unique_id"] == i, "taxid"].values[0])
        taxid_j = int(valid.loc[valid["unique_id"] == j, "taxid"].values[0])
        d = 0 if taxid_i == taxid_j else distances.get((taxid_i, taxid_j), distances.get((taxid_j, taxid_i), 1e6))
        dispd.at[i, j] = dispd.at[j, i] = d
    np.fill_diagonal(dispd.values, 0)
    linkage = hierarchy.linkage(squareform(dispd.astype(float)), method="single")
    return _linkage_to_newick(linkage, dispd.index.tolist())

def _make_neigh_similarity_tree(all_prot):
    pa = all_prot[all_prot["id"] != all_prot["target_prot"]]
    mat = pa.pivot_table(index="target_prot", columns="fam_cluster", values="id", aggfunc="count", fill_value=0)
    binmat = mat.applymap(lambda x: 1 if x > 0 else 0)
    dist = pdist(binmat.values, metric='jaccard')
    linkage = hierarchy.linkage(dist, method="single")
    return _linkage_to_newick(linkage, binmat.index.tolist())

def _make_neigh_phylo_tree(records, all_prot):
    pa = all_prot[all_prot["id"] != all_prot["target_prot"]].copy()
    pa["rel_pos"] = (pa["rel_start"] + pa["rel_end"]) / 2
    targets = pa["target_prot"].unique()
    features = pa["fam_cluster"].unique()
    mat = pd.DataFrame(0, index=targets, columns=features)
    for t in targets:
        for _, row in pa[pa["target_prot"] == t].iterrows():
            mat.at[t, row["fam_cluster"]] += 1 / (1 + abs(row["rel_pos"]))
    mat = mat.div(mat.sum(axis=1), axis=0)
    dist = pdist(mat.values, metric="cosine")
    linkage = hierarchy.linkage(dist, method="single")
    return _linkage_to_newick(linkage, mat.index.tolist())


def _pairwise_to_matrix(pairwise_df: pd.DataFrame,
                        ids: Optional[Iterable[str]] = None,
                        qcol: str = "qseqid",
                        scol: str = "sseqid",
                        valcol: str = "pident",
                        value_is_identity: bool = True) -> pd.DataFrame:
    """Convert pairwise table to full square distance matrix DataFrame.

    - pairwise_df: rows with qcol, scol, valcol
    - ids: iterable of identifiers that must be present (will be included even if missing)
    - value_is_identity: if True, converts identity -> distance as (100 - val)

    Missing pairs are left as NaN (caller will decide fill strategy).
    """
    if pairwise_df is None or pairwise_df.empty:
        base_ids = []
    else:
        base_ids = pd.unique(pairwise_df[[qcol, scol]].values.ravel())
        base_ids = [str(x) for x in base_ids]

    if ids is not None:
        ids_list = [str(x) for x in ids]
        # preserve order of provided ids, append any extras from pairwise
        extras = [x for x in base_ids if x not in ids_list]
        all_ids = ids_list + extras
    else:
        all_ids = sorted(set(base_ids))

    n = len(all_ids)
    # initialize DataFrame with NaN
    mat = pd.DataFrame(index=all_ids, columns=all_ids, dtype=float)
    # fill diagonal with 0
    for i in all_ids:
        mat.at[i, i] = 0.0

    if pairwise_df is None or pairwise_df.empty:
        return mat

    for _, r in pairwise_df.iterrows():
        q = str(r[qcol])
        s = str(r[scol])
        try:
            v = float(r[valcol])
        except Exception:
            continue
        if value_is_identity:
            # convert percent identity to distance (0..100)
            d = 100.0 - v
        else:
            d = v
        if q not in mat.index:
            mat.loc[q] = float('nan')
            mat[q] = float('nan')
            mat.at[q, q] = 0.0
        if s not in mat.index:
            mat.loc[s] = float('nan')
            mat[s] = float('nan')
            mat.at[s, s] = 0.0
        mat.at[q, s] = d
        mat.at[s, q] = d

    return mat


def aai_tree(pairwise_aai: pd.DataFrame,
             valid_uids: Optional[Iterable[str]] = None,
             qcol: str = "qseqid",
             scol: str = "sseqid",
             pcol: str = "pident",
             algorithm: str = "nj",
             threads: int = 1,
             mode: Optional[str] = None,
             subset_mode: Optional[str] = None) -> str:
    """Build a tree from AAI pairwise table (pident) using DecentTree.

    Missing pairs (and ids not present in the table but in valid_uids) are
    filled with (max_observed_distance + 2*std_observed).
    """
    from hoodini.decenttree_runner import run_decenttree_from_table

    if pairwise_aai is None or pairwise_aai.empty:
        raise ValueError("pairwise_aai is empty")

    print(pairwise_aai)
    # Prepare A/B/AAI table expected by run_decenttree_from_table
    df = pairwise_aai.copy()
    df["AAI"] = pd.to_numeric(df["AAI"], errors="coerce")
    df = df.dropna(subset=["AAI"])
    if df.empty:
        raise ValueError("No valid AAI values available to build tree")

    # Scale to 0..100 if values are in 0..1
    if df["AAI"].max() <= 1.0:
        df["AAI"] = df["AAI"] * 100.0

    # disallow hyper mode for AAI-derived trees
    if mode and mode.lower() == 'hyper':
        raise ValueError("'hyper' mode is not supported for AAI trees")

    # Use run_decenttree_from_table; fill_missing='max' uses observed max distance for missing
    return run_decenttree_from_table(df, qcol="A", tcol="B", dcol="distance", algorithm=algorithm, threads=threads, fill_missing='max+2std', ids=valid_uids)


def ani_tree(pairwise_ani: pd.DataFrame,
             valid_uids: Optional[Iterable[str]] = None,
             qcol: str = "A",
             scol: str = "B",
             pcol: str = "ANI",
             algorithm: str = "nj",
             threads: int = 1,
             mode: Optional[str] = None) -> str:
    """Build a tree from ANI pairwise table using DecentTree.

    If the ANI table stores percent identity, distances are computed as
    (100 - ANI). Missing pairs are filled with max+2std as for AAI.
    """
    from hoodini.decenttree_runner import run_decenttree_from_table

    if pairwise_ani is None or pairwise_ani.empty:
        raise ValueError("pairwise_ani is empty")

    # Normalize to A, B, ANI columns
    df = pairwise_ani[[qcol, scol, pcol]].rename(columns={qcol: "A", scol: "B", pcol: "ANI"}).copy()
    df["ANI"] = pd.to_numeric(df["ANI"], errors="coerce")
    df = df.dropna(subset=["ANI"])
    if df.empty:
        raise ValueError("No valid ANI values available to build tree")

    # Scale to 0..100 if values are in 0..1
    if df["ANI"].max() <= 1.0:
        df["ANI"] = df["ANI"] * 100.0

    # Map A/B identifiers to any matching valid_uids (basename or substring) so
    # DecentTree matrix includes all requested UIDs even when pairwise table
    # uses filenames produced by fastANI.
    def _map_ids_to_uids(table: pd.DataFrame, cols=("A", "B"), uids: Optional[Iterable[str]] = None):
        if uids is None:
            return table
        uid_list = [str(x) for x in uids]
        uid_set = set(uid_list)
        import os

        def map_val(x: str) -> str:
            s = str(x)
            if s in uid_set:
                return s
            b = os.path.basename(s)
            if b in uid_set:
                return b
            for uid in uid_list:
                if uid in s:
                    return uid
            return s

        for c in cols:
            if c in table.columns:
                table[c] = table[c].apply(map_val)
        return table

    df = _map_ids_to_uids(df, cols=("A", "B"), uids=valid_uids)

    # Use run_decenttree_from_table and convert ANI->distance internally
    return run_decenttree_from_table(df, qcol="A", tcol="B", dcol="ANI", algorithm=algorithm, threads=threads, fill_missing='max+2std', ids=valid_uids)

def _make_foldmason_tree(records, all_prot, output_dir, threads):
    prots = records.loc[records["failed"].isnull(), "protein_id"].dropna().unique()
    targets = records.loc[records["failed"].isnull(), "unique_id"].dropna().tolist()
    mapper = ProtMapper()
    mapped, no_map = mapper(ids=targets, from_db="EMBL-GenBank-DDBJ_CDS", to_db="UniProtKB")

    fetcher = AlphaFetcher(base_savedir=f"{output_dir}/struct")
    fetcher.add_proteins(mapped["Entry"].unique().tolist())
    fetcher.fetch_metadata(multithread=True, workers=threads)

    no_pdb = mapped[mapped["Entry"].isin(fetcher.failed_ids)]["From"].unique().tolist()
    no_pdb.extend(no_map)
    fetcher.download_all_files(pdb=True, cif=False, multithread=True, workers=threads)

    subprocess.run(["foldmason", "easy-msa", f"{output_dir}/struct/pdb_files", f"{output_dir}/foldmason", f"{output_dir}/temp"], check=True)

    if no_pdb:
        results = all_prot.copy()
        results[results["id"].isin(no_pdb)][["id", "sequence"]].drop_duplicates().to_fasta("id", "sequence", f"{output_dir}/no_pdb.fasta")
        cmd = ["mafft", "--keeplength", "--add", f"{output_dir}/no_pdb.fasta", "--reorder", f"{output_dir}/foldmason_aa.fa"]
        with open(f"{output_dir}/target_prots.aln", 'w') as out:
            subprocess.run(cmd, check=True, stdout=out)
    else:
        os.rename(f"{output_dir}/foldmason_aa.fa", f"{output_dir}/target_prots.aln")

    aln_df = read_fasta(f"{output_dir}/target_prots.aln")
    aln_df = aln_df.merge(mapped[["From", "Entry"]], left_on="id", right_on="Entry", how="left").drop(columns="Entry")
    aln_df["From"] = aln_df["From"].fillna(aln_df["id"])
    aln_df.drop_duplicates(subset=["From"]).to_fasta("From", "sequence", f"{output_dir}/target_prots.aln")

    result = subprocess.run(["VeryFastTree", f"{output_dir}/target_prots.aln"], capture_output=True, text=True)
    return result.stdout

def _linkage_to_newick(linkage, labels):
    tree = hierarchy.to_tree(linkage)
    def build_newick(node):
        if node.is_leaf():
            return labels[node.id]
        left = build_newick(node.left)
        right = build_newick(node.right)
        return f"({left},{right})"
    return build_newick(tree) + ";"

def calculate_taxid_distances(taxids, update_db=False):
    taxids_str = [str(int(taxid)) for taxid in taxids]
    ncbi = ete3.NCBITaxa()
    if update_db:
        try:
            print("Updating NCBI taxonomy database. This may take a while...")
            ncbi.update_taxonomy_database()
            print("Taxonomy database updated successfully.")
        except Exception as e:
            print(f"Error updating taxonomy database: {e}")
            sys.exit(1)

    try:
        tree = ncbi.get_topology(taxids_str, intermediate_nodes=True)
    except Exception as e:
        print(f"Error retrieving topology: {e}")
        sys.exit(1)

    tree_taxids = set()
    taxid_to_node = {}
    for node in tree.traverse():
        try:
            taxid = int(node.name)
            tree_taxids.add(taxid)
            taxid_to_node[taxid] = node
        except ValueError:
            continue

    try:
        taxids_int = [int(taxid) for taxid in taxids_str]
    except ValueError as ve:
        print(f"Error converting taxids to integers: {ve}")
        sys.exit(1)

    missing_taxids = [taxid for taxid in taxids_int if taxid not in tree_taxids]
    if missing_taxids:
        print("The following taxids are missing from the taxonomy tree:")
        for mtaxid in missing_taxids:
            try:
                name = ncbi.get_taxid_translator([mtaxid]).get(mtaxid, "Unknown")
            except Exception:
                name = "Unknown"
            print(f" - {mtaxid} ({name})")
        raise ValueError("Some taxids are missing from the taxonomy tree. Please verify their validity.")
    else:
        print("All taxids are present in the taxonomy tree.")

    distances = {}
    print("\nCalculating pairwise distances using node objects...")
    for taxid1, taxid2 in itertools.combinations(taxids_int, 2):
        try:
            node1 = taxid_to_node[taxid1]
            node2 = taxid_to_node[taxid2]
            distance = tree.get_distance(node1, node2)
            distances[(taxid1, taxid2)] = distance
        except Exception as e:
            print(f"Error calculating distance between {taxid1} and {taxid2}: {e}")

    if not distances:
        raise ValueError("No distances were calculated. Please check the taxids and try again.")

    print("\nPairwise Distances Calculated Successfully.")
    return distances
