from __future__ import annotations
from typing import Optional, Tuple, Union
import math
import os
import concurrent.futures as _fut

import polars as pl
import numpy as np
import pandas as pd
from hoodini.utils.core import console

try:
    from scipy.stats import hypergeom
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================
# Helpers & normalizers
# =========================

def _ensure_polars(df_or_lazy) -> pl.DataFrame:
    """Convert pandas/polars input to eager Polars DataFrame."""
    if isinstance(df_or_lazy, pd.DataFrame):
        return pl.from_pandas(df_or_lazy)
    elif isinstance(df_or_lazy, pl.LazyFrame):
        return df_or_lazy.collect()
    elif isinstance(df_or_lazy, pl.DataFrame):
        return df_or_lazy
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df_or_lazy)}")


def _normalize_hits_df(hits: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize a BLAST outfmt 6 table to internal schema:
      qseqid -> qprot, sseqid -> tprot, keep pident (float).
    Other columns (length, evalue, bitscore) are preserved but unused.
    """
    need = {"qseqid", "sseqid", "pident"}
    missing = need - set(hits.columns)
    if missing:
        raise ValueError(
            f"hits_df is missing required columns: {sorted(missing)}. "
            f"Available: {hits.columns}"
        )
    out = hits.rename({"qseqid": "qprot", "sseqid": "tprot"})
    return out.with_columns([
        pl.col("qprot").cast(pl.Utf8, strict=False),
        pl.col("tprot").cast(pl.Utf8, strict=False),
        pl.col("pident").cast(pl.Float64, strict=False),
    ])


def _normalize_proteins_df(prots: pl.DataFrame, require_fam: bool = False) -> pl.DataFrame:
    """
    Normalize protein annotation table to internal schema:
      prot_id, target_nuc, optional fam_cluster.
    Accepts common aliases.
    """
    cols = set(prots.columns)

    prot_col = next((c for c in
                     ["prot_id", "protein_id", "protid", "proteinId", "proteinID"]
                     if c in cols), None)
    nuc_col = next((c for c in
                    ["target_nuc", "nucleotide_id", "seqid", "seq_id",
                     "contig_id", "contig", "scaffold_id", "target_seq"]
                    if c in cols), None)
    fam_col = next((c for c in
                    ["fam_cluster", "fam_clustter", "family_cluster",
                     "cluster", "protein_family", "pfam_cluster", "fam"]
                    if c in cols), None)

    miss = []
    if prot_col is None:
        miss.append("prot_id (aliases: protein_id, protid, ...)")
    if nuc_col is None:
        miss.append("target_nuc (aliases: nucleotide_id, seqid, contig_id, ...)")
    if miss:
        raise ValueError(
            f"proteins_df is missing required columns: {miss}. Available: {prots.columns}"
        )

    out = prots.rename({prot_col: "prot_id", nuc_col: "target_nuc"})
    if fam_col:
        out = out.rename({fam_col: "fam_cluster"})
    elif require_fam:
        raise ValueError(
            "proteins_df needs a family column (e.g., fam_cluster) for the vContact2-like metric."
        )
    return out


def _annotate_hits(hits: pl.DataFrame, prots: pl.DataFrame,
                   pident_min: Optional[float], exclude_self: bool) -> pl.DataFrame:
    """Join hits to seq IDs and apply filters in Polars."""
    qp = prots.rename({"prot_id": "qprot", "target_nuc": "q_seq"})
    tp = prots.rename({"prot_id": "tprot", "target_nuc": "t_seq"})

    ann = (
        hits
        .join(qp, on="qprot", how="left")
        .join(tp, on="tprot", how="left")
        .drop_nulls(["q_seq", "t_seq"])
    )
    if pident_min is not None:
        ann = ann.filter(pl.col("pident") >= float(pident_min))
    if exclude_self:
        ann = ann.filter(pl.col("q_seq") != pl.col("t_seq"))
    return ann


def _sizes_per_seq(prots: pl.DataFrame, col_name="n_prots") -> pl.DataFrame:
    return (
        prots
        .group_by("target_nuc")
        .agg(pl.col("prot_id").n_unique().alias(col_name))
    )


def _find_prot_id_col(prots: pl.DataFrame) -> str:
    """Return the protein identifier column name present in `prots`.
    Prefer common names: 'prot_id', 'protein_id', 'id'."""
    for c in ("prot_id", "protein_id", "id", "protid"):
        if c in prots.columns:
            return c
    raise ValueError(f"cannot find protein id column in proteins table. Available: {prots.columns}")


def _compute_subset_protein_ids(all_prots: pl.DataFrame,
                                all_neigh: pl.DataFrame,
                                all_gff: pl.DataFrame,
                                subset_mode: str,
                                win: Optional[int] = None,
                                win_mode: str = "bp") -> set:
    """Compute a set of protein ids to keep according to subset_mode.

    Modes supported:
      - 'target_prot': keep unique values from column 'target_prot' in `all_prots`.
      - 'target_region': keep all proteins whose coordinates (in all_gff) fall within
         the neighborhood window (start_win..end_win) from `all_neigh`.
      - 'window': like 'target_region' but expand the neighborhood window by `win` on
         both sides before selecting proteins.
    """
    mode = (subset_mode or "").lower()
    if mode not in {"target_prot", "target_region", "window"}:
        raise ValueError(f"Unknown subset_mode={subset_mode}")

    if mode == "target_prot":
        if "target_prot" not in all_prots.columns:
            raise ValueError("'target_prot' column not found in all_prots for subset_mode='target_prot'")
        vals = all_prots.select("target_prot").drop_nulls().unique().to_series().to_list()
        return set([v for v in vals if v is not None])

    # for region/window modes we need coords in all_neigh and features in all_gff
    if not {"seqid", "start_win", "end_win"}.issubset(set(all_neigh.columns)):
        raise ValueError("all_neigh must contain columns: seqid, start_win, end_win for region/window subsetting")
    # all_gff should contain seqid, start, end and protein id. If no id-like
    # column exists but there is an `attributes` column, extract `ID=` from it
    # into a temporary 'id' column so we don't mutate the caller's frame.
    prot_col = None
    for c in ("protein_id", "prot_id", "id"):
        if c in all_gff.columns:
            prot_col = c
            break
    if prot_col is None:
        if "attributes" in all_gff.columns:
            # create a temporary copy with extracted id; do not modify original
            # Use Polars string extract to capture the ID=... value before semicolon
            temp_gff = all_gff.with_columns(
                pl.col("attributes").str.extract(r"ID=([^;]+)").alias("id")
            )
            prot_col = "id"
            all_gff = temp_gff
        else:
            raise ValueError("all_gff must contain a protein id column (protein_id/id) or an 'attributes' column with ID= to perform region/window subsetting")

    # build join between gff and neighbourhoods then filter ranges
    neigh = all_neigh.select(["seqid", "start_win", "end_win"]).drop_nulls()
    gff = all_gff.select(["seqid", "start", "end", prot_col]).drop_nulls()

    if mode == "window":
        w = int(win or 0)
        # two supported expansion modes: basepairs (bp / win_nts) or number of genes (genes / win_genes)
        wm = (win_mode or "").lower()
        gene_mode = wm in ("genes", "win_genes", "win-genes", "win_genes", "gene", "win_gene")
        if gene_mode:
            # assign a gene index per seqid based on start coordinate (0-based)
            # use ordinal rank over seqid to get deterministic ordering
            gff = gff.with_columns(
                ((pl.col("start").rank(method="ordinal").over("seqid") - 1).cast(pl.Int64)).alias("gene_idx")
            )

            # join and find gene_idx range for features contained in the neighbourhood
            joined = gff.join(neigh, on="seqid", how="inner")
            contained = joined.filter((pl.col("start") >= pl.col("start_win")) & (pl.col("end") <= pl.col("end_win")))
            if contained.is_empty():
                return set()
            min_idx = int(contained.select(pl.col("gene_idx").min()).to_series().iloc[0])
            max_idx = int(contained.select(pl.col("gene_idx").max()).to_series().iloc[0])
            start_idx = max(0, min_idx - w)
            end_idx = max_idx + w
            sel = gff.filter((pl.col("gene_idx") >= start_idx) & (pl.col("gene_idx") <= end_idx))
        else:
            # bp-mode: expand neighborhood windows by w (basepairs)
            neigh = neigh.with_columns([
                (pl.col("start_win") - w).alias("start_exp"),
                (pl.col("end_win") + w).alias("end_exp"),
            ])
            neigh = neigh.with_columns([
                pl.when(pl.col("start_exp") < 0).then(0).otherwise(pl.col("start_exp")).alias("start_exp"),
            ])
            # perform join on seqid then filter using expanded bounds
            joined = gff.join(neigh, on="seqid", how="inner")
            sel = joined.filter((pl.col("start") >= pl.col("start_exp")) & (pl.col("end") <= pl.col("end_exp")))
    else:
        # target_region
        joined = gff.join(neigh, on="seqid", how="inner")
        sel = joined.filter((pl.col("start") >= pl.col("start_win")) & (pl.col("end") <= pl.col("end_win")))

    if sel.is_empty():
        return set()
    vals = sel.select(prot_col).unique().to_series().to_list()
    return set([v for v in vals if v is not None])


def _rbh_from_ann(ann: pl.DataFrame) -> pl.DataFrame:
    """
    Compute Reciprocal Best Hits fully in Polars.
    Tie-breaking: ordinal rank => first occurrence wins within each group.
    Returns columns: A, B, qprot, tprot, pident
    """
    # Best tprot per (qprot, t_seq)
    best_q_to_B = (
        ann
        .with_columns(
            pl.col("pident")
              .rank(method="ordinal", descending=True)  # deterministic ties
              .over(["qprot", "t_seq"])
              .alias("_rk")
        )
        .filter(pl.col("_rk") == 1)
        .select([
            pl.col("qprot"),
            pl.col("t_seq").alias("B"),
            pl.col("tprot"),
            pl.col("pident").alias("pident_best_qB"),
        ])
    )

    # Best qprot per (tprot, q_seq)
    best_t_to_A = (
        ann
        .with_columns(
            pl.col("pident")
              .rank(method="ordinal", descending=True)
              .over(["tprot", "q_seq"])
              .alias("_rk")
        )
        .filter(pl.col("_rk") == 1)
        .select([
            pl.col("tprot"),
            pl.col("q_seq").alias("A"),
            pl.col("qprot"),
            pl.col("pident").alias("pident_best_tA"),
        ])
    )

    # Reciprocal join: qprot chose tprot AND tprot chose qprot
    rbh = (
        best_q_to_B
        .join(
            best_t_to_A,
            left_on=["qprot", "tprot"],
            right_on=["qprot", "tprot"],
            how="inner",
            suffix="_r",
        )
        .select([
            pl.col("A"),
            pl.col("B"),
            pl.col("qprot"),
            pl.col("tprot"),
            pl.col("pident_best_qB").alias("pident"),  # one pident per RBH pair
        ])
    )

    return rbh



# =========================
# Metrics (return pandas)
# =========================

def compute_wgrr(hits_df: pl.DataFrame | pl.LazyFrame,
                 proteins_df: pl.DataFrame | pl.LazyFrame,
                 pident_min: Optional[float] = 30.0,
                 exclude_self: bool = True,
                 symmetric: bool = True) -> pd.DataFrame:
    """
    Weighted Gene Repertoire Relatedness (wGRR) between proteomes.

    Definition (symmetric by construction):
      - Build RBHs between proteomes A and B.
      - Sum identity FRACTIONS (pident/100) across RBH pairs per (A,B).
      - Normalize by min(nA, nB).
      - Clip to [0, 1].

    Returns pandas.DataFrame with columns: ["A", "B", "wGRR_sym"]
    """
    hits_raw = _ensure_polars(hits_df)
    prots_raw = _ensure_polars(proteins_df)

    hits = _normalize_hits_df(hits_raw)
    prots = _normalize_proteins_df(prots_raw)

    sizes = _sizes_per_seq(prots, col_name="n_prots")

    ann = _annotate_hits(hits, prots, pident_min=pident_min, exclude_self=exclude_self)
    rbh = _rbh_from_ann(ann)
    if rbh.is_empty():
        return pd.DataFrame(columns=["A", "B", "wGRR_sym"])

    w = (
        rbh
        .group_by(["A", "B"])
        .agg(pl.col("pident").sum().alias("sum_pident_pct"))
        .join(sizes.rename({"target_nuc": "A"}), on="A", how="left")
        .join(sizes.rename({"target_nuc": "B"}), on="B", how="left", suffix="_B")
        .with_columns([
            (pl.col("sum_pident_pct") / 100.0).alias("sum_pident_frac"),
            pl.min_horizontal(pl.col(["n_prots", "n_prots_B"])).alias("n_min"),
        ])
        .with_columns(
            (pl.when(pl.col("n_min") > 0)
               .then(pl.col("sum_pident_frac") / pl.col("n_min"))
               .otherwise(0.0)
            ).alias("wGRR_sym")
        )
        .with_columns(pl.col("wGRR_sym").clip(0.0, 1.0))
        .select(["A", "B", "wGRR_sym"])
    )
    df = w.to_pandas()
    # Standardize output: add AAI column as duplicate of wGRR_sym
    df["AAI"] = df["wGRR_sym"]
    return df[["A", "B", "AAI"]]


def compute_aai_rbh(hits_df: pl.DataFrame | pl.LazyFrame,
                    proteins_df: pl.DataFrame | pl.LazyFrame,
                    pident_min: Optional[float] = 30.0,
                    exclude_self: bool = True) -> pd.DataFrame:
    """
    Average Amino-acid Identity using Reciprocal Best Hits (RBH).
    Returns pandas.DataFrame.
    """
    hits_raw = _ensure_polars(hits_df)
    prots_raw = _ensure_polars(proteins_df)

    hits = _normalize_hits_df(hits_raw)
    prots = _normalize_proteins_df(prots_raw)

    ann = _annotate_hits(hits, prots, pident_min=pident_min, exclude_self=exclude_self)
    rbh = _rbh_from_ann(ann)
    if rbh.is_empty():
        return pd.DataFrame(columns=["A", "B", "AAI", "n_RBH", "RBH_frac_min"])

    aai = (
        rbh
        .group_by(["A", "B"])
        .agg([
            pl.len().alias("n_RBH"),
            pl.col("pident").mean().alias("AAI"),
        ])
    )

    sizes = _sizes_per_seq(prots, col_name="n_prots")
    aai = (
        aai
        .join(sizes.rename({"target_nuc": "A"}), on="A", how="left")
        .join(sizes.rename({"target_nuc": "B"}), on="B", how="left", suffix="_B")
        .with_columns(
            (pl.col("n_RBH") / pl.min_horizontal(pl.col(["n_prots", "n_prots_B"]))).alias("RBH_frac_min")
        )
        .select(["A", "B", "AAI", "n_RBH", "RBH_frac_min"])
    )
    df = aai.to_pandas()
    # Standardize output: add AAI column as duplicate of AAI
    df["AAI"] = df["AAI"]
    return df[["A", "B", "AAI"]]


def compute_vcontact2_hypergeom(proteins_df: pl.DataFrame | pl.LazyFrame,
                                max_df_frac: float = 0.2,
                                min_shared: int = 2,
                                multiple_test: str = "BH") -> pd.DataFrame:
    """
    vContact2-like hypergeometric similarity on presence/absence of protein families.
    Vectorized version (no Python loops). Returns pandas.DataFrame.
    """
    prots_raw = _ensure_polars(proteins_df)

    fam_like = [c for c in ["fam_cluster", "fam_clustter", "family_cluster",
                            "cluster", "protein_family", "pfam_cluster", "fam"]
                if c in prots_raw.columns]
    if not fam_like:
        return pd.DataFrame(columns=["A", "B", "k", "K_A", "K_B", "M", "pval", "p_adj", "score"])

    prots = _normalize_proteins_df(prots_raw, require_fam=True)

    # Presence/absence table (unique pairs)
    df_pa = prots.select(["target_nuc", "fam_cluster"]).drop_nulls().unique()
    if df_pa.is_empty():
        return pd.DataFrame(columns=["A", "B", "k", "K_A", "K_B", "M", "pval", "p_adj", "score"])

    # Filter ubiquitous families by document frequency
    fam_counts = df_pa.group_by("fam_cluster").agg(pl.len().alias("df"))
    Nseqs = df_pa["target_nuc"].n_unique()
    keep_thresh = int(math.ceil(max_df_frac * max(1, Nseqs)))
    keep_fams = fam_counts.filter(pl.col("df") <= keep_thresh).select("fam_cluster")
    if keep_fams.is_empty():
        return pd.DataFrame(columns=["A", "B", "k", "K_A", "K_B", "M", "pval", "p_adj", "score"])

    df_pa = df_pa.join(keep_fams, on="fam_cluster", how="inner")

    # K_A, K_B (num fams per sequence after filtering)
    sizes = df_pa.group_by("target_nuc").agg(pl.col("fam_cluster").n_unique().alias("K"))

    # Total number of families kept
    M = int(df_pa["fam_cluster"].n_unique())

    # --- Build all unordered sequence pairs sharing a family via a single self-join ---
    # (A, B) with A < B to avoid duplicates; count families => k
    pairs = (
        df_pa.join(df_pa, on="fam_cluster", how="inner", suffix="_r")
             .filter(pl.col("target_nuc") < pl.col("target_nuc_r"))
             .group_by(["target_nuc", "target_nuc_r"])
             .agg(pl.len().alias("k"))
             .rename({"target_nuc": "A", "target_nuc_r": "B"})
             .filter(pl.col("k") >= int(min_shared))
    )
    if pairs.is_empty():
        return pd.DataFrame(columns=["A", "B", "k", "K_A", "K_B", "M", "pval", "p_adj", "score"])

    pairs = (
        pairs
        .join(sizes.rename({"target_nuc": "A", "K": "K_A"}), on="A", how="left")
        .join(sizes.rename({"target_nuc": "B", "K": "K_B"}), on="B", how="left")
        .with_columns(pl.lit(M).alias("M"))
    )

    pdf = pairs.to_pandas()
    if pdf.empty:
        return pd.DataFrame(columns=["A", "B", "k", "K_A", "K_B", "M", "pval", "p_adj", "score"])

    # Hypergeometric survival function (P[X >= k])
    try:
        if _HAS_SCIPY:
            rv = hypergeom(pdf["M"].iloc[0], pdf["K_A"].to_numpy(), pdf["K_B"].to_numpy())
            pvals = rv.sf(pdf["k"].to_numpy() - 1)
        else:
            from math import comb
            def hypergeom_sf(k, Mv, K, n):
                denom = comb(Mv, n)
                top = 0
                for i in range(int(k), int(min(K, n)) + 1):
                    top += comb(K, i) * comb(Mv - K, n - i)
                return top / denom if denom != 0 else 1.0
            pvals = np.array([hypergeom_sf(k, M, Ka, Kb) for k, Ka, Kb, M in zip(pdf["k"], pdf["K_A"], pdf["K_B"], pdf["M"])])
    except Exception:
        pvals = np.ones(len(pdf))

    pdf["pval"] = np.clip(pvals, 1e-300, 1.0)
    mtests = len(pdf)

    if multiple_test.lower() == "bonferroni":
        pdf["p_adj"] = np.minimum(1.0, pdf["pval"] * mtests)
    else:
        order = np.argsort(pdf["pval"].to_numpy())
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, mtests + 1)
        pdf["p_adj"] = np.minimum(1.0, pdf["pval"] * mtests / ranks)

    pdf["score"] = -np.log10(np.maximum(1e-300, pdf["p_adj"]))
    # Standardize output: add AAI column as duplicate of score
    pdf["AAI"] = pdf["score"]
    return pdf[["A", "B", "AAI"]].copy()


# =========================
# Orchestrator (mode + pandas) with optional parallelism
# =========================

def run_proteome_similarity(all_prots: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
                            pairwise_aa: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
                            all_neigh: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
                            all_gff: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
                            outdir: Optional[str] = None,
                            pident_min: float = 30.0,
                            mode: str = "all",
                            subset_mode: Optional[str] = None,
                            win: Optional[int] = None,
                            win_mode: str = "bp",
                            parallel: bool = False) -> Union[
                                pd.DataFrame,
                                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
                            ]:
    """
    Compute wGRR, AAI (RBH), and/or vContact2-like hypergeometric scores.

    Parameters
    ----------
    all_prots : pandas/Polars DataFrame
    pairwise_aa : pandas/Polars DataFrame (BLAST outfmt 6 or normalized)
    all_neigh : unused (kept for API compatibility)
    outdir : optional path to write TSVs
    pident_min : identity threshold for hits (default 30.0)
    mode : {'wgrr','aai','hyper','all'}
    parallel : if True and mode='all', run metrics concurrently (3 threads)

    Returns
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
    """
    mode = (mode or "all").lower()
    valid = {"wgrr", "aai", "hyper", "all"}
    if mode not in valid:
        raise ValueError(f"Invalid mode='{mode}'. Choose from {sorted(valid)}")

    # normalize once to Polars internally
    all_prots_pl = _ensure_polars(all_prots)
    pairwise_aa_pl = _ensure_polars(pairwise_aa)
    all_neigh_pl = None
    all_gff_pl = None
    if all_neigh is not None:
        all_neigh_pl = _ensure_polars(all_neigh)
    if all_gff is not None:
        all_gff_pl = _ensure_polars(all_gff)

    # Prefer neighborhood-wise calculations: if the proteins table contains a
    # 'unique_id' column, treat that as the sequence unit for all grouping and
    # RBH calculations so results are reported per-neighborhood (unique_id).
    # If 'unique_id' is missing but `all_neigh` is provided, attempt to map
    # target_nuc (seqid/temp_seqid) -> unique_id for seqids that map to a
    # single neighborhood to avoid ambiguous cross-products.
    try:
        if 'unique_id' in set(all_prots_pl.columns):
            console.log("run_proteome_similarity: using 'unique_id' from proteins table to compute neighborhood-wise similarities")
            # overwrite target_nuc with unique_id for downstream grouping/annotation
            # ensure string type
            all_prots_pl = all_prots_pl.with_columns(
                pl.col('unique_id').cast(pl.Utf8).alias('target_nuc')
            )
        elif all_neigh_pl is not None and not all_neigh_pl.is_empty() and 'target_nuc' in set(all_prots_pl.columns):
            # build mapping from temp_seqid and seqid -> unique_id where unambiguous
            try:
                an_pd = all_neigh_pl.select(['temp_seqid', 'seqid', 'unique_id']).drop_nulls().to_pandas()
            except Exception:
                an_pd = all_neigh_pl.to_pandas() if not all_neigh_pl.is_empty() else pd.DataFrame()

            mapping = {}
            if not an_pd.empty:
                # temp_seqid -> unique_id (explicit neighborhood instance)
                if 'temp_seqid' in an_pd.columns:
                    for _, r in an_pd.dropna(subset=['temp_seqid', 'unique_id']).iterrows():
                        mapping[str(r['temp_seqid'])] = str(r['unique_id'])
                # seqid -> unique_id only when seqid maps to a single unique_id
                if 'seqid' in an_pd.columns:
                    grp = an_pd.dropna(subset=['seqid', 'unique_id']).groupby('seqid')['unique_id'].nunique()
                    singles = set(grp[grp == 1].index.astype(str).tolist())
                    if singles:
                        for seq in singles:
                            uid = an_pd.loc[an_pd['seqid'] == seq, 'unique_id'].iloc[0]
                            mapping[str(seq)] = str(uid)

            if mapping:
                console.log(f"run_proteome_similarity: mapping target_nuc -> unique_id for {len(mapping)} seq keys (unambiguous mapping)")
                # apply mapping in pandas then back to Polars for simplicity
                ap = all_prots_pl.to_pandas()
                ap['target_nuc_mapped'] = ap['target_nuc'].astype(str).map(mapping)
                # replace target_nuc where mapping exists
                ap['target_nuc'] = ap['target_nuc_mapped'].where(pd.notna(ap['target_nuc_mapped']), ap['target_nuc'])
                ap = ap.drop(columns=['target_nuc_mapped'])
                all_prots_pl = pl.from_pandas(ap)
                console.log(f"run_proteome_similarity: after mapping, unique target_nuc values={all_prots_pl.select('target_nuc').n_unique()}")
    except Exception as e:
        console.log(f"run_proteome_similarity: neighborhood remapping skipped due to error: {e}")

    # If subsetting requested, compute set of protein IDs and filter both tables
    if subset_mode:
        if all_neigh_pl is None or all_gff_pl is None:
            raise ValueError("subset_mode requires both all_neigh and all_gff to be provided")
        keep_ids = _compute_subset_protein_ids(all_prots_pl, all_neigh_pl, all_gff_pl, subset_mode, win=win, win_mode=win_mode)
        # filter all_prots
        if keep_ids:
            pid_col = _find_prot_id_col(all_prots_pl)
            all_prots_pl = all_prots_pl.filter(pl.col(pid_col).is_in(list(keep_ids)))
        else:
            # nothing to keep; make empty frames
            all_prots_pl = all_prots_pl.filter(pl.lit(False))
        # filter pairwise_aa by qseqid/sseqid matching kept protein ids
        if not pairwise_aa_pl.is_empty():
            # pairwise_aa expected to have qseqid and sseqid
            if {"qseqid", "sseqid"}.issubset(set(pairwise_aa_pl.columns)):
                pairwise_aa_pl = pairwise_aa_pl.filter(
                    (pl.col("qseqid").is_in(list(keep_ids))) & (pl.col("sseqid").is_in(list(keep_ids)))
                )
            else:
                # try normalized names (qprot/tprot)
                if {"qprot", "tprot"}.issubset(set(pairwise_aa_pl.columns)):
                    pairwise_aa_pl = pairwise_aa_pl.filter(
                        (pl.col("qprot").is_in(list(keep_ids))) & (pl.col("tprot").is_in(list(keep_ids)))
                    )
                else:
                    # cannot subset pairwise file; warn by raising
                    raise ValueError("pairwise_aa must contain qseqid/sseqid (or qprot/tprot) to apply subsetting")

    wgrr_df = aai_df = vcon_df = None

    if mode == "all" and parallel:
        with _fut.ThreadPoolExecutor(max_workers=3) as ex:
            fut_w = ex.submit(compute_wgrr, pairwise_aa_pl, all_prots_pl, pident_min, True, True)
            fut_a = ex.submit(compute_aai_rbh, pairwise_aa_pl, all_prots_pl, pident_min, True)
            fut_h = ex.submit(compute_vcontact2_hypergeom, all_prots_pl, 0.2, 2, "BH")
            wgrr_df = fut_w.result()
            aai_df = fut_a.result()
            vcon_df = fut_h.result()
    else:
        if mode in {"wgrr", "all"}:
            wgrr_df = compute_wgrr(pairwise_aa_pl, all_prots_pl, pident_min=pident_min, symmetric=True)
        if mode in {"aai", "all"}:
            aai_df = compute_aai_rbh(pairwise_aa_pl, all_prots_pl, pident_min=pident_min)
        if mode in {"hyper", "all"}:
            vcon_df = compute_vcontact2_hypergeom(all_prots_pl, max_df_frac=0.2, min_shared=2)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if wgrr_df is not None:
            wgrr_df.to_csv(os.path.join(outdir, "wgrr.tsv"), sep="\t", index=False)
        if aai_df is not None:
            aai_df.to_csv(os.path.join(outdir, "aai.tsv"), sep="\t", index=False)
        if vcon_df is not None:
            vcon_df.to_csv(os.path.join(outdir, "vcontact_hypergeom.tsv"), sep="\t", index=False)

    if mode == "wgrr":
        return wgrr_df
    elif mode == "aai":
        return aai_df
    elif mode == "hyper":
        return vcon_df
    else:
        # 'all' – always return DataFrames even if empty
        if wgrr_df is None:
            wgrr_df = pd.DataFrame(columns=["A", "B", "wGRR_sym"])
        if aai_df is None:
            aai_df = pd.DataFrame(columns=["A", "B", "AAI", "n_RBH", "RBH_frac_min"])
        if vcon_df is None:
            vcon_df = pd.DataFrame(columns=["A", "B", "k", "K_A", "K_B", "M", "pval", "p_adj", "score"])
        return wgrr_df, aai_df, vcon_df
