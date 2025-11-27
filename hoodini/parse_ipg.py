#!/usr/bin/env python3
import os
import sys
import pandas as pd
import polars as pl
from importlib.resources import files
from rich.console import Console

from hoodini.fetch_ipg_from_accessions import fetch_ipg_from_accessions
from hoodini.nuc2asmlen import run_nuc2asmlen

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe_collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Try Polars documented streaming engine; if not available, fall back
    to non-streaming. No verbose logs, just do it.
    """
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        # engine kwarg not supported by this wheel
        return lf.collect(streaming=False)
    except Exception:
        return lf.collect(streaming=False)

# ──────────────────────────────────────────────────────────────────────────────
# IPG fill helper
# ──────────────────────────────────────────────────────────────────────────────

def fill_ipg(records: pd.DataFrame, ipg_df: pd.DataFrame) -> pd.DataFrame:
    ipg_map = ipg_df[["protein_id", "ipg_id"]].drop_duplicates("protein_id")
    records = records.merge(ipg_map, on="protein_id", how="left")

    # mark as failed when enrichment eligible but no ipg found
    cond = (
        records["protein_id"].notnull()
        & records["failed"].isnull()
        & records["premade"].isnull()
        & records["ipg_id"].isna()
    )
    records.loc[cond, "failed"] = "Unable to retrieve IPG"

    # prepare for conditional fill
    records["_has_both"] = records["protein_id"].notna() & records["nucleotide_id"].notna()
    records["_original_nuc"] = records["nucleotide_id"]
    records["__row_id__"] = range(len(records))

    merged = records.merge(
        ipg_df.drop(columns=["protein_id"]),
        on="ipg_id",
        how="left",
        suffixes=("", "_filler"),
    )

    def keep_matching_or_all(group):
        if not group["_has_both"].any():
            return group
        orig_nuc = group["_original_nuc"].iloc[0]
        matched = group[group["nucleotide_id_filler"] == orig_nuc]
        return matched if not matched.empty else group

    filtered = (
        merged.groupby("__row_id__", group_keys=False)
        .apply(keep_matching_or_all)
        .reset_index(drop=True)
    )

    # fill from *_filler columns when missing/empty
    for col in records.columns:
        if col in ["_has_both", "_original_nuc", "__row_id__"]:
            continue
        filler_col = f"{col}_filler"
        if filler_col in filtered.columns:
            filtered[col] = filtered[col].where(
                filtered[col].notna() & (filtered[col] != ""), filtered[filler_col]
            )

    final = filtered[
        [c for c in filtered.columns if not c.endswith("_filler") and not c.startswith("_")]
    ].reset_index(drop=True)

    return final

# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_ipg(records_df: pd.DataFrame, *, cand_mode: str) -> pd.DataFrame:
    df = records_df.copy()
    console.print("🔍  Fetching IPG data...")
    df = _fetch_ipg_data(df, cand_mode)
    console.print("🔍  Fetching nucleotide data...")
    df = _fetch_nucleotide_data(df)
    console.print("✅  Selecting best IPG records...")
    df = _select_best_ipg(df, cand_mode)
    df = _finalize_ipg(df, cand_mode)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Fetch IPG data (no is_in; joins only)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_ipg_data(df: pd.DataFrame, cand_mode: str) -> pd.DataFrame:
    cond = (
        df["protein_id"].notnull()
        & df["failed"].isnull()
        & df["premade"].isnull()
    )
    proteins = df.loc[cond, "protein_id"].dropna().unique().tolist()
    if not proteins:
        console.print("ℹ️  No records match conditions for IPG.")
        return df

    ipg_df = fetch_ipg_from_accessions(proteins)
    console.log(f"Fetched {len(ipg_df)} IPG records for {len(proteins)} proteins.")

    assemblies = ipg_df["assembly"].dropna().unique().tolist()
    if not assemblies:
        console.print("ℹ️  No assemblies found in IPG data.")
        return df

    # dive_combined.parquet via semi-join (join with small key table)
    try:
        dive_path = files("hoodini").joinpath("data", "dive_combined.parquet")
        asm_df = pl.DataFrame({"assembly_id": assemblies})
        ts = (
            safe_collect(
                pl.scan_parquet(dive_path).join(asm_df.lazy(), on="assembly_id", how="inner")
            ).to_pandas()
        )
        if not ts.empty:
            ipg_df = (
                ipg_df.merge(ts, left_on="assembly", right_on="assembly_id", how="left")
                      .drop(columns=["assembly_id"], errors="ignore")
            )
    except Exception as e:
        console.print(f"[WARN] Skipping dive_combined.parquet due to error: {e}")

    # assembly_summary.parquet via semi-join
    try:
        summary_path = files("hoodini").joinpath("data", "assembly_summary.parquet")
        asm_df2 = pl.DataFrame({"assembly_accession": assemblies})
        summary = (
            safe_collect(
                pl.scan_parquet(summary_path).join(asm_df2.lazy(), on="assembly_accession", how="inner")
            ).to_pandas()
        )
        if not summary.empty:
            ipg_df = ipg_df.merge(
                summary[
                    [
                        "assembly_accession", "taxid", "species_taxid", "organism_name",
                        "infraspecific_name", "assembly_level", "group",
                    ]
                ],
                left_on="assembly",
                right_on="assembly_accession",
                how="left",
            ).drop(columns=["assembly_accession"], errors="ignore")
    except Exception as e:
        console.print(f"[WARN] Skipping assembly_summary.parquet due to error: {e}")

    # normalize & rename
    ipg_df.rename(
        columns={
            "nucleotide accession": "nucleotide_id",
            "assembly": "assembly_id",
            "stop": "end",
            "protein": "protein_id",
            "id": "ipg_id",
        },
        inplace=True,
    )

    # fill IPG info
    with_ipg_info = fill_ipg(records=df, ipg_df=ipg_df)

    # fix assembly prefix consistency
    from hoodini.utils.core import is_refseq_nuccore, switch_assembly_prefix

    def fix_assembly_id(row):
        nuc_id = row.get("nucleotide_id", None)
        asm_id = row.get("assembly_id", None)
        if pd.isna(nuc_id) or pd.isna(asm_id):
            return asm_id
        is_refseq = is_refseq_nuccore(nuc_id)
        if is_refseq and str(asm_id).startswith("GCA_"):
            return switch_assembly_prefix(asm_id)
        elif (not is_refseq) and str(asm_id).startswith("GCF_"):
            return switch_assembly_prefix(asm_id)
        return asm_id

    mask = (
        with_ipg_info["failed"].isnull()
        & with_ipg_info["premade"].isnull()
        & with_ipg_info["nucleotide_id"].notnull()
    )
    with_ipg_info.loc[mask, "assembly_id"] = with_ipg_info.loc[mask].apply(fix_assembly_id, axis=1)
    return with_ipg_info

# ──────────────────────────────────────────────────────────────────────────────
# Fetch nucleotide data (no is_in; joins/semi-joins only)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_nucleotide_data(df: pd.DataFrame) -> pd.DataFrame:
    nucs = df["nucleotide_id"].dropna().unique().tolist()
    if not nucs:
        console.print("ℹ️  No nucleotide IDs to fetch.")
        return df

    base = files("hoodini").joinpath("data")
    contig_path = base / "contig_lengths"
    summary_path = base / "assembly_summary.parquet"

    # 1. Build a small mapping table in Polars
    #    (nucleotide_id -> sequence_length, assembly_id, taxid, group, etc.)
    try:
        # Filter out empty strings just in case
        nucs = [n for n in nucs if n and str(n).strip()]

        # Lazy scan of contig_lengths
        cols_keep = [
            "assemblyAccession", "genbankAccession", "refseqAccession",
            "length", "assemblyUnit", "role",
        ]
        contigs_scan = pl.scan_parquet(contig_path, allow_missing_columns=True).select(cols_keep)

        # Use filter with is_in which pushes down better to Parquet readers than joins
        # and avoids scanning the whole file if row groups can be skipped.
        contigs_lf = contigs_scan.filter(
            pl.col("genbankAccession").is_in(nucs) | 
            pl.col("refseqAccession").is_in(nucs)
        )

        # Lazy scan of assembly_summary
        summary_scan = pl.scan_parquet(summary_path).select([
            "assembly_accession", "taxid", "species_taxid", "organism_name",
            "infraspecific_name", "assembly_level", "group"
        ])

        # Join contigs + summary entirely in Polars
        # contigs.assemblyAccession <-> summary.assembly_accession
        joined_lf = contigs_lf.join(
            summary_scan, 
            left_on="assemblyAccession", 
            right_on="assembly_accession", 
            how="left"
        )

        # Select & rename only what we need for the final mapping
        final_cols = [
            pl.col("assemblyAccession").alias("assembly_id"),
            pl.col("length").alias("sequence_length"),
            pl.col("taxid"),
            pl.col("group"),
            pl.col("species_taxid"),
            pl.col("organism_name"),
            pl.col("infraspecific_name"),
            pl.col("assembly_level"),
            pl.col("genbankAccession"),
            pl.col("refseqAccession")
        ]

        # Collect the joined result (streaming). 
        joined_df = safe_collect(joined_lf.select(final_cols))

        # Convert to pandas
        mapping_df = joined_df.to_pandas()

    except Exception as e:
        console.print(f"[WARN] Failed Polars fetch in _fetch_nucleotide_data: {e}")
        mapping_df = pd.DataFrame()

    # 2. Apply the mapping to the main df
    if not mapping_df.empty:
        # Split into GCA/GCF logic
        # Ensure assembly_id is string
        mapping_df["assembly_id"] = mapping_df["assembly_id"].astype(str)
        
        gca_mask = mapping_df["assembly_id"].str.startswith("GCA_")
        gcf_mask = mapping_df["assembly_id"].str.startswith("GCF_")
        
        # GCA -> key is genbankAccession
        gb_map = (
            mapping_df.loc[gca_mask]
            .rename(columns={"genbankAccession": "nucleotide_id"})
            .drop(columns=["refseqAccession"], errors="ignore")
            .drop_duplicates(subset="nucleotide_id", keep="first")
            .set_index("nucleotide_id")
        )
        
        # GCF -> key is refseqAccession
        rs_map = (
            mapping_df.loc[gcf_mask]
            .rename(columns={"refseqAccession": "nucleotide_id"})
            .drop(columns=["genbankAccession"], errors="ignore")
            .drop_duplicates(subset="nucleotide_id", keep="first")
            .set_index("nucleotide_id")
        )

        # Prepare df for update
        wanted = [
            "sequence_length", "assembly_id", "taxid", "group", 
            "species_taxid", "organism_name", "infraspecific_name", "assembly_level"
        ]
        for c in wanted:
            if c not in df.columns:
                df[c] = pd.NA
        
        # We use set_index/update to respect the original logic
        df = df.set_index("nucleotide_id", drop=False)
        df.update(gb_map, overwrite=False)
        df.update(rs_map, overwrite=False)
        df = df.reset_index(drop=True)

    # 3. Fill missing sequence_length via EDirect (unchanged logic)
    missing = df["sequence_length"].isna()
    if missing.any():
        to_fetch = df.loc[missing, "nucleotide_id"].dropna().unique().tolist()
        meta = run_nuc2asmlen(to_fetch)
        meta = (
            meta.rename(
                columns={
                    "NucleotideAccession": "nucleotide_id",
                    "length": "sequence_length",
                    "AssemblyAccession": "assembly_id",
                }
            )
            .drop_duplicates(subset="nucleotide_id", keep="first")
            .set_index("nucleotide_id")
        )
        df2 = df.set_index("nucleotide_id")
        df2.update(meta[["sequence_length", "assembly_id"]])
        df = df2.reset_index()

        # backfill taxid/group for any new assemblies
        new_asms = df["assembly_id"].dropna().unique().tolist()
        if new_asms:
            try:
                ids_asm2 = pl.LazyFrame({"assembly_accession": new_asms})
                summary2_tbl = safe_collect(
                    pl.scan_parquet(summary_path)
                      .join(ids_asm2, on="assembly_accession", how="inner")
                      .select(["assembly_accession", "taxid", "group"])
                )
                summary2 = summary2_tbl.to_pandas()
                taxid_map = dict(zip(summary2["assembly_accession"], summary2["taxid"]))
                group_map = dict(zip(summary2["assembly_accession"], summary2["group"]))
                df["taxid"] = df["assembly_id"].map(taxid_map).fillna(df["taxid"])
                df["group"] = df["assembly_id"].map(group_map).fillna(df["group"])
            except Exception as e:
                console.print(f"[WARN] Failed backfill join: {e}")

    # 4. Fill start/end (unchanged logic)
    cond = (
        (df["input_type"] == "nucleotide")
        & df["nucleotide_id"].notnull()
        & df["failed"].isnull()
        & df["premade"].isnull()
        & df["start"].isnull()
        & df["end"].isnull()
    )
    df.loc[cond, "start"] = 0
    df.loc[cond, "end"] = df["sequence_length"]
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Selection & Finalization
# ──────────────────────────────────────────────────────────────────────────────

def _select_best_ipg(df: pd.DataFrame, cand_mode: str) -> pd.DataFrame:
    if "ipg_id" not in df.columns:
        return df
    mask_ipg = df["ipg_id"].notnull()
    df_no_ipg = df[~mask_ipg].copy()
    df_ipg = df[mask_ipg].copy()

    def assign_rank(row):
        asm = row.get("assembly_id", "")
        lvl = row.get("assembly_level", "")
        if pd.notna(asm):
            if asm.startswith("GCF") and lvl in ["Chromosome", "Complete Genome"]:
                return 1
            if lvl in ["Chromosome", "Complete Genome"]:
                return 2
            return 3
        return 4

    if not df_ipg.empty:
        df_ipg["ranked"] = df_ipg.apply(assign_rank, axis=1)
        key = "ipg_id" if cand_mode == "any_ipg" else "og_index"
        if cand_mode == "any_ipg":
            df_ipg = df_ipg.drop_duplicates(subset=[key, "nucleotide_id", "start", "end"], keep="first")
        elif cand_mode in ["best_ipg", "best_id"]:
            df_ipg = (
                df_ipg.sort_values(by=[key, "ranked", "sequence_length"], ascending=[True, True, False])
                .drop_duplicates(subset=[key], keep="first")
            )
        elif cand_mode == "one_id":
            df_ipg = df_ipg.drop_duplicates(subset=[key], keep="first")
        elif cand_mode == "same_id":
            df_ipg = df_ipg.drop_duplicates(subset=[key, "nucleotide_id", "start", "end"], keep="first")

    df_out = pd.concat([df_no_ipg, df_ipg], ignore_index=True).sort_index(kind="stable")
    return df_out

def _finalize_ipg(df: pd.DataFrame, cand_mode: str) -> pd.DataFrame:
    cond = (
        df["failed"].isnull()
        & (~(df["gff_path"].notnull() & df["faa_path"].notnull()) | df["assembly_id"].isnull())
        & df["premade"].isnull()
    )
    to_fail = cond & df["assembly_id"].isnull()
    df.loc[to_fail, "failed"] = "Unable to retrieve IPG/Nuccore data"

    cond = (
        df["failed"].isnull()
        & (~(df["gff_path"].notnull() & df["faa_path"].notnull()) | df["assembly_id"].isnull())
        & df["premade"].isnull()
    )
    valid = {"bacteria", "viral", "archaea", "metagenomes"}
    to_invalid = cond & (~df["group"].isin(valid))
    df.loc[to_invalid, "failed"] = "Invalid superkingdom"

    return df.where(pd.notnull(df), None)
