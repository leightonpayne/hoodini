#!/usr/bin/env python3
"""IPG (Identical Protein Groups) enrichment pipeline using Polars."""

from importlib.resources import files

import polars as pl
from rich.console import Console

from hoodini.fetch_ipg_from_accessions import fetch_ipg_from_accessions
from hoodini.models.schemas import RECORDS

PlDF = pl.DataFrame

console = Console()


def safe_collect(lf: pl.LazyFrame) -> PlDF:
    """Collect a lazy frame, preferring streaming when available."""
    try:
        return lf.collect(engine="streaming")
    except (TypeError, AttributeError):
        return lf.collect(streaming=False)
    except Exception:
        return lf.collect(streaming=False)


def run_ipg(records_df: PlDF, *, cand_mode: str) -> PlDF:
    """Polars-based IPG enrichment pipeline."""
    df = records_df.clone()
    console.print("🔍  Fetching IPG data...")
    df = _fetch_ipg_data(df, cand_mode)
    console.print("🔍  Fetching nucleotide data...")
    df = _fetch_nucleotide_data(df)
    console.print("✅  Selecting best IPG records...")
    df = _select_best_ipg(df, cand_mode)
    df = _finalize_ipg(df, cand_mode)
    return df


def _fill_ipg_polars(records: PlDF, ipg_df: PlDF) -> PlDF:
    """Fill IPG information into records (Polars version)."""
    if ipg_df.height == 0:
        return records

    # Get unique ipg_id per protein_id
    ipg_map = ipg_df.select(["protein_id", "ipg_id"]).unique(subset=["protein_id"])
    records = records.join(ipg_map, on="protein_id", how="left", suffix="_ipg")

    if "ipg_id_ipg" in records.columns:
        records = records.with_columns(
            pl.when(pl.col("ipg_id_ipg").is_not_null()).then(pl.col("ipg_id_ipg")).otherwise(pl.col("ipg_id")).alias("ipg_id")
        ).drop("ipg_id_ipg")

    # Mark as failed when enrichment eligible but no ipg found
    cond = (
        (pl.col("protein_id").is_not_null())
        & (pl.col("failed").is_null())
        & (~pl.col("premade"))
        & (pl.col("ipg_id").is_null())
    )
    records = records.with_columns(
        pl.when(cond).then(pl.lit("Unable to retrieve IPG")).otherwise(pl.col("failed")).alias("failed")
    )

    # Join full IPG info
    ipg_cols = [c for c in ipg_df.columns if c not in ["protein_id", "ipg_id"]]
    select_cols = ["ipg_id"] + ipg_cols
    if ipg_df.height > 0 and "ipg_id" in ipg_df.columns:
        select_cols_present = [c for c in select_cols if c in ipg_df.columns]
        if select_cols_present:
            ipg_subset = ipg_df.select(select_cols_present)
            records = records.join(ipg_subset, on="ipg_id", how="left", suffix="_ipg_data")

            # Fill missing values from ipg_data columns
            for col in ipg_cols:
                ipg_col = f"{col}_ipg_data"
                if ipg_col in records.columns:
                    records = records.with_columns(
                        pl.when(pl.col(col).is_null()).then(pl.col(ipg_col)).otherwise(pl.col(col)).alias(col)
                    ).drop(ipg_col)

    return records


def _fetch_ipg_data(df: PlDF, cand_mode: str) -> PlDF:
    """Fetch IPG data for proteins and enrich with assembly/taxid info."""
    cond_mask = (
        (pl.col("protein_id").is_not_null())
        & (pl.col("failed").is_null())
        & (~pl.col("premade"))  # Not premade (null or False)
    )

    proteins = df.filter(cond_mask).select("protein_id").unique().to_series().to_list()
    proteins = [p for p in proteins if p and str(p).strip()]

    if not proteins:
        console.print("ℹ️  No records match conditions for IPG.")
        return df

    ipg_df = fetch_ipg_from_accessions(proteins)
    if ipg_df.height == 0:
        console.print("ℹ️  No IPG records found.")
        return df
    console.log(f"Fetched {ipg_df.height} IPG records for {len(proteins)} proteins.")

    assemblies = ipg_df.select("assembly").unique().to_series().to_list()
    assemblies = [a for a in assemblies if a and str(a).strip()]

    if not assemblies:
        console.print("ℹ️  No assemblies found in IPG data.")
        return df

    # Enrich with dive_combined.parquet
    try:
        dive_path = files("hoodini").joinpath("data", "dive_combined.parquet")
        asm_ids = pl.DataFrame({"assembly_id": assemblies})
        ts = safe_collect(pl.scan_parquet(dive_path).join(asm_ids.lazy(), on="assembly_id", how="inner"))
        if ts.height > 0:
            ipg_df = ipg_df.join(ts, left_on="assembly", right_on="assembly_id", how="left")
    except Exception as e:
        console.print(f"[WARN] Skipping dive_combined.parquet: {e}")

    # Enrich with assembly_summary.parquet
    try:
        summary_path = files("hoodini").joinpath("data", "assembly_summary.parquet")
        asm_ids2 = pl.DataFrame({"assembly_accession": assemblies})
        summary = safe_collect(pl.scan_parquet(summary_path).join(asm_ids2.lazy(), on="assembly_accession", how="inner"))
        keep_cols = ["assembly_accession", "taxid", "species_taxid", "organism_name", "infraspecific_name", "assembly_level", "group"]
        summary = summary.select([c for c in keep_cols if c in summary.columns])
        if summary.height > 0:
            ipg_df = ipg_df.join(summary, left_on="assembly", right_on="assembly_accession", how="left")
    except Exception as e:
        console.print(f"[WARN] Skipping assembly_summary.parquet: {e}")

    # Normalize column names
    rename_map = {
        "nucleotide accession": "nucleotide_id",
        "assembly": "assembly_id",
        "stop": "end",
        "protein": "protein_id",
        "id": "ipg_id",
    }
    ipg_df = ipg_df.rename({k: v for k, v in rename_map.items() if k in ipg_df.columns})

    # Merge IPG info
    df = _fill_ipg_polars(records=df, ipg_df=ipg_df)

    # Fix assembly prefix consistency
    from hoodini.utils.core import is_refseq_nuccore, switch_assembly_prefix

    def fix_asm(nuc_id: str | None, asm_id: str | None) -> str | None:
        if nuc_id is None or asm_id is None:
            return asm_id
        if is_refseq_nuccore(nuc_id) and str(asm_id).startswith("GCA_"):
            return switch_assembly_prefix(asm_id)
        elif not is_refseq_nuccore(nuc_id) and str(asm_id).startswith("GCF_"):
            return switch_assembly_prefix(asm_id)
        return asm_id

    mask = (
        (pl.col("failed").is_null())
        & (~pl.col("premade"))
        & (pl.col("nucleotide_id").is_not_null())
    )

    df = df.with_columns(
        pl.when(mask).then(
            pl.concat_list(["nucleotide_id", "assembly_id"]).map_elements(lambda x: fix_asm(x[0], x[1]), return_dtype=pl.Utf8)
        ).otherwise(pl.col("assembly_id")).alias("assembly_id")
    )

    return df


def _fetch_nucleotide_data(df: PlDF) -> PlDF:
    """Fetch nucleotide sequence lengths and assembly metadata."""
    nucs = df.select("nucleotide_id").unique().to_series().to_list()
    nucs = [n for n in nucs if n and str(n).strip()]

    if not nucs:
        console.print("ℹ️  No nucleotide IDs to fetch.")
        return df

    base = files("hoodini").joinpath("data")
    contig_path = base / "contig_lengths"
    summary_path = base / "assembly_summary.parquet"

    # Build mapping in Polars
    try:
        contigs_lf = pl.scan_parquet(contig_path, allow_missing_columns=True).filter(
            (pl.col("genbankAccession").is_in(nucs)) | (pl.col("refseqAccession").is_in(nucs))
        )

        summary_scan = pl.scan_parquet(summary_path).select([
            "assembly_accession", "taxid", "species_taxid", "organism_name",
            "infraspecific_name", "assembly_level", "group"
        ])

        joined_lf = contigs_lf.join(summary_scan, left_on="assemblyAccession", right_on="assembly_accession", how="left")

        mapping_df = safe_collect(joined_lf.select([
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
        ]))

    except Exception as e:
        console.print(f"[WARN] Failed Polars fetch in _fetch_nucleotide_data: {e}")
        mapping_df = pl.DataFrame()

    # Apply the mapping
    if mapping_df.height > 0:
        mapping_df = mapping_df.with_columns(pl.col("assembly_id").cast(pl.Utf8))

        # GCA -> key is genbankAccession
        gca_map = mapping_df.filter(pl.col("assembly_id").str.starts_with("GCA_")).select([
            pl.col("genbankAccession").alias("nucleotide_id"),
            "assembly_id", "sequence_length", "taxid", "group",
            "species_taxid", "organism_name", "infraspecific_name", "assembly_level"
        ]).unique(subset=["nucleotide_id"], keep="first")

        # GCF -> key is refseqAccession
        gcf_map = mapping_df.filter(pl.col("assembly_id").str.starts_with("GCF_")).select([
            pl.col("refseqAccession").alias("nucleotide_id"),
            "assembly_id", "sequence_length", "taxid", "group",
            "species_taxid", "organism_name", "infraspecific_name", "assembly_level"
        ]).unique(subset=["nucleotide_id"], keep="first")

        # Ensure columns exist
        for c in ["sequence_length", "assembly_id", "taxid", "group", "species_taxid", "organism_name", "infraspecific_name", "assembly_level"]:
            if c not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8 if c not in ["taxid"] else pl.Int64).alias(c))

        # Apply GCA map
        df = df.join(gca_map, on="nucleotide_id", how="left", suffix="_gca")
        for col in gca_map.columns:
            if col != "nucleotide_id" and f"{col}_gca" in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col).is_null()).then(pl.col(f"{col}_gca")).otherwise(pl.col(col)).alias(col)
                ).drop(f"{col}_gca")

        # Apply GCF map
        df = df.join(gcf_map, on="nucleotide_id", how="left", suffix="_gcf")
        for col in gcf_map.columns:
            if col != "nucleotide_id" and f"{col}_gcf" in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col).is_null()).then(pl.col(f"{col}_gcf")).otherwise(pl.col(col)).alias(col)
                ).drop(f"{col}_gcf")

    # Fill missing sequence_length via EDirect
    missing_mask = df.select(pl.col("sequence_length").is_null()).to_series()
    if missing_mask.any():
        to_fetch = df.filter(missing_mask).select("nucleotide_id").unique().to_series().to_list()
        to_fetch = [t for t in to_fetch if t and str(t).strip()]
        if to_fetch:
            meta = run_nuc2asmlen(to_fetch)
            meta = to_polars(meta)
            if meta.height > 0:
                meta = meta.rename({
                    "NucleotideAccession": "nucleotide_id",
                    "length": "sequence_length",
                    "AssemblyAccession": "assembly_id",
                }).unique(subset=["nucleotide_id"], keep="first")
                df = df.join(meta, on="nucleotide_id", how="left", suffix="_edirect")
                for col in ["sequence_length", "assembly_id"]:
                    if f"{col}_edirect" in df.columns:
                        df = df.with_columns(
                            pl.when(pl.col(col).is_null()).then(pl.col(f"{col}_edirect")).otherwise(pl.col(col)).alias(col)
                        ).drop(f"{col}_edirect")

                # Backfill taxid/group for new assemblies
                new_asms = df.filter(pl.col("assembly_id").is_not_null()).select("assembly_id").unique().to_series().to_list()
                if new_asms:
                    try:
                        ids_asm = pl.DataFrame({"assembly_accession": new_asms})
                        summary2 = safe_collect(pl.scan_parquet(summary_path).join(ids_asm.lazy(), on="assembly_accession", how="inner"))
                        if summary2.height > 0:
                            summary2 = summary2.select(["assembly_accession", "taxid", "group"]).rename({"assembly_accession": "assembly_id"})
                            df = df.join(summary2, on="assembly_id", how="left", suffix="_backfill")
                            for col in ["taxid", "group"]:
                                if f"{col}_backfill" in df.columns:
                                    df = df.with_columns(
                                        pl.when(pl.col(col).is_null()).then(pl.col(f"{col}_backfill")).otherwise(pl.col(col)).alias(col)
                                    ).drop(f"{col}_backfill")
                    except Exception as e:
                        console.print(f"[WARN] Failed backfill join: {e}")

    # Fill start/end for nucleotide input type
    cond = (
        (pl.col("input_type") == "nucleotide")
        & (pl.col("nucleotide_id").is_not_null())
        & (pl.col("failed").is_null())
        & (~pl.col("premade"))
        & (pl.col("start").is_null())
        & (pl.col("end").is_null())
    )
    df = df.with_columns(
        pl.when(cond).then(pl.lit(0)).otherwise(pl.col("start")).alias("start"),
        pl.when(cond).then(pl.col("sequence_length")).otherwise(pl.col("end")).alias("end"),
    )

    return df


def _select_best_ipg(df: pl.DataFrame, cand_mode: str) -> pl.DataFrame:
    """Select best IPG record per og_index based on ranking."""
    if "ipg_id" not in df.columns:
        return df

    mask_ipg = df.select(pl.col("ipg_id").is_not_null()).to_series()
    df_no_ipg = df.filter(~mask_ipg)
    df_ipg = df.filter(mask_ipg)

    if df_ipg.height == 0:
        return df

    def rank_func(asm, lvl):
        if asm and str(asm).startswith("GCF") and lvl in ["Chromosome", "Complete Genome"]:
            return 1
        elif lvl in ["Chromosome", "Complete Genome"]:
            return 2
        elif asm:
            return 3
        else:
            return 4

    df_ipg = df_ipg.with_columns(
        pl.concat_list(["assembly_id", "assembly_level"]).map_elements(
            lambda x: rank_func(x[0], x[1]), return_dtype=pl.Int32
        ).alias("ranked")
    )

    if cand_mode == "any_ipg":
        df_ipg = df_ipg.unique(subset=["ipg_id", "nucleotide_id", "start", "end"], keep="first")
    elif cand_mode in ["best_ipg", "best_id"]:
        df_ipg = (
            df_ipg.with_columns(pl.col("sequence_length").fill_null(0).cast(pl.Int64).alias("_seq_len"))
            .sort(["og_index", "ranked", "_seq_len"], descending=[False, False, True])
            .unique(subset=["og_index"], keep="first")
            .drop(["_seq_len", "ranked"])
        )
    elif cand_mode == "one_id":
        df_ipg = df_ipg.unique(subset=["og_index"], keep="first").drop("ranked")
    elif cand_mode == "same_id":
        df_ipg = df_ipg.unique(subset=["og_index", "nucleotide_id", "start", "end"], keep="first").drop("ranked")

    return pl.concat([df_no_ipg, df_ipg], how="vertical")


def _finalize_ipg(df: pl.DataFrame, cand_mode: str) -> pl.DataFrame:
    """Mark records as failed if they don't meet requirements."""
    # Ensure all required columns exist
    for col in ["group", "gff_path", "faa_path", "assembly_id", "failed", "premade"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    cond1 = (
        (pl.col("failed").is_null())
        & (~((pl.col("gff_path").is_not_null() & pl.col("faa_path").is_not_null()) | pl.col("assembly_id").is_null()))
        & (~pl.col("premade"))
    )
    to_fail1 = cond1 & pl.col("assembly_id").is_null()
    df = df.with_columns(
        pl.when(to_fail1).then(pl.lit("Unable to retrieve IPG/Nuccore data")).otherwise(pl.col("failed")).alias("failed")
    )

    valid = {"bacteria", "viral", "archaea", "metagenomes"}
    cond2 = (
        (pl.col("failed").is_null())
        & (~((pl.col("gff_path").is_not_null() & pl.col("faa_path").is_not_null()) | pl.col("assembly_id").is_null()))
        & (~pl.col("premade"))
    )
    to_invalid = cond2 & (~pl.col("group").is_in(valid))
    df = df.with_columns(
        pl.when(to_invalid).then(pl.lit("Invalid superkingdom")).otherwise(pl.col("failed")).alias("failed")
    )

    return df
