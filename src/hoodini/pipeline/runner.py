"""Pipeline orchestration for the hoodini CLI.

Keeps the CLI thin by encapsulating the main workflow in a single callable
that accepts a typed RuntimeConfig.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from hoodini.config import RuntimeConfig
from hoodini.utils.logging_utils import stage_done, stage_header

log = logging.getLogger(__name__)


def run_pipeline(config: RuntimeConfig) -> None:
    """Execute the hoodini workflow using the provided config."""
    # ─── Step 1: Create records dataframe from input and initialize folder ─────────────
    stage_header("Initializing Hoodini", "🚀")
    from hoodini.pipeline.initialize import initialize_inputs

    records = initialize_inputs(
        input_path=config.input_path,
        inputsheet=config.inputsheet,
        output=config.output,
        force=config.force,
    )

    stage_done("Initialization complete")

    # ─── Step 2: Fetch IPG data from NCBI ──────────────────────────
    stage_header("Parsing IPG data", "🔍")
    from hoodini.pipeline.parse_ipg import run_ipg

    records = run_ipg(
        records_df=records,
        cand_mode=config.cand_mode,
    )

    stage_done("IPG parsing complete")

    # ─── Step 3: Fetch assembly data from NCBI and extract neighborhoods ──────────
    stage_header("Downloading and parsing assemblies", "📥")
    from hoodini.pipeline.parse_assemblies import run_assembly_parser

    result = run_assembly_parser(
        records_df=records,
        output_dir=config.output,
        ncrna=config.ncrna,
        cctyper=config.cctyper,
        genomad=config.genomad,
        blast=config.blast,
        apikey=config.apikey,
        max_concurrent_downloads=config.max_concurrent_downloads,
        num_threads=config.num_threads,
        mod=config.mod,
        wn=config.wn,
        sorfs=config.sorfs,
        minwin=config.minwin,
        minwin_type=config.minwin_type,
    )

    records = result["records"]
    all_gff = result["all_gff"]
    all_prots = result["all_prots"]
    all_neigh = result["all_neigh"]
    valid_uids = result["valid_uids"]

    stage_done("Assembly parsing and neighborhood extraction complete")

    # ─── Step 4: Getting protein links ──────────
    if config.tree_mode == "aai_tree" or config.prot_links:
        stage_header("Running all-vs-all protein comparisons", "🦠")
        from hoodini.pipeline.protein_links import run_protein_links

        pairwise_aa = run_protein_links(
            output_dir=config.output,
            all_prots=all_prots,
            threads=config.num_threads,
            evalue=1e-5,
        )

        stage_done("All-vs-all protein comparisons complete")
    else:
        pairwise_aa = None

    # ─── Step 5: Getting pairwise nt comparisons ──────────
    if config.tree_mode == "ani_tree" or config.nt_links:
        stage_header("Running pairwise nucleotide comparisons", "🦠")
        from hoodini.pipeline.pairwise_nt import run_pairwise_nt

        pairwise_ani, nt_links = run_pairwise_nt(
            all_neigh=all_neigh,
            all_gff=all_gff,
            output_dir=config.output,
            nt_aln_mode=config.nt_aln_mode,
            ani_mode=config.ani_mode,
            nt_links=bool(config.nt_links),
            threads=config.num_threads,
        )

        stage_done("Pairwise nucleotide comparisons complete")
    else:
        pairwise_ani = None
        nt_links = None

    # ─── Step 5: Clustering neighbor proteins ──────────
    stage_header("Clustering neighbor proteins", "✨")
    from hoodini.pipeline.cluster_proteins import cluster_proteins

    all_prots = cluster_proteins(
        all_prots,
        output_dir=config.output,
        clust_method=config.clust_method,
        sorfs=config.sorfs,
    )

    if config.sorfs:
        discarded_sorfs = all_prots[
            (all_prots["id"].str.contains("sORF") & all_prots["fam_cluster"].is_null())
        ]
        discarded_sorfs["gff_id"] = "ID=" + discarded_sorfs["id"]
        all_prots = all_prots[~all_prots["id"].isin(discarded_sorfs["id"].unique())]
        all_gff = all_gff[~(all_gff["attributes"].isin(set(discarded_sorfs["gff_id"].unique())))]

    stage_done("Clustering complete")

    # ─── Step 6: Proteome similarity (AAI) ──────────
    if config.tree_mode == "aai_tree":
        from hoodini.pipeline.proteome_similarity import run_proteome_similarity

        stage_header("Computing proteome similarity", "🔗")
        pairwise_aai = run_proteome_similarity(
            all_prots=all_prots,
            pairwise_aa=pairwise_aa,
            all_neigh=all_neigh,
            all_gff=all_gff,
            outdir=config.output,
            mode=config.aai_mode,
            pident_min=(
                config.min_prevalence
                if hasattr(config, "min_prevalence") and config.min_prevalence is not None
                else 30.0
            ),
            subset_mode="target_region",
            win=config.wn,
            win_mode=(
                config.mod if hasattr(config, "mod") and config.mod is not None else "win_nts"
            ),
        )

        stage_done("Proteome similarity complete")
    else:
        pairwise_aai = None

    # ─── Step 7: Extracting taxonomic information ──────────
    stage_header("Extracting taxonomic information", "🦠")
    from hoodini.pipeline.taxonomy import parse_taxonomy_and_build_tree

    tree_str, den_data = parse_taxonomy_and_build_tree(
        records=records,
        all_gff=all_gff,
        all_neigh=all_neigh,
        all_prots=all_prots,
        output_dir=config.output,
        tree_mode=config.tree_mode,
        tree_file=config.tree_file,
        num_threads=config.num_threads,
        valid_uids=valid_uids,
        aai_mode=config.aai_mode,
        ani_mode=config.ani_mode,
        aai_subset_mode=config.aai_subset_mode,
        nj_algorithm=config.nj_algorithm,
        pairwise_ani=pairwise_ani,
        pairwise_aai=pairwise_aai,
    )

    # ─── Step 8: Running extra annotation tools ──────────
    stage_header("Running extra annotation tools", "🦠")

    # Domain annotation
    domains_data = None
    if config.domains:
        from hoodini.extra_tools.domain import run_domain

        domains_data = run_domain(all_prots, config.output, config.domains, config.num_threads)

    # BLAST annotation
    if config.blast:
        from hoodini.extra_tools.blast import run_blast

        blast_data = run_blast(
            all_neigh, config.output, config.blast, config.num_threads, valid_uids
        )
        if blast_data.height > 0:
            gff_df = pl.DataFrame(
                {
                    "seqid": blast_data["seqid"],
                    "source": "hoodini",
                    "type": "region",
                    "start": blast_data["start"],
                    "end": blast_data["end"],
                    "score": ".",
                    "strand": "+",
                    "phase": ".",
                    "attributes": "ID=" + blast_data["nc_feature"] + ";",
                }
            )
            all_gff = pl.concat([all_gff, gff_df], how="vertical")

    # PADLOC annotation
    if config.padloc:
        from hoodini.extra_tools.padloc import run_padloc

        padloc_df = run_padloc(all_gff, all_prots, config.output, config.num_threads)
        if padloc_df.height > 0:
            all_prots = all_prots.join(padloc_df, on="id", how="left")

    # eggNOG-mapper (emapper) annotation
    if config.emapper:
        from hoodini.extra_tools.emapper import run_emapper

        emapper_df = run_emapper(all_prots, config.output, config.num_threads)

        if emapper_df.height > 0:
            if "description" in emapper_df.columns and "product" in all_prots.columns:
                desc_map = emapper_df.set_index("id")["description"].to_dict()
                all_prots["product"] = all_prots["product"].where(
                    all_prots["product"].notna()
                    & (all_prots["product"].astype(str).str.strip() != ""),
                    all_prots["id"].map(desc_map),
                )

            if "id" in emapper_df.columns and "id" in all_prots.columns:
                all_prots = all_prots.join(emapper_df, on="id", how="left")

    # DefenseFinder annotation
    if config.deffinder:
        from hoodini.extra_tools.defensefinder import run_defensefinder

        deffinder_df = run_defensefinder(all_gff, all_prots, config.output)
        if deffinder_df.height > 0:
            all_prots = all_prots.join(deffinder_df, on="id", how="left")

    # CCTyper annotation
    if config.cctyper:
        from hoodini.extra_tools.cctyper import run_cctyper

        cctyper_df, crispr_df = run_cctyper(
            all_gff,
            all_prots,
            all_neigh,
            den_data,
            config.output,
            config.num_threads,
            valid_uids,
        )
        if cctyper_df.height > 0:
            all_prots = all_prots.join(cctyper_df, on="id", how="left")
        if crispr_df.height > 0:
            gff_df = pl.DataFrame(
                {
                    "seqid": crispr_df["Contig"],
                    "source": "hoodini",
                    "type": "region",
                    "start": crispr_df["start"],
                    "end": crispr_df["end"],
                    "score": ".",
                    "strand": ".",
                    "phase": ".",
                    "attributes": "ID=" + crispr_df["nc_feature"] + ";",
                }
            )
            all_gff = pl.concat([all_gff, gff_df], how="vertical")

    # ncRNA/Infernal annotation
    if config.ncrna:
        from hoodini.extra_tools.ncrna import run_ncrna

        ncrna_data = run_ncrna(all_neigh, den_data, config.output, config.num_threads, valid_uids)
        if ncrna_data.height > 0:
            gff_df = ncrna_data.select(
                [
                    pl.col("nucid").alias("seqid"),
                    pl.lit("hoodini").alias("source"),
                    pl.lit("ncRNA").alias("type"),
                    pl.min_horizontal([pl.col("start"), pl.col("end")]).alias("start"),
                    pl.max_horizontal([pl.col("start"), pl.col("end")]).alias("end"),
                    pl.lit(".").alias("score"),
                    pl.col("strand_ncrna").alias("strand"),
                    pl.lit(".").alias("phase"),
                    (pl.lit("ID=") + pl.col("nc_feature") + pl.lit(";")).alias("attributes"),
                ]
            )
            all_gff = pl.concat([all_gff, gff_df], how="vertical")

    # GenoMAD annotation
    if config.genomad:
        from hoodini.extra_tools.genomad import run_genomad

        genomad_df = run_genomad(all_neigh, config.output, config.num_threads, valid_uids)
        if genomad_df.height > 0:
            gff_df = genomad_df.select(
                [
                    pl.col("seqid"),
                    pl.lit("hoodini").alias("source"),
                    pl.lit("region").alias("type"),
                    pl.min_horizontal([pl.col("start"), pl.col("end")]).alias("start"),
                    pl.max_horizontal([pl.col("start"), pl.col("end")]).alias("end"),
                    pl.lit(".").alias("score"),
                    pl.lit(".Z").alias("strand"),
                    pl.lit(".").alias("phase"),
                    (pl.lit("ID=") + pl.col("mge_type") + pl.lit(";")).alias("attributes"),
                ]
            )
            all_gff = pl.concat([all_gff, gff_df], how="vertical")

    stage_done("Extra annotation complete")

    # Write outputs for viz in a single place
    from hoodini.pipeline.write_data import write_viz_outputs

    write_viz_outputs(
        output_dir=config.output,
        all_gff=all_gff,
        all_neigh=all_neigh,
        all_prots=all_prots,
        den_data=den_data,
        tree_str=tree_str,
        nt_links=nt_links,
        pairwise_aa=pairwise_aa,
        domains_data=domains_data,
        write_domains=bool(config.domains),
    )
