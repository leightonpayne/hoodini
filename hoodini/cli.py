# hoodini/cli.py

import os
import logging
from pathlib import Path
from typing import Optional

import polars as pl
import rich_click as click
import tomli
from rich.logging import RichHandler
from click.core import ParameterSource

from hoodini.config import load_default_config
from hoodini.utils.cli_helpers import MutuallyExclusiveOption
from hoodini.utils.core import validate_input_file, validate_domains
from hoodini.utils.logging_utils import (
    console,
    stage_header,
    stage_done,
    run_with_spinner,
)

stage_header("Initializing Hoodini", "🚀")


click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_ERRORS_SUGGESTIONS = "yellow"
click.rich_click.STYLE_HELP_OPTIONS = "bold cyan"
click.rich_click.STYLE_HELP_OPTIONS_DEFAULTS = "dim"


@click.group()
def cli():
    """🦉 hoodini: gene-centric comparative genomic analysis using publicly available data"""
    pass


@cli.command()
@click.option(
    "--config", "config_file",
    default=None,
    type=click.Path(exists=True),
    help="TOML config file to load parameters from."
)
@click.option(
    "--input",
    "input_path",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["inputsheet"],
    default=None,
    type=click.Path(exists=True),
    callback=validate_input_file,
    help="Path to a single-column input file (mutually exclusive with --inputsheet)."
)
@click.option(
    "--inputsheet",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["input_path"],
    default=None,
    type=click.Path(exists=True),
    callback=validate_input_file,
    help="Path to a TSV input file (mutually exclusive with --input)."
)
@click.option("--output", help="Output folder name.")
@click.option("--max-concurrent-downloads", "max_concurrent_downloads", type=int, help="Maximum concurrent downloads.")
@click.option("--api-key", "apikey", help="NCBI API key.")
@click.option("--num-threads", "num_threads", type=int, help="Number of threads.")
@click.option("--assembly-folder", "assembly_folder", help="Path to a local assembly folder.")
@click.option("--assembly-db", "assembly_db", help="Assembly DB path.")
@click.option("--img-db", "img_db", help="IMG database file (proteins).")
@click.option("--img-nuc", "img_nuc", help="IMG database file (nucleotides).")
@click.option("--prot-links", "prot_links", is_flag=True, help="Run pairwise protein comparisons.")
@click.option("--nt-links", "nt_links", is_flag=True, help="Run pairwise nucleotide comparisons.")
@click.option("--ani-mode", "ani_mode", help="Choose ANI calculation method.")
@click.option("--nt-aln-mode", "nt_aln_mode", type=click.Choice(["blastn", "fastani", "minimap2", "intergenic_blastn"]), help="Nucleotide alignment mode to use for pairwise comparisons: 'blastn' or 'fastani'.")
@click.option("--blast", help="BLAST query file to use.")
@click.option("--cand-mode", "cand_mode", help="Mode for selecting IPG representative.")
@click.option("--clust-method", "clust_method", help="Protein clustering method.")
@click.option("--win-mode", "mod", help="Window mode: 'win_nts' or 'win_genes'.")
@click.option("--win", "wn", type=int, help="Window size (genes or nucleotides).")
@click.option("--height-factor", "height_factor", type=int, help="Height factor for plotting.")
@click.option("--ngenes", type=int, help="Number of genes in context window.")
@click.option("--min-win", "minwin", type=int, help="Min window size on each side.")
@click.option("--min-win-type", "minwin_type", type=click.Choice(["total", "upstream", "downstream", "both"]), help="Type of min window.")
@click.option("--tree-mode", "tree_mode", help="Tree building method.")
@click.option("--tree-file", "tree_file", help="Path to the tree file.")
@click.option("--aai-mode", "aai_mode", help="Mode for AAI tree construction (e.g. 'nj' or 'hyper' — 'hyper' will be rejected for AAI trees).")
@click.option("--aai-subset-mode", "aai_subset_mode", help="Subset mode to use for AAI tree construction (e.g. 'target_region', 'target_prot', 'window').")
@click.option("--padloc", is_flag=True, help="Run PADLOC for antiphage defense.")
@click.option("--deffinder", is_flag=True, help="Run DefenseFinder for antiphage defense.")
@click.option("--ncrna", is_flag=True, help="Run Infernal for ncRNA prediction.")
@click.option("--cctyper", is_flag=True, help="Run CCtyper for CRISPR-Cas prediction.")
@click.option("--genomad", is_flag=True, help="Run GenoMAD for MGE identification.")
@click.option("--antidefense", is_flag=True, help="Identify anti-defense or ACR genes.")
@click.option("--phrogs", is_flag=True, help="Annotate with PHROGs.")
@click.option("--sorfs", is_flag=True, help="Reannotate small open reading frames.")
@click.option("--domains", callback=validate_domains, help="Comma-separated list of MetaCerberus domain databases.")
@click.option("--emapper", "emapper", is_flag=True, default=False, help="Run eggNOG-mapper (emapper) to annotate proteins and append annotations to protein metadata.")
@click.option("--min-prevalence", "min_prevalence", type=float, help="Min prevalence threshold for gene coloring.")
@click.option("--img", help="Path to IMG database files (proteins).")
@click.option("--img-metadata", "img_metadata", help="Path to IMG database metadata.")
@click.option("--keep", is_flag=True, help="Keep temporary files (do not delete).")
@click.option("--force", is_flag=True, help="Overwrite existing output folder if it exists.")
@click.pass_context

def run(ctx, config_file: Optional[str], **cli_kwargs) -> None:
    """
    Run hoodini with default parameters or from a config file.
    """
    logging.basicConfig(level="NOTSET", format="%(message)s", handlers=[RichHandler()])

    # Load user config file if provided
    file_kwargs = {}
    if config_file:
        with open(config_file, "rb") as f:
            grouped = tomli.load(f)
            file_kwargs = {k: v for section in grouped.values() for k, v in section.items()}

    # Keep only CLI args explicitly set by user
    cli_clean = {}
    for k, v in cli_kwargs.items():
        src = ctx.get_parameter_source(k)
        if src in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP):
            continue
        cli_clean[k] = v

    # Merge: bundled defaults.toml < user config < CLI
    merged_params = {**load_default_config(), **file_kwargs, **cli_clean}

    # Ensure runtime-only keys exist
    merged_params.setdefault("input_path", None)
    merged_params.setdefault("inputsheet", None)

    # Wrap in simple object for attribute-style access
    class ConfigObj(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    config = ConfigObj(merged_params)
    
    config.ani_mode = config.ani_mode if config.tree_mode == "ani_tree" else None
    config.nt_aln_mode = config.nt_aln_mode if (config.nt_links or config.tree_mode == "ani_tree") else None
    
    if not config.input_path and not config.inputsheet:
        raise click.UsageError("One of '--input' or '--inputsheet' must be provided.")
    
    ctx.ensure_object(dict)
    ctx.obj["config"] = config

    # ─── Step 1: Create records dataframe from input and initialize folder ─────────────
        
    from hoodini.initialize import initialize_inputs
    records = initialize_inputs(
            input_path=config.input_path,
            inputsheet=config.inputsheet,
            output=config.output,
            force=config.force,
        )
    
    stage_done("Initialization complete")

    # ─── Step 2: Fetch IPG data from NCBI ──────────────────────────

    stage_header("Parsing IPG data", "🔍")
    
    from hoodini.parse_ipg import run_ipg
    records = run_ipg(
        records_df=records,
        cand_mode=config.cand_mode,
    )
    
    stage_done("IPG parsing complete")
    
    # ─── Step 3: Fetch assembly data from NCBI and extract neighborhoods ──────────

    stage_header("Downloading and parsing assemblies", "📥")

    from hoodini.parse_assemblies import run_assembly_parser
    result = run_assembly_parser(
        records_df=records,
        output_dir=config.output,
        ncrna=config.ncrna,
        cctyper=config.cctyper,
        genomad=config.genomad,
        blast=config.blast,
        apikey=config.apikey,
        max_concurrent_downloads=config.max_concurrent_downloads,
        img=config.img,
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
    print(config.tree_mode)
    # if --tree-mode is aai or --prot-links is set to True
    if config.tree_mode == "aai_tree" or config.prot_links:
        
        stage_header("Running all-vs-all protein comparisons", "🦠")

        from hoodini.protein_links import run_protein_links
        pairwise_aa = run_protein_links(
            output_dir=config.output,
            all_prots=all_prots,
            threads=config.num_threads,
            evalue=1e-5
        )

        stage_done("All-vs-all protein comparisons complete")

    # proteome similarity will be computed after clustering (needs fam_cluster)


    # ─── Step 5: Getting pairwise nt comparisons ──────────

    if config.tree_mode == "ani_tree" or config.nt_links:
        
        stage_header("Running pairwise nucleotide comparisons", "🦠")

        # Use the unified pairwise NT runner which supports blastn and fastANI flows.
        from hoodini.pairwise_nt import run_pairwise_nt
        print(config.nt_aln_mode)        
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
    
    from hoodini.cluster_proteins import cluster_proteins  # assuming it's saved here

    all_prots = cluster_proteins(
        all_prots,
        output_dir=config.output,
        clust_method=config.clust_method,
        sorfs=config.sorfs
    )
    
    print(all_prots)
    
    if config.sorfs:
        discarded_sorfs = all_prots[(all_prots["id"].str.contains("sORF") & all_prots["fam_cluster"].is_null())]
        discarded_sorfs["gff_id"] = "ID=" + discarded_sorfs["id"]
        all_prots = all_prots[~all_prots["id"].isin(discarded_sorfs["id"].unique())]
        #remove all gff in which attributes contains discarded_orfs
        all_gff = all_gff[~(all_gff["attributes"].isin(set(discarded_sorfs["gff_id"].unique())))]

    stage_done("Clustering complete")

    if config.tree_mode == "aai_tree":

        # --- Proteome similarity (wGRR / AAI / vContact2-style) ---
        from hoodini.run_proteome_similarity import run_proteome_similarity
        stage_header("Computing proteome similarity", "🔗")
        pairwise_aai = run_proteome_similarity(
            all_prots=all_prots,
            pairwise_aa=pairwise_aa if 'pairwise_aa' in locals() else None,
            all_neigh=all_neigh,
            all_gff=all_gff,
            outdir=config.output,
            mode=config.aai_mode,
            pident_min=(config.min_prevalence if hasattr(config, 'min_prevalence') and config.min_prevalence is not None else 30.0),
            subset_mode="target_region",
            win=config.wn,
            win_mode=(config.mod if hasattr(config, 'mod') and config.mod is not None else 'win_nts'),
        )

        stage_done("Proteome similarity complete")
        
    else:
        pairwise_aai = None

    # ─── Step 6: Extracting taxonomic information ──────────

    stage_header("Extracting taxonomic information", "🦠")

    from hoodini.taxonomy import parse_taxonomy_and_build_tree

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
        pairwise_aai=pairwise_aai
    )
    
    # ─── Step 7: Running extra annotation tools ──────────

    stage_header("Running extra annotation tools", "🦠")
    
    # Domain annotation
    if config.domains:
        from hoodini.extra_tools.domain import run_domain
        # config.domains is already a validated list of database names
        domains_data = run_domain(all_prots, config.output, config.domains, config.num_threads)

    # BLAST annotation
    if config.blast:
        
        from hoodini.extra_tools.blast import run_blast
        blast_data = run_blast(all_neigh, config.output, config.blast, config.num_threads, valid_uids)
        if blast_data.height > 0:
            gff_df = pl.DataFrame({
                "seqid": blast_data["seqid"],
                "source": "hoodini",
                "type": "region",
                "start": blast_data["start"],
                "end": blast_data["end"],
                "score": ".",
                "strand": "+",
                "phase": ".",
                "attributes": "ID=" + blast_data["nc_feature"] + ";"
            })
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
            # If description present and product missing, fill product with description
            if "description" in emapper_df.columns and "product" in all_prots.columns:
                desc_map = emapper_df.set_index("id")["description"].to_dict()
                # vectorised: fill product where empty with description
                all_prots["product"] = all_prots["product"].where(
                    all_prots["product"].notna() & (all_prots["product"].astype(str).str.strip() != ""),
                    all_prots["id"].map(desc_map)
                )

            # merge other columns; keep emapper column names as-is
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
        cctyper_df, crispr_df = run_cctyper(all_gff, all_prots, all_neigh, den_data, config.output, config.num_threads, valid_uids)
        if cctyper_df.height > 0:
            all_prots = all_prots.join(cctyper_df, on="id", how="left")
        if crispr_df.height > 0:
            gff_df = pl.DataFrame({
                "seqid": crispr_df["Contig"],
                "source": "hoodini",
                "type": "region",
                "start": crispr_df["start"],
                "end": crispr_df["end"],
                "score": ".",
                "strand": ".",
                "phase": ".",
                "attributes": "ID=" + crispr_df["nc_feature"] + ";"
            })
            #append to all_gff
            all_gff = pl.concat([all_gff, gff_df], how="vertical")
        
        
    # ncRNA/Infernal annotation
    if config.ncrna:
        from hoodini.extra_tools.ncrna import run_ncrna
        ncrna_data = run_ncrna(all_neigh, den_data, config.output, config.num_threads, valid_uids)
        if ncrna_data.height > 0:
            gff_df = ncrna_data.select([
                pl.col("nucid").alias("seqid"),
                pl.lit("hoodini").alias("source"),
                pl.lit("ncRNA").alias("type"),
                pl.min_horizontal([pl.col("start"), pl.col("end")]).alias("start"),
                pl.max_horizontal([pl.col("start"), pl.col("end")]).alias("end"),
                pl.lit(".").alias("score"),
                pl.col("strand_ncrna").alias("strand"),
                pl.lit(".").alias("phase"),
                (pl.lit("ID=") + pl.col("nc_feature") + pl.lit(";")).alias("attributes"),
            ])
            all_gff = pl.concat([all_gff, gff_df], how="vertical")

    # GenoMAD annotation
    if config.genomad:
        from hoodini.extra_tools.genomad import run_genomad
        genomad_df = run_genomad(all_neigh, config.output, config.num_threads, valid_uids)
        print(genomad_df)
        if genomad_df.height > 0:
            gff_df = genomad_df.select([
                pl.col("seqid"),
                pl.lit("hoodini").alias("source"),
                pl.lit("region").alias("type"),
                pl.min_horizontal([pl.col("start"), pl.col("end")]).alias("start"),
                pl.max_horizontal([pl.col("start"), pl.col("end")]).alias("end"),
                pl.lit(".").alias("score"),
                pl.lit(".Z").alias("strand"),
                pl.lit(".").alias("phase"),
                (pl.lit("ID=") + pl.col("mge_type") + pl.lit(";")).alias("attributes"),
            ])
            all_gff = pl.concat([all_gff, gff_df], how="vertical")

        
        
    stage_done("Extra annotation complete")

        
        
    # Write outputs for viz in a single place
    from hoodini.write_data import write_viz_outputs
    write_viz_outputs(
        output_dir=config.output,
        all_gff=all_gff,
        all_neigh=all_neigh,
        all_prots=all_prots,
        den_data=den_data,
        tree_str=tree_str,
        nt_links=(nt_links if 'nt_links' in locals() else None),
        pairwise_aa=(pairwise_aa if 'pairwise_aa' in locals() else None),
        domains_data=(domains_data if 'domains_data' in locals() else None),
        write_domains=bool(config.domains),
    )


@cli.group()
def download():
    """Download resources used by Hoodini."""
    pass


@download.command("assembly_summary")
def download_assembly_summary():
    """Download or update the assembly_summary.parquet database."""
    from hoodini.download.assembly_summary import download_assembly_summary_db
    stage_header("Downloading NCBI Assembly Database", "📥")
    asm_summary_path = download_assembly_summary_db()
    stage_done(f"Saved to {asm_summary_path}")


@download.command("metacerberus")
@click.argument("dbs", required=False, default="all")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def download_metacerberus(dbs, force):
    """Download MetaCerberus HMM/TSV databases from OSF.io."""
    from hoodini.download.metacerberus import main as metacerberus_main
    # Treat empty string or 'all' as 'all' (list all, do not download)
    if not dbs or dbs.strip().lower() == "all":
        metacerberus_main(None, force=force)
    else:
        metacerberus_main(dbs, force=force)
        stage_done(f"MetaCerberus download of {dbs} complete!")


@download.command("type_dive")
def download_type_dive():
    """Download and normalize BacDive and PhageDive DSMZ databases."""
    from hoodini.download.type_dive import main as type_dive_main
    stage_header("Downloading DSMZ BacDive/PhageDive databases", "🦠")
    type_dive_main()
    stage_done("DSMZ BacDive/PhageDive download and normalization complete!")

@download.command("contig_lengths")
@click.option(
    "--api-key", "api_key",
    default=os.environ.get("NCBI_API_KEY", None),
    help="NCBI API key (overrides environment variable NCBI_API_KEY)."
)
@click.option(
    "--skip-assembly-summary",
    "skip_assembly_summary",
    is_flag=True,
    default=False,
    help="Skip refreshing local assembly_summary.parquet (use existing local copy)."
)
def download_contig_lengths(api_key, skip_assembly_summary):
    """Download missing NCBI contig length records and update precomputed list."""
    stage_header("Downloading NCBI contig lengths", "📥")
    from hoodini.download.contig_lengths import download_contig_lengths as impl
    impl(api_key=api_key, skip_assembly_summary=skip_assembly_summary)
    stage_done("NCBI contig length download complete")



@download.command("databases")
@click.option("--force", is_flag=True, help="Force re-download of files")
@click.option("--skip-padloc", is_flag=True, help="Skip padloc DB update.")
@click.option("--skip-deffinder", is_flag=True, help="Skip defense-finder model install.")
@click.option("--skip-genomad", is_flag=True, help="Skip GenoMAD download.")
@click.option("--skip-emapper", is_flag=True, help="Skip downloading emapper/mmseqs DB.")
@click.option("--skip-parquet", is_flag=True, help="Skip downloading eggNOG parquet support files.")
@click.option("--skip-contig-lengths", is_flag=True, help="Skip downloading contig_lengths.parquet.")
@click.option("--threads", "num_threads", type=int, default=0, help="Number of threads for aria2c and pigz (0 = use all cores).")
def download_databases(force, skip_padloc, skip_deffinder, skip_genomad, skip_emapper, skip_parquet, skip_contig_lengths, num_threads):
    """Download support databases used by some extra tools (emapper, PADLOC models, etc.)"""
    from hoodini.download.databases import main as db_main
    db_main(
        force=force,
        skip_padloc=skip_padloc,
        skip_deffinder=skip_deffinder,
        skip_genomad=skip_genomad,
        skip_emapper=skip_emapper,
        skip_parquet=skip_parquet,
        skip_contig_lengths=skip_contig_lengths,
        num_threads=num_threads,
    )


@cli.group()
def utils():
    """Utility commands for Hoodini (e.g., sequence metadata helpers)."""
    pass

@utils.command("nuc2asmlen")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "output_file",
    type=click.Path(),
    default=None,
    help="Optional output file (TSV). If not set, prints to stdout."
)
def nuc2asmlen(input_file, output_file):
    """Fetch assembly and length metadata for nuccore/contig accessions (with fallback)."""
    from hoodini.nuc2asmlen import run_nuc2asmlen
    import sys

    df = run_nuc2asmlen(input_file)

    if output_file:
        df.write_csv(output_file, separator="\t")
        console.print(f"[green]Saved results to {output_file}[/green]")
    else:
        sys.stdout.write(df.write_csv(separator="\t", include_header=False))


@utils.command("prefetch_links")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "output_file", type=click.Path(), default=None, help="Optional output file (TSV). If not set, prints to stdout.")
@click.option("--kinds", "kinds", default="gbff,gff,fna,faa,sequence_report", help="Comma-separated file kinds to generate links for.")
def prefetch_links(input_file, output_file, kinds):
    """Generate prefetched links (assembly_id, file_type, link) for a list of assemblies."""
    import sys
    from hoodini.prefetch_links import get_prefetched_link_table

    with open(input_file, "r") as fh:
        accs = [l.strip() for l in fh if l.strip()]

    kinds_list = [k.strip() for k in kinds.split(",") if k.strip()]
    # pass through api_fallback flag
    df = get_prefetched_link_table(accs, kinds=kinds_list)
    if output_file:
        df.write_csv(output_file, separator="\t", include_header=False, columns=["assembly_id", "file_type", "link"])
        console.print(f"[green]Saved prefetched links to {output_file}[/green]")
    else:
        print(df.write_csv(separator="\t", include_header=False, columns=["assembly_id", "file_type", "link"]))

if __name__ == "__main__":
    cli()

main = cli
