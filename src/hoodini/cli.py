# hoodini/cli.py

import logging
import os
import sys
from typing import Optional

from click.core import ParameterSource
from rich.logging import RichHandler
import rich_click as click
import tomli

from hoodini.config import build_runtime_config, load_default_config
from hoodini.pipeline.runner import run_pipeline
from hoodini.utils.cli_helpers import MutuallyExclusiveOption
from hoodini.utils.logging_utils import console, stage_done, stage_header
from hoodini.utils.validation import validate_domains, validate_input_file

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
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True),
    help="TOML config file to load parameters from.",
)
@click.option(
    "--input",
    "input_path",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["inputsheet"],
    default=None,
    type=click.Path(exists=True),
    callback=validate_input_file,
    help="Path to a single-column input file (mutually exclusive with --inputsheet).",
)
@click.option(
    "--inputsheet",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["input_path"],
    default=None,
    type=click.Path(exists=True),
    callback=validate_input_file,
    help="Path to a TSV input file (mutually exclusive with --input).",
)
@click.option("--output", help="Output folder name.")
@click.option(
    "--max-concurrent-downloads",
    "max_concurrent_downloads",
    type=int,
    help="Maximum concurrent downloads.",
)
@click.option("--api-key", "apikey", help="NCBI API key.")
@click.option("--num-threads", "num_threads", type=int, help="Number of threads.")
@click.option("--assembly-folder", "assembly_folder", help="Path to a local assembly folder.")
@click.option("--assembly-db", "assembly_db", help="Assembly DB path.")
@click.option("--prot-links", "prot_links", is_flag=True, help="Run pairwise protein comparisons.")
@click.option("--nt-links", "nt_links", is_flag=True, help="Run pairwise nucleotide comparisons.")
@click.option("--ani-mode", "ani_mode", help="Choose ANI calculation method.")
@click.option(
    "--nt-aln-mode",
    "nt_aln_mode",
    type=click.Choice(["blastn", "fastani", "minimap2", "intergenic_blastn"]),
    help="Nucleotide alignment mode to use for pairwise comparisons: 'blastn' or 'fastani'.",
)
@click.option("--blast", help="BLAST query file to use.")
@click.option("--cand-mode", "cand_mode", help="Mode for selecting IPG representative.")
@click.option("--clust-method", "clust_method", help="Protein clustering method.")
@click.option("--win-mode", "mod", help="Window mode: 'win_nts' or 'win_genes'.")
@click.option("--win", "wn", type=int, help="Window size (genes or nucleotides).")
@click.option("--height-factor", "height_factor", type=int, help="Height factor for plotting.")
@click.option("--ngenes", type=int, help="Number of genes in context window.")
@click.option("--min-win", "minwin", type=int, help="Min window size on each side.")
@click.option(
    "--min-win-type",
    "minwin_type",
    type=click.Choice(["total", "upstream", "downstream", "both"]),
    help="Type of min window.",
)
@click.option("--tree-mode", "tree_mode", help="Tree building method.")
@click.option("--tree-file", "tree_file", help="Path to the tree file.")
@click.option(
    "--aai-mode",
    "aai_mode",
    help="Mode for AAI tree construction (e.g. 'nj' or 'hyper' — 'hyper' will be rejected for AAI trees).",
)
@click.option(
    "--aai-subset-mode",
    "aai_subset_mode",
    help="Subset mode to use for AAI tree construction (e.g. 'target_region', 'target_prot', 'window').",
)
@click.option("--padloc", is_flag=True, help="Run PADLOC for antiphage defense.")
@click.option("--deffinder", is_flag=True, help="Run DefenseFinder for antiphage defense.")
@click.option("--ncrna", is_flag=True, help="Run Infernal for ncRNA prediction.")
@click.option("--cctyper", is_flag=True, help="Run CCtyper for CRISPR-Cas prediction.")
@click.option("--genomad", is_flag=True, help="Run GenoMAD for MGE identification.")
@click.option("--sorfs", is_flag=True, help="Reannotate small open reading frames.")
@click.option(
    "--domains",
    callback=validate_domains,
    help="Comma-separated list of MetaCerberus domain databases.",
)
@click.option(
    "--emapper",
    "emapper",
    is_flag=True,
    default=False,
    help="Run eggNOG-mapper (emapper) to annotate proteins and append annotations to protein metadata.",
)
@click.option(
    "--min-prevalence",
    "min_prevalence",
    type=float,
    help="Min prevalence threshold for gene coloring.",
)
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

    config = build_runtime_config(
        defaults=load_default_config(),
        file_overrides=file_kwargs,
        cli_overrides=cli_clean,
    )

    config = config.replace(
        ani_mode=config.ani_mode if config.tree_mode == "ani_tree" else None,
        nt_aln_mode=(
            config.nt_aln_mode if (config.nt_links or config.tree_mode == "ani_tree") else None
        ),
    )

    if not config.input_path and not config.inputsheet:
        raise click.UsageError("One of '--input' or '--inputsheet' must be provided.")

    ctx.ensure_object(dict)
    ctx.obj["config"] = config

    run_pipeline(config)


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
    "--api-key",
    "api_key",
    default=os.environ.get("NCBI_API_KEY", None),
    help="NCBI API key (overrides environment variable NCBI_API_KEY).",
)
@click.option(
    "--skip-assembly-summary",
    "skip_assembly_summary",
    is_flag=True,
    default=False,
    help="Skip refreshing local assembly_summary.parquet (use existing local copy).",
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
@click.option(
    "--skip-contig-lengths", is_flag=True, help="Skip downloading contig_lengths.parquet."
)
@click.option(
    "--threads",
    "num_threads",
    type=int,
    default=0,
    help="Number of threads for aria2c and pigz (0 = use all cores).",
)
def download_databases(
    force,
    skip_padloc,
    skip_deffinder,
    skip_genomad,
    skip_emapper,
    skip_parquet,
    skip_contig_lengths,
    num_threads,
):
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
    "--output",
    "output_file",
    type=click.Path(),
    default=None,
    help="Optional output file (TSV). If not set, prints to stdout.",
)
def nuc2asmlen(input_file, output_file):
    """Fetch assembly and length metadata for nuccore/contig accessions (with fallback)."""
    from hoodini.pipeline.helpers.nuc2asmlen import run_nuc2asmlen

    df = run_nuc2asmlen(input_file)

    if output_file:
        df.write_csv(output_file, separator="\t")
        console.print(f"[green]Saved results to {output_file}[/green]")
    else:
        sys.stdout.write(df.write_csv(separator="\t", include_header=False))


@utils.command("prefetch_links")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "output_file",
    type=click.Path(),
    default=None,
    help="Optional output file (TSV). If not set, prints to stdout.",
)
@click.option(
    "--kinds",
    "kinds",
    default="gbff,gff,fna,faa,sequence_report",
    help="Comma-separated file kinds to generate links for.",
)
def prefetch_links(input_file, output_file, kinds):
    """Generate prefetched links (assembly_id, file_type, link) for a list of assemblies."""
    from hoodini.pipeline.helpers.prefetch_links import get_prefetched_link_table

    with open(input_file, "r") as fh:
        accs = [l.strip() for l in fh if l.strip()]

    kinds_list = [k.strip() for k in kinds.split(",") if k.strip()]
    # pass through api_fallback flag
    df = get_prefetched_link_table(accs, kinds=kinds_list)
    if output_file:
        df.write_csv(
            output_file,
            separator="\t",
            include_header=False,
            columns=["assembly_id", "file_type", "link"],
        )
        console.print(f"[green]Saved prefetched links to {output_file}[/green]")
    else:
        sys.stdout.write(
            df.write_csv(
                separator="\t", include_header=False, columns=["assembly_id", "file_type", "link"]
            )
        )


if __name__ == "__main__":
    cli()

main = cli
