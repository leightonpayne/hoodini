# hoodini/cli.py

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import rich_click as click
import tomli
from rich.logging import RichHandler
from rich.console import Console

from hoodini.config.schema import Config
from hoodini.config import load_default_config

# Custom Option class for mutually exclusive flags
from hoodini.utils.cli_helpers import MutuallyExclusiveOption

# File‐validation callback
from hoodini.utils.core import validate_input_file

# Shared console/spinner helpers
from hoodini.utils.logging_utils import (
    console,
    stage_header,
    stage_done,
    run_with_spinner
)

# The new function that replaces the Initialize class
from hoodini.initialize import initialize_inputs

# The remaining workflow stages
from hoodini.parse_ipg import IPGParser
from hoodini.parse_assemblies import AssemblyParser
from hoodini.cluster_proteins import ProteinClusterer
from hoodini.taxonomy import TaxonomyParser
from hoodini.annotate_domains import AnnotDomains
from hoodini.arrange_data import Arranger
from hoodini.extra_tools import ExtraAnnotation
from hoodini.extra_plot import ExtraPlotter
from hoodini.clean_data import Cleaner

# -----------------------------------------------------------------------------
# 1) Load flattened defaults from hoodini/config/defaults.toml
# -----------------------------------------------------------------------------
TOML_DEFAULTS = load_default_config()

# -----------------------------------------------------------------------------
# 2) Configure Rich‐Click styling
# -----------------------------------------------------------------------------
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_ERRORS_SUGGESTIONS = "yellow"
click.rich_click.STYLE_HELP_OPTIONS = "bold cyan"
click.rich_click.STYLE_HELP_OPTIONS_DEFAULTS = "dim"


@click.command()
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
    default=TOML_DEFAULTS.get("input_path", None),
    type=click.Path(exists=True),
    callback=validate_input_file,
    help="Path to a single‐column input file (mutually exclusive with --inputsheet)."
)
@click.option(
    "--inputsheet",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["input_path"],
    default=TOML_DEFAULTS.get("inputsheet", None),
    type=click.Path(exists=True),
    callback=validate_input_file,
    help="Path to a TSV input file (mutually exclusive with --input)."
)
@click.option(
    "--output",
    default=TOML_DEFAULTS.get("output", None),
    help="Output folder name."
)
@click.option(
    "--max-concurrent-downloads",
    "max_concurrent_downloads",
    default=TOML_DEFAULTS.get("max_concurrent_downloads", 8),
    type=int,
    help="Maximum concurrent downloads."
)
@click.option(
    "--api-key", "apikey",
    default=TOML_DEFAULTS.get("apikey", ""),
    help="NCBI API key."
)
@click.option(
    "--num-threads", "num_threads",
    default=TOML_DEFAULTS.get("num_threads", 10),
    type=int,
    help="Number of threads."
)
@click.option(
    "--assembly-db",
    "assembly_db",
    default=TOML_DEFAULTS.get(
        "assembly_db",
        os.environ.get("CONDA_PREFIX", "") + "/db/hoodini"
    ),
    help="Assembly DB path."
)
@click.option(
    "--img-db",
    "img_db",
    default=TOML_DEFAULTS.get("img_db", None),
    help="IMG database file (proteins)."
)
@click.option(
    "--img-nuc",
    "img_nuc",
    default=TOML_DEFAULTS.get("img_nuc", None),
    help="IMG database file (nucleotides)."
)
@click.option(
    "--blast",
    default=TOML_DEFAULTS.get("blast", None),
    help="BLAST query file to use."
)
@click.option(
    "--cand-mode",
    "cand_mode",
    default=TOML_DEFAULTS.get("cand_mode", "best_id"),
    help="Mode for selecting IPG representative."
)
@click.option(
    "--clust-method",
    "clust_method",
    default=TOML_DEFAULTS.get("clust_method", "diamond_deepclust"),
    help="Protein clustering method."
)
@click.option(
    "--win-mode",
    "mod",
    default=TOML_DEFAULTS.get("mod", "win_nts"),
    help="Window mode: 'win_nts' or 'win_genes'."
)
@click.option(
    "--win",
    "wn",
    default=TOML_DEFAULTS.get("wn", 20_000),
    type=int,
    help="Window size (genes or nucleotides)."
)
@click.option(
    "--height-factor",
    "height_factor",
    default=TOML_DEFAULTS.get("height_factor", 20),
    type=int,
    help="Height factor for plotting."
)
@click.option(
    "--ngenes",
    default=TOML_DEFAULTS.get("ngenes", 10),
    type=int,
    help="Number of genes in context window."
)
@click.option(
    "--min-win",
    "minwin",
    default=TOML_DEFAULTS.get("minwin", 2000),
    type=int,
    help="Min window size on each side."
)
@click.option(
    "--min-win-type",
    "minwin_type",
    default=TOML_DEFAULTS.get("minwin_type", "both"),
    type=click.Choice(["total", "upstream", "downstream", "both"]),
    help="Type of min window: total | upstream | downstream | both."
)
@click.option(
    "--tree-mode",
    "tree_mode",
    default=TOML_DEFAULTS.get("tree_mode", "make_tree"),
    help="Tree building method."
)
@click.option(
    "--tree-file",
    "tree_file",
    default=TOML_DEFAULTS.get("tree_file", "target_prots.nwk"),
    help="Path to the tree file."
)
@click.option(
    "--padloc",
    is_flag=True,
    default=TOML_DEFAULTS.get("padloc", False),
    help="Run PADLOC for antiphage defense."
)
@click.option(
    "--deffinder",
    is_flag=True,
    default=TOML_DEFAULTS.get("deffinder", False),
    help="Run DefenseFinder for antiphage defense."
)
@click.option(
    "--ncrna",
    is_flag=True,
    default=TOML_DEFAULTS.get("ncrna", False),
    help="Run Infernal for ncRNA prediction."
)
@click.option(
    "--cctyper",
    is_flag=True,
    default=TOML_DEFAULTS.get("cctyper", False),
    help="Run CCtyper for CRISPR‐Cas prediction."
)
@click.option(
    "--genomad",
    is_flag=True,
    default=TOML_DEFAULTS.get("genomad", False),
    help="Run GenoMAD for MGE identification."
)
@click.option(
    "--antidefense",
    is_flag=True,
    default=TOML_DEFAULTS.get("antidefense", False),
    help="Identify anti-defense or ACR genes."
)
@click.option(
    "--phrogs",
    is_flag=True,
    default=TOML_DEFAULTS.get("phrogs", False),
    help="Annotate with PHROGs."
)
@click.option(
    "--sorfs",
    is_flag=True,
    default=TOML_DEFAULTS.get("sorfs", False),
    help="Reannotate small open reading frames."
)
@click.option(
    "--domains",
    default=TOML_DEFAULTS.get("domains", None),
    help="MMseqs2 domain database path.",
    required="--domains-metadata" in os.sys.argv
)
@click.option(
    "--domains-metadata",
    "domains_metadata",
    default=TOML_DEFAULTS.get("domains_metadata", None),
    help="Metadata for MMseqs2 domain database.",
    required="--domains" in os.sys.argv
)
@click.option(
    "--min-prevalence",
    "min_prevalence",
    default=TOML_DEFAULTS.get("min_prevalence", 0.0),
    type=float,
    help="Min prevalence threshold for gene coloring."
)
@click.option(
    "--img",
    default=TOML_DEFAULTS.get("img", "/mnt/fastdb/imgpr_imgvr/data_structure/"),
    help="Path to IMG database files (proteins)."
)
@click.option(
    "--img-metadata",
    "img_metadata",
    default=TOML_DEFAULTS.get(
        "img_metadata",
        "/mnt/fastdb/imgpr_imgvr/merged_metadata/imgvr_pr_taxids.txt"
    ),
    help="Path to IMG database metadata."
)
@click.option(
    "--keep",
    is_flag=True,
    default=TOML_DEFAULTS.get("keep", False),
    help="Keep temporary files (do not delete)."
)
@click.option(
    "--force",
    is_flag=True,
    default=TOML_DEFAULTS.get("force", False),
    help="Overwrite existing output folder if it exists."
)
@click.pass_context
def main(ctx, config_file: Optional[str], **cli_kwargs) -> None:
    """
    🦉 hoodini: gene-centric comparative genomic analysis using publicly available data
    """
    # -------------------------------------------------------------------------
    # 3) Setup Python logging to use RichHandler
    # -------------------------------------------------------------------------
    logging.basicConfig(level="NOTSET", format="%(message)s", handlers=[RichHandler()])

    # -------------------------------------------------------------------------
    # 4) Load external TOML if provided (overrides code defaults)
    # -------------------------------------------------------------------------
    file_kwargs = {}
    if config_file is None and Path("config.toml").exists():
        config_file = "config.toml"
    if config_file:
        with open(config_file, "rb") as f:
            grouped = tomli.load(f)
            file_kwargs = {k: v for section in grouped.values() for k, v in section.items()}

    # -------------------------------------------------------------------------
    # 5) Merge all sources: TOML_DEFAULTS < file_kwargs < cli_kwargs
    # -------------------------------------------------------------------------
    merged = {**TOML_DEFAULTS, **file_kwargs, **cli_kwargs}

    # -------------------------------------------------------------------------
    # 6) Check “neither” case (Click blocked “both” via MutuallyExclusiveOption)
    # -------------------------------------------------------------------------
    if not merged.get("input_path") and not merged.get("inputsheet"):
        raise click.UsageError("One of '--input' or '--inputsheet' must be provided.")

    # -------------------------------------------------------------------------
    # 7) Cast input_path / inputsheet to Path if present
    # -------------------------------------------------------------------------
    if merged.get("input_path"):
        merged["input_path"] = Path(merged["input_path"])
    if merged.get("inputsheet"):
        merged["inputsheet"] = Path(merged["inputsheet"])

    # -------------------------------------------------------------------------
    # 8) Instantiate Config dataclass and store in Click context
    # -------------------------------------------------------------------------
    config = Config(**merged)
    ctx.ensure_object(dict)
    ctx.obj["config"] = config

    console.print()
    stage_header("Initializing Hoodini", "🚀")

    # -------------------------------------------------------------------------
    # 9) Initialize & read input data via the standalone function
    #    (handles folder creation/prompt + returns DataFrame)
    # -------------------------------------------------------------------------
    init_results = run_with_spinner(
        "Initializing & reading input data",
        lambda: initialize_inputs(
            input_path=config.input_path,
            inputsheet=config.inputsheet,
            output=config.output,
            force=config.force,
            assembly_db=config.assembly_db
        )
    )
    stage_done("Initialization complete")

    console.print()
    stage_header("Fetching IPG data", "📥")

    # -------------------------------------------------------------------------
    # 10) IPGParser stage
    # -------------------------------------------------------------------------
    ipg_parser = IPGParser(
        records=init_results,
        api_key=config.apikey,
        num_threads=config.num_threads,
        output_path=Path(config.output)
    )
    ipg_df = run_with_spinner("Querying NCBI IPG", ipg_parser.run)
    stage_done("IPG fetch complete")

    console.print()
    stage_header("Downloading assemblies", "📥")

    # -------------------------------------------------------------------------
    # 11) AssemblyParser stage
    # -------------------------------------------------------------------------
    assembly_parser = AssemblyParser(
        ipg_df=ipg_df,
        assembly_db=config.assembly_db,
        output_path=Path(config.output),
        padloc=config.padloc,
        deffinder=config.deffinder
    )
    assembly_paths = run_with_spinner("Downloading from NCBI", assembly_parser.run)
    stage_done("Assemblies downloaded")

    console.print()
    stage_header("Clustering neighbor proteins", "✨")

    # -------------------------------------------------------------------------
    # 12) ProteinClusterer stage
    # -------------------------------------------------------------------------
    protein_clusterer = ProteinClusterer(
        assemblies=assembly_paths,
        clust_method=config.clust_method,
        num_threads=config.num_threads,
        output_path=Path(config.output)
    )
    cluster_results = run_with_spinner("Running clustering", protein_clusterer.run)
    stage_done("Clustering complete")

    console.print()
    stage_header("Extracting taxonomic information", "🦠")

    # -------------------------------------------------------------------------
    # 13) TaxonomyParser stage
    # -------------------------------------------------------------------------
    taxonomy_parser = TaxonomyParser(
        clusters=cluster_results,
        tree_mode=config.tree_mode,
        tree_file=config.tree_file,
        output_path=Path(config.output)
    )
    taxonomy_df = run_with_spinner("Querying taxonomy", taxonomy_parser.run)
    stage_done("Taxonomy parsing complete")

    console.print()
    stage_header("Annotating domains", "🦠")

    # -------------------------------------------------------------------------
    # 14) AnnotDomains stage
    # -------------------------------------------------------------------------
    annot_domains = AnnotDomains(
        taxonomy_df=taxonomy_df,
        domains_db=config.domains,
        domains_metadata=config.domains_metadata,
        output_path=Path(config.output)
    )
    annotated_df = run_with_spinner("Running domain annotation", annot_domains.run)
    stage_done("Domain annotation complete")

    console.print()
    stage_header("Arranging data", "🧹")

    # -------------------------------------------------------------------------
    # 15) Arranger stage
    # -------------------------------------------------------------------------
    arranger = Arranger(
        annotated_df=annotated_df,
        mod=config.mod,
        wn=config.wn,
        height_factor=config.height_factor,
        output_path=Path(config.output)
    )
    arranged_df = run_with_spinner("Arranging for plotting", arranger.run)
    stage_done("Data arrangement complete")

    console.print()
    stage_header("Extra annotation & plotting", "🖼️")

    # -------------------------------------------------------------------------
    # 16) ExtraAnnotation & ExtraPlotter stages
    # -------------------------------------------------------------------------
    extra_annot = ExtraAnnotation(
        arranged_df=arranged_df,
        output_path=Path(config.output)
    )
    run_with_spinner("Running extra annotation", extra_annot.run)

    extra_plotter = ExtraPlotter(
        arranged_df=arranged_df,
        output_path=Path(config.output)
    )
    run_with_spinner("Rendering extra plots", extra_plotter.run)
    stage_done("Extra annotation & plotting complete")

    console.print()
    stage_header("Cleaning up", "♻️")

    # -------------------------------------------------------------------------
    # 17) Cleaner stage
    # -------------------------------------------------------------------------
    cleaner = Cleaner(
        arranged_df=arranged_df,
        output_path=Path(config.output),
        keep=config.keep
    )
    run_with_spinner("Removing temporary files", cleaner.run)
    stage_done("Cleanup complete")


if __name__ == "__main__":
    main()
