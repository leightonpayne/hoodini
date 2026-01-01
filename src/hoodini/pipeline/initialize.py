import sys
import shutil
import datetime
from pathlib import Path
from typing import Optional, Union
from importlib.resources import files

import polars as pl
from rich.prompt import Prompt
from rich.console import Console

from hoodini.models.schemas import RECORDS
from hoodini.utils.polars_adapters import to_polars
from hoodini.utils.validation import read_input_list, read_input_sheet, uniprot2ncbi
from hoodini.download.assembly_summary import download_assembly_db

console = Console()


def initialize_inputs(
    *,
    input_path: Optional[Union[Path, str]] = None,
    inputsheet: Optional[Union[Path, str]] = None,
    output: Optional[Union[Path, str]] = None,
    force: bool = False,
) -> pl.DataFrame:
    """
    Initialize the working directory and read the user’s input records (Polars).

    1. Creates or (if existing) optionally overwrites the output folder.
    2. Reads either a single‐column input list or a TSV “inputsheet”.
    3. Converts UniProt IDs to NCBI IDs via `uniprot2ncbi(...)`.
    4. Drops duplicate records based on “og_index”.
    5. Returns a Polars DataFrame of final, deduplicated records.
    """
    check_assembly_db()

    # Step 1: Determine output folder
    if output:
        output_folder = Path(output)
    elif input_path:
        output_folder = Path(input_path).with_suffix("")
    elif inputsheet:
        output_folder = Path(inputsheet).with_suffix("")
    else:
        console.print(
            "[bold red]Error:[/bold red] Either `input_path` or `inputsheet` must be provided."
        )
        sys.exit(1)

    # Step 2: Create or overwrite folder
    if output_folder.exists():
        if force:
            console.print(f"🗑️  Overwriting existing folder [bold]{output_folder}[/bold].")
            shutil.rmtree(output_folder)
        else:
            console.print(f"⚠️  Folder [bold]{output_folder}[/bold] already exists.")
            answer = _prompt_yes_no("⌨️  Remove it? (y/N)", default="no")
            if not answer:
                console.print(f"[bold red]Aborting:[/bold red] Folder not modified.")
                sys.exit(1)
            shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✔️  Created folder [bold]{output_folder}[/bold].[/green]\n")

    # Step 3: Read input records
    if inputsheet:
        records_raw = read_input_sheet(inputsheet)
    elif input_path:
        records_raw = read_input_list(input_path)
    else:
        console.print("[bold red]Error:[/bold red] No input_path or inputsheet provided.")
        sys.exit(1)

    # Step 4: Convert UniProt → NCBI, drop duplicates (Polars)
    records_pd = uniprot2ncbi(records_raw)
    records = to_polars(records_pd, schema=RECORDS)

    records = records.unique(subset=["og_index"], keep="first")

    # Set premade=True for rows that have (gff_path and faa_path) or (gbf_path)
    records = records.with_columns(
        (
            (pl.col("gff_path").is_not_null() & pl.col("faa_path").is_not_null())
            | (pl.col("gbf_path").is_not_null())
        ).alias("premade")
    )

    return records


def check_assembly_db() -> None:
    """
    Check if assembly_summary.parquet exists and optionally download it.
    If the file is older than 1 month, show a warning and instruct the user to run the update command.
    """
    try:
        summary_path = files("hoodini").joinpath("data", "assembly_summary.parquet")
        now = datetime.datetime.now()
        one_month = datetime.timedelta(days=30)

        if summary_path.exists():
            mtime = datetime.datetime.fromtimestamp(summary_path.stat().st_mtime)
            age = now - mtime
            console.print(
                f"📁 Assembly DB found: [bold]{summary_path}[/bold] (updated: {mtime:%Y-%m-%d})"
            )
            if age > one_month:
                console.print(
                    f"[bold yellow]⚠️  WARNING: The assembly database is older than 30 days.[/bold yellow]"
                )
                console.print(
                    "[yellow]To update the database, run: [bold]hoodini update assembly_summary[/bold][/yellow]\n"
                )
            else:
                console.print("✅ Using existing database (less than 1 month old).\n")
            return
        else:
            console.print("📭 No local NCBI assembly summary database found. Downloading now...\n")
            from hoodini.download.assembly_summary import download_assembly_summary_db

            output_path = download_assembly_summary_db(summary_path)
            console.print(
                f"[green]✔️  Downloaded assembly database to [bold]{output_path}[/bold].[/green]\n"
            )
            return
    except Exception as e:
        console.print(f"[bold red]❌ Error checking or downloading assembly DB: {e}[/bold red]\n")


def _prompt_yes_no(prompt_text: str, default: str = "no") -> bool:
    """Ask user a yes/no question via prompt."""
    valid_yes = {"y", "yes"}
    valid_no = {"n", "no"}
    while True:
        response = Prompt.ask(prompt_text, default=default).strip().lower()
        if response in valid_yes:
            return True
        if response in valid_no:
            return False
        console.print("❗  Please enter 'y' or 'n'.")
