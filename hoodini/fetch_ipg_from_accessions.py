

import subprocess
import argparse
import os
import pandas as pd
from io import StringIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn


# Constants
TOOL = "ipg_fetcher"
# Number of accessions per efetch chunk
CHUNK_SIZE = 100

DEFAULT_MAX_CONCURRENT = 9        # when using API key
DEFAULT_FALLBACK_CONCURRENT = 3   # without API key
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
MAX_WORKERS = DEFAULT_MAX_CONCURRENT if NCBI_API_KEY else DEFAULT_FALLBACK_CONCURRENT
MAX_PARALLEL = MAX_WORKERS

def run_efetch_chunk(accessions: list[str]) -> str:
    joined_ids = ",".join(accessions)
    cmd = [
        "efetch",
        "-db", "protein",
        "-id", joined_ids,
        "-format", "ipg",
        "-mode", "text",
        "-tool", TOOL
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] efetch failed: {e.stderr}")
        return ""

def fetch_ipg_from_accessions(accessions: list[str]) -> pd.DataFrame:
    """
    Fetch IPG data for a list of protein accessions in parallel with rate limiting.
    """
    semaphore = threading.Semaphore(MAX_PARALLEL)
    chunks = [accessions[i:i + CHUNK_SIZE] for i in range(0, len(accessions), CHUNK_SIZE)]
    results = [None] * len(chunks)

    def wrapped_efetch(chunk, idx):
        with semaphore:
            return idx, run_efetch_chunk(chunk)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        TextColumn("Chunk {task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Fetching IPG chunks", total=len(chunks))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {executor.submit(wrapped_efetch, chunk, idx): idx for idx, chunk in enumerate(chunks)}
            for future in as_completed(future_to_idx):
                idx, chunk_result = future.result()
                # Parse immediately to avoid holding massive strings in memory
                if chunk_result and chunk_result.strip():
                    try:
                        chunk_df = pd.read_csv(StringIO(chunk_result), sep="\t")
                        chunk_df.columns = [col.lower() for col in chunk_df.columns]
                        results[idx] = chunk_df
                    except Exception:
                        results[idx] = None
                else:
                    results[idx] = None
                progress.update(task, advance=1)

    dfs = [df for df in results if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Retrieve IPG data using local efetch for protein accessions.")
    parser.add_argument("input", help="Input file with protein accessions (one per line)")
    parser.add_argument("-o", "--output", help="Output TSV file to save IPG results", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        accessions = [line.strip() for line in f if line.strip()]

    df = fetch_ipg_from_accessions(accessions)

    if df.empty:
        print("[WARN] No IPG data retrieved.")
    else:
        out_path = Path(args.output)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"[INFO] Saved {len(df)} IPG entries to {out_path}")


if __name__ == "__main__":
    main()
