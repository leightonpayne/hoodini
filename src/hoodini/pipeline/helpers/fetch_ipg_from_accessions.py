import argparse
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import polars as pl
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


# Constants
TOOL = "ipg_fetcher"
# Number of accessions per efetch chunk
CHUNK_SIZE = 100

DEFAULT_MAX_CONCURRENT = 9  # when using API key
DEFAULT_FALLBACK_CONCURRENT = 3  # without API key
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
MAX_WORKERS = DEFAULT_MAX_CONCURRENT if NCBI_API_KEY else DEFAULT_FALLBACK_CONCURRENT
MAX_PARALLEL = MAX_WORKERS


def _efetch_chunk(accessions: list[str]) -> str:
    joined_ids = ",".join(accessions)
    cmd = [
        "efetch",
        "-db",
        "protein",
        "-id",
        joined_ids,
        "-format",
        "ipg",
        "-mode",
        "text",
        "-tool",
        TOOL,
    ]

    # Retry logic for transient NCBI errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
            if not result.stdout or not result.stdout.strip():
                if "500" in result.stderr or "ERROR" in result.stderr:
                    if attempt < max_retries - 1:
                        print(
                            f"[WARN] efetch error 500/network issue (attempt {attempt+1}/{max_retries}), retrying in 5s..."
                        )
                        time.sleep(5)
                        continue
                print(
                    f"[WARN] efetch returned empty for IDs: {accessions[:3]}... (stderr: {result.stderr[:200]})"
                )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                print(
                    f"[WARN] efetch failed (attempt {attempt+1}/{max_retries}): {e.stderr[:200]}, retrying..."
                )
                time.sleep(5)
                continue
            print(
                f"[ERROR] efetch failed for IDs {accessions[:3]}... after {max_retries} attempts: {e.stderr[:500]}"
            )
            return ""
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                print(f"[WARN] efetch timeout (attempt {attempt+1}/{max_retries}), retrying...")
                time.sleep(5)
                continue
            print(
                f"[ERROR] efetch timeout for IDs {accessions[:3]}... after {max_retries} attempts"
            )
            return ""

    return ""


def fetch_ipg_from_accessions(accessions: list[str]) -> pl.DataFrame:
    """
    Fetch IPG data for a list of protein accessions in parallel with rate limiting.
    """
    # Check efetch availability
    try:
        subprocess.run(["efetch", "-version"], check=True, capture_output=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[ERROR] efetch not available or not working: {e}")
        return pl.DataFrame()

    # Quiet: avoid noisy debug prints during normal runs
    semaphore = threading.Semaphore(MAX_PARALLEL)
    chunks = [accessions[i : i + CHUNK_SIZE] for i in range(0, len(accessions), CHUNK_SIZE)]
    results = [None] * len(chunks)

    def wrapped_efetch(chunk, idx):
        with semaphore:
            return idx, _efetch_chunk(chunk)

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
            future_to_idx = {
                executor.submit(wrapped_efetch, chunk, idx): idx for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx, chunk_result = future.result()
                # Parse immediately to avoid holding massive strings in memory
                if chunk_result and chunk_result.strip():
                    try:
                        chunk_df = pl.read_csv(
                            StringIO(chunk_result),
                            separator="\t",
                            infer_schema_length=1000,
                        )
                        chunk_df = chunk_df.rename({col: col.lower() for col in chunk_df.columns})
                        results[idx] = chunk_df
                    except Exception as e:
                        snippet = chunk_result[:200].replace("\n", "\\n")
                        print(f"[WARN] Failed to parse IPG chunk {idx}: {e}. Snippet: {snippet}")
                        results[idx] = None
                else:
                    print(f"[WARN] Empty IPG chunk {idx} returned by efetch")
                    results[idx] = None
                progress.update(task, advance=1)

    dfs = [df for df in results if df is not None and df.height > 0]
    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="vertical")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve IPG data using local efetch for protein accessions."
    )
    parser.add_argument("input", help="Input file with protein accessions (one per line)")
    parser.add_argument("-o", "--output", help="Output TSV file to save IPG results", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        accessions = [line.strip() for line in f if line.strip()]

    df = fetch_ipg_from_accessions(accessions)

    if df.height == 0:
        print("[WARN] No IPG data retrieved.")
    else:
        out_path = Path(args.output)
        df.write_csv(out_path, separator="\t")
        print(f"[INFO] Saved {df.height} IPG entries to {out_path}")


if __name__ == "__main__":
    main()
