import asyncio
import os, io
import math
from time import perf_counter
from pathlib import Path
from importlib.resources import files
from typing import Optional, List, Tuple, Dict, Any

import aiohttp  # type: ignore[import]
import polars as pl  # type: ignore[import]
import pyarrow.parquet as pq  # type: ignore[import]
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn  # type: ignore[import]
import json
import re
import pyarrow as pa  # for PartRotatingWriter

from hoodini.prefetch_links import get_prefetched_link_table
from hoodini.utils.core import console
from hoodini.download.assembly_summary import download_assembly_summary_db

# Optional API key override
NCBI_API_KEY: Optional[str] = os.environ.get("NCBI_API_KEY")

# File paths
DATA_DIR = files("hoodini").joinpath("data")
CONTIG_LENGTHS_DIR = DATA_DIR.joinpath("contig_lengths")
MASTER_CONTIGS = DATA_DIR.joinpath("contig_lengths.parquet")
ASSEMBLY_SUMMARY = DATA_DIR.joinpath("assembly_summary.parquet")

DEFAULT_GROUPS = {"bacteria", "viral", "archaea", "metagenomes", "other"}
MAX_RETRIES = 3

# Candidate assembly ID columns
_ASM_CANDIDATES: Tuple[str, ...] = (
    "assembly_accession",
    "assemblyAccession",
    "assembly_id",
    "assemblyId",
    "assembly",
)

# ---------------------------------------------------------------------
# Helpers to detect which column to use
# ---------------------------------------------------------------------
def _detect_assembly_col_in_file(path: Path) -> Optional[str]:
    try:
        pf = pq.ParquetFile(path)
        names = set(pf.schema.names)
        for c in _ASM_CANDIDATES:
            if c in names:
                return c
    except Exception:
        pass
    return None

def _detect_assembly_col_in_dir(dirpath: Path, sample: int = 10) -> Optional[str]:
    try:
        files = sorted(dirpath.glob("part-*.parquet"))[:sample]
        for f in files:
            c = _detect_assembly_col_in_file(f)
            if c is not None:
                return c
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------
# Main function to check which assemblies are missing contig lengths
# ---------------------------------------------------------------------
def get_missing_contigs_from_summary(
    assembly_summary_path: Path,
    allowed_assemblies: Optional[set] = None
) -> Tuple[List[str], Optional[float]]:
    summary_df = pl.read_parquet(ASSEMBLY_SUMMARY)
    console.log(f"Number of rows in assembly_summary: {summary_df.height}")

    summary_df = summary_df.filter(
        (pl.col("group").is_in(DEFAULT_GROUPS)) &
        pl.col("ftp_path").is_not_null() &
        (pl.col("ftp_path").str.strip_chars() != "") &
        (pl.col("ftp_path").str.to_lowercase() != "na")
    )

    contig_set: set[str] = set()
    latest_mtime: Optional[float] = None

    # --- parts hive ---
    if CONTIG_LENGTHS_DIR.exists():
        part_files = list(CONTIG_LENGTHS_DIR.glob("part-*.parquet"))
        part_col = _detect_assembly_col_in_dir(CONTIG_LENGTHS_DIR) or "assemblyAccession"
        parts_df = (
            pl.scan_parquet(str(CONTIG_LENGTHS_DIR / "*.parquet"))
            .select([part_col])
            .unique()
            .collect()
        )
        contig_set.update(parts_df[part_col].cast(pl.Utf8).to_list())
        console.log(f"Loaded {len(contig_set)} assemblies from contig_lengths/ (hive)")
        try:
            part_mtime = max(p.stat().st_mtime for p in part_files)
            latest_mtime = max(latest_mtime, part_mtime) if latest_mtime is not None else part_mtime
        except Exception:
            pass

    summary_ids = set(summary_df["assembly_accession"].cast(pl.Utf8).to_list())
    if allowed_assemblies is not None:
        summary_ids = summary_ids.intersection(allowed_assemblies)

    # Debug info
    console.log(f"summary_df rows after filter: {summary_df.height}")
    console.log("First 5 rows of summary_df:")
    console.log(f"Number of assembly_accession in summary_df: {len(summary_df['assembly_accession'])}")
    console.log(f"contig_set size: {len(contig_set)}")
    if contig_set:
        console.log(f"First 10 contigs in contig_set: {list(contig_set)[:10]}")
    else:
        console.log("contig_set is empty!")
    console.log(f"latest_mtime: {latest_mtime}")

    missing = [aid for aid in summary_ids if aid not in contig_set]
    console.log(f"❗ {len(missing)} assemblies missing contig lengths.")
    return missing, latest_mtime

# ---------------------------------------------------------------------
# Async fetchers + rotating writer (unchanged)
# ---------------------------------------------------------------------
SENTINEL = object()

def _consume_lines(buffer: bytearray):
    while True:
        nl = buffer.find(b"\n")
        if nl == -1:
            break
        line = buffer[:nl]
        del buffer[:nl+1]
        yield line.decode("utf-8", errors="ignore").strip()

class PartRotatingWriter:
    def __init__(self, dataset_dir: Path, target_bytes: int = 30*1024*1024, start_rows: int = 80_000):
        self.dir = dataset_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.target_bytes = target_bytes
        self.rows_target = start_rows
        self._buffer: List[Dict[str, Any]] = []
        self.part_idx = self._next_index()
        self.total_rows = 0
        self.total_files = 0
    def _next_index(self) -> int:
        existing = [p for p in self.dir.glob("part-*.parquet")]
        if not existing:
            return 0
        max_id = -1
        for p in existing:
            try:
                max_id = max(max_id, int(p.stem.split("-")[1]))
            except Exception:
                pass
        return max_id + 1
    def _write_once(self, rows: List[Dict[str, Any]]) -> int:
        table = pa.Table.from_pylist(rows)
        tmp = self.dir / f"part-{self.part_idx:05d}.parquet.tmp"
        pq.write_table(table, tmp, compression="zstd")
        return os.path.getsize(tmp)
    def _commit_tmp(self):
        tmp = self.dir / f"part-{self.part_idx:05d}.parquet.tmp"
        final = self.dir / f"part-{self.part_idx:05d}.parquet"
        os.replace(tmp, final)
        self.part_idx += 1
        self.total_files += 1
    def add_many(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        self._buffer.extend(rows)
        self.total_rows += len(rows)
        while len(self._buffer) >= self.rows_target:
            self._flush_targeted()
    def _flush_targeted(self):
        if not self._buffer:
            return
        rows = self._buffer
        target_n = min(self.rows_target, len(rows))
        try_rows = rows[:target_n]
        size = self._write_once(try_rows)
        self._commit_tmp()
        del rows[:target_n]
        self._buffer = rows
    def close(self):
        if self._buffer:
            _ = self._write_once(self._buffer)
            self._commit_tmp()
            self._buffer.clear()

async def fetch_to_queue(session, asm, url, queue, batch_rows, retries, timeout_s, sem):
    attempt = 0
    CHUNK = 1 << 19
    sent = 0
    async with sem:
        while True:
            attempt += 1
            try:
                batch: List[Dict[str, Any]] = []
                async with session.get(url, timeout=timeout_s) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    buf = bytearray()
                    async for chunk in resp.content.iter_chunked(CHUNK):
                        if not chunk:
                            continue
                        buf.extend(chunk)
                        for line in _consume_lines(buf):
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            batch.append(obj)
                            if len(batch) >= batch_rows:
                                await queue.put(batch)
                                sent += len(batch)
                                batch = []
                    if buf:
                        tail = buf.decode("utf-8", errors="ignore").strip()
                        if tail:
                            try:
                                obj = json.loads(tail)
                                batch.append(obj)
                            except Exception:
                                pass
                if batch:
                    await queue.put(batch)
                    sent += len(batch)
                await queue.put(SENTINEL)
                return True, f"rows={sent}", sent
            except Exception as e:
                if attempt <= retries:
                    await asyncio.sleep(min(2**attempt, 10))
                    continue
                await queue.put(SENTINEL)
                return False, str(e), sent

async def writer_consumer(queue, n_producers, writer):
    done = 0
    while True:
        item = await queue.get()
        if item is SENTINEL:
            done += 1
            if done >= n_producers:
                break
            continue
        writer.add_many(item)
    writer.close()
    return writer.total_rows, writer.total_files

async def stream_and_write(pairs, target_mb=30, batch_rows=5000, concurrency=10, retries=3, timeout=60):
    target_bytes = target_mb * 1024 * 1024
    writer = PartRotatingWriter(dataset_dir=CONTIG_LENGTHS_DIR, target_bytes=target_bytes)
    queue = asyncio.Queue(maxsize=20)
    sem = asyncio.Semaphore(concurrency)
    consumer_task = asyncio.create_task(writer_consumer(queue, n_producers=len(pairs), writer=writer))

    timeout_cfg = aiohttp.ClientTimeout(total=None, connect=30, sock_read=600)
    headers = {"Accept-Encoding": "gzip, deflate", "User-Agent": "seqrep-dl/hoodini"}
    connector = aiohttp.TCPConnector(limit_per_host=64, ttl_dns_cache=300)

    async with aiohttp.ClientSession(timeout=timeout_cfg, connector=connector, headers=headers) as session:
        progress = Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.completed}/{task.total} {task.description}"), TimeElapsedColumn())
        task_id = progress.add_task(f"Downloading (conc={concurrency})", total=len(pairs))
        ok = 0
        total_rows = 0
        with progress:
            tasks = [asyncio.create_task(fetch_to_queue(session, asm, url, queue, batch_rows, retries, timeout, sem)) for asm, url in pairs]
            for fut in asyncio.as_completed(tasks):
                ok1, info, sent = await fut
                total_rows += sent
                progress.update(task_id, advance=1, description=info[:120])
                if ok1:
                    ok += 1
    total_rows_written, files_written = await consumer_task
    return ok, total_rows, total_rows_written, files_written

def download_contig_lengths(api_key: Optional[str] = None, workers: int = 10, skip_assembly_summary: bool = False):
    global NCBI_API_KEY
    NCBI_API_KEY = api_key
    if not skip_assembly_summary:
        console.log("🔄 Updating local assembly_summary.parquet...")
        download_assembly_summary_db()
    else:
        console.log("⏭️  Skipping assembly_summary refresh (using local copy)")

    # Determine allowed assemblies
    allowed_assemblies = None
    try:
        asm_df = pl.read_parquet(ASSEMBLY_SUMMARY)
        asm_df = asm_df.filter(
            (pl.col("group").is_in(DEFAULT_GROUPS))
            & pl.col("ftp_path").is_not_null()
            & (pl.col("ftp_path").str.strip_chars() != "")
            & (pl.col("ftp_path").str.to_lowercase() != "na")
        )
        candidate_ids = list(set(asm_df["assembly_accession"].to_list()))
        if candidate_ids:
            links_df = get_prefetched_link_table(candidate_ids, kinds=["sequence_report"])
            allowed_assemblies = set(links_df[links_df["filetype"] == "sequence_report"]["assembly_id"].to_list())
    except Exception:
        allowed_assemblies = None

    missing, latest_mtime = get_missing_contigs_from_summary(ASSEMBLY_SUMMARY, allowed_assemblies=allowed_assemblies)

    if latest_mtime is not None and missing:
        import pandas as pd
        asm_df = pd.read_parquet(ASSEMBLY_SUMMARY)
        if "seq_rel_date" in asm_df.columns:
            asm_df["seq_rel_date"] = pd.to_datetime(asm_df["seq_rel_date"], errors="coerce", infer_datetime_format=True)
            asm_dates = dict(zip(asm_df["assembly_accession"].astype(str), asm_df["seq_rel_date"]))
            latest_ts = pd.to_datetime(latest_mtime, unit="s")
            filtered = []
            for aid in missing:
                dt = asm_dates.get(str(aid))
                if pd.isna(dt) or dt.to_datetime64() > latest_ts.to_datetime64():
                    filtered.append(aid)
            console.log(f"Filtered missing assemblies by seq_rel_date vs latest parquet mtime: {len(filtered)} remain")
            missing = filtered
        else:
            console.log("❌ 'seq_rel_date' not found; skipping date-based filtering")

    if not missing:
        console.log("✅ No missing contig lengths to download.")
        return

    df_links = get_prefetched_link_table(missing, kinds=["sequence_report"], seqrep_only=True)
    pairs: List[Tuple[str, str]] = []
    for _, row in df_links.iterrows():
        if row.get("filetype") == "sequence_report":
            pairs.append((row.get("assembly_id"), row.get("url")))

    if not pairs:
        console.log("✅ No sequence_report links available for missing assemblies.")
        return

    ok, rows_fetched, rows_written, files_written = asyncio.run(
        stream_and_write(pairs, target_mb=30, batch_rows=5000, concurrency=workers, retries=MAX_RETRIES, timeout=60)
    )

    if rows_written == 0:
        console.log("✅ No contig length records returned.")
        return

    console.log(f"✅ Downloaded contig lengths for {len(pairs)} sequence_report links (ok={ok}/{len(pairs)}).")
    console.log(f"    rows fetched={rows_fetched}; rows written={rows_written}; new parts={files_written}")
