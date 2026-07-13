import contextlib
import re
import tempfile
from pathlib import Path

import polars as pl

from hoodini.utils.downloader import download_with_aria2c
from hoodini.utils.logging_utils import logger

ACCESSION_RE = re.compile(r"^GC[AF]_\d+\.\d+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PUBMED_RE = re.compile(r"^\d+(?:;\d+)*$")
MERGE_PRIORITY = {
    "asm_submitter": 100,
    "organism_name": 95,
    "annotation_provider": 95,
    "annotation_name": 90,
    "relation_to_type_material": 80,
    "infraspecific_name": 75,
    "isolate": 70,
    "asm_name": 60,
}


def _merge_parts(parts: list[str], start: int, width: int) -> list[str]:
    merged_value = " ".join(part.strip() for part in parts[start : start + width] if part.strip())
    return parts[:start] + [merged_value] + parts[start + width :]


def _is_int_or_na(value: str) -> bool:
    return value == "na" or value.isdigit()


def _is_float_or_na(value: str) -> bool:
    if value == "na":
        return True
    with contextlib.suppress(ValueError):
        float(value)
        return True
    return False


def _is_date_or_na(value: str) -> bool:
    return value == "na" or bool(DATE_RE.fullmatch(value))


def _is_accession_or_na(value: str) -> bool:
    return value == "na" or bool(ACCESSION_RE.fullmatch(value))


def _is_ftp_path_or_na(value: str) -> bool:
    return value == "na" or value.startswith("https://ftp.ncbi.nlm.nih.gov/")


def _is_pubmed_or_na(value: str) -> bool:
    return value == "na" or bool(PUBMED_RE.fullmatch(value))


def _is_annotation_provider_or_na(value: str) -> bool:
    return value == "na" or "Annotation submitted by" not in value


def _row_alignment_score(parts: list[str], header: list[str]) -> int:
    validators = {
        "assembly_accession": lambda value: bool(ACCESSION_RE.fullmatch(value)),
        "taxid": _is_int_or_na,
        "species_taxid": _is_int_or_na,
        "version_status": lambda value: value in {"latest", "replaced", "suppressed", "na"},
        "assembly_level": lambda value: value
        in {"Complete Genome", "Chromosome", "Scaffold", "Contig", "na"},
        "release_type": lambda value: value in {"Major", "Minor", "Patch", "na"},
        "genome_rep": lambda value: value in {"Full", "Partial", "na"},
        "seq_rel_date": _is_date_or_na,
        "gbrs_paired_asm": _is_accession_or_na,
        "paired_asm_comp": lambda value: value in {"identical", "different", "na"},
        "ftp_path": _is_ftp_path_or_na,
        "asm_not_live_date": _is_date_or_na,
        "genome_size": _is_int_or_na,
        "genome_size_ungapped": _is_int_or_na,
        "gc_percent": _is_float_or_na,
        "replicon_count": _is_int_or_na,
        "scaffold_count": _is_int_or_na,
        "contig_count": _is_int_or_na,
        "annotation_provider": _is_annotation_provider_or_na,
        "annotation_date": _is_date_or_na,
        "total_gene_count": _is_int_or_na,
        "protein_coding_gene_count": _is_int_or_na,
        "non_coding_gene_count": _is_int_or_na,
        "pubmed_id": _is_pubmed_or_na,
    }
    score = 0
    for column_name, validator in validators.items():
        try:
            value = parts[header.index(column_name)]
        except (IndexError, ValueError):
            continue
        if validator(value):
            score += 1
    return score


def _normalize_overwide_row(parts: list[str], header: list[str]) -> list[str]:
    expected_len = len(header)
    extra_columns = len(parts) - expected_len
    if extra_columns <= 0:
        raise ValueError("Row is not wider than the header")
    candidates: list[tuple[int, int, int, list[str]]] = []

    def search(current_parts: list[str], merges_remaining: int, start_idx: int, priority: int) -> None:
        if merges_remaining == 0:
            if len(current_parts) != expected_len:
                return
            score = _row_alignment_score(current_parts, header)
            candidates.append((score, priority, -start_idx, current_parts))
            return

        for merge_start in range(start_idx, len(current_parts) - 1):
            merged = _merge_parts(current_parts, merge_start, 2)
            column_name = header[min(merge_start, expected_len - 1)]
            merge_priority = priority + MERGE_PRIORITY.get(column_name, 0)
            search(merged, merges_remaining - 1, merge_start, merge_priority)

    search(parts, extra_columns, 0, 0)

    if not candidates:
        raise ValueError(
            f"Could not normalize row with {len(parts)} columns to expected width {expected_len}"
        )

    best_score, _, _, best_candidate = max(candidates, key=lambda item: item[:3])
    if best_score < 12:
        raise ValueError(
            f"Could not confidently normalize row with {len(parts)} columns to expected width {expected_len}"
        )
    return best_candidate


def generate_summary_urls(dbs: list[str], include_historical: bool = True) -> list[str]:
    """Generate NCBI assembly summary URLs based on database and historical flag."""
    suffixes = ["", "_historical"] if include_historical else [""]
    return [
        f"https://ftp.ncbi.nlm.nih.gov/genomes/{db}/assembly_summary_{db}{suf}.txt"
        for db in dbs
        for suf in suffixes
    ]


def get_ncbi_header(file_path: Path) -> list[str]:
    """Extract header from an NCBI assembly summary file (line starting with #assembly_accession)."""
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#assembly_accession"):
                return line.lstrip("#").strip().split("\t")
    raise ValueError(f"No header found in {file_path}")


def normalize_assembly_summary_row(parts: list[str], header: list[str]) -> list[str]:
    """Normalize a data row to the expected NCBI assembly summary width.

    Some upstream rows contain literal tab characters inside a text field, which causes the row to
    become wider than the header. Repair those rows by finding the merge point that restores the
    best semantic alignment across constrained columns.
    """
    expected_len = len(header)
    if len(parts) == expected_len:
        return parts

    if len(parts) > expected_len:
        return _normalize_overwide_row(parts, header)

    raise ValueError(
        f"Could not normalize row with {len(parts)} columns to expected width {expected_len}"
    )


def normalize_assembly_summary_file(file_path: Path, header: list[str]) -> tuple[Path, int]:
    """Write a normalized TSV copy of an NCBI assembly summary file."""
    repaired_rows = 0
    with (
        open(file_path, encoding="utf-8") as src,
        tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
            dir=file_path.parent,
            suffix=".normalized.tsv",
        ) as dst,
    ):
        for line in src:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            normalized = normalize_assembly_summary_row(parts, header)
            if len(parts) != len(header):
                repaired_rows += 1
            dst.write("\t".join(normalized))
            dst.write("\n")

    return Path(dst.name), repaired_rows


def download_assembly_db(
    dbs: list[str],
    output_path: Path,
    columns_to_keep: list[str] | None = None,
    include_historical: bool = True,
) -> None:
    """Download multiple assembly summary files using aria2c and save combined table."""
    urls = generate_summary_urls(dbs, include_historical)

    out_names = [Path(url).name for url in urls]
    data_dir = output_path.parent
    logger.info(f"Downloading {len(urls)} assembly summary files with aria2c...")

    result_files = download_with_aria2c(urls, data_dir, show_progress=True, out_names=out_names)

    dfs = []
    failed_files = []
    for file_path in result_files:
        logger.info(f"Parsing {file_path}")
        normalized_path = None
        try:
            header = get_ncbi_header(file_path)
            normalized_path, repaired_rows = normalize_assembly_summary_file(
                Path(file_path), header
            )
            if repaired_rows:
                logger.warning(
                    f"Repaired {repaired_rows} malformed rows in {file_path} before parsing"
                )
            df = pl.read_csv(
                normalized_path,
                separator="\t",
                has_header=False,
                new_columns=header,
                quote_char=None,
                null_values="na",
                schema_overrides={column: pl.Utf8 for column in header},
            )
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            failed_files.append(file_path)
            with contextlib.suppress(Exception):
                Path(file_path).unlink()
            if normalized_path is not None:
                with contextlib.suppress(Exception):
                    normalized_path.unlink()
            continue

        if "assembly_accession" not in df.columns:
            logger.error(f"No 'assembly_accession' in {file_path}!")
            continue

        if columns_to_keep:
            keep = [c for c in columns_to_keep if c in df.columns]
            df = df.select(keep)

        df = df.with_columns(
            pl.col(["taxid", "species_taxid", "genome_size", "replicon_count"]).cast(
                pl.Int64, strict=False
            ),
            pl.col(["gc_percent"]).cast(pl.Float64, strict=False),
            pl.col(["scaffold_count", "contig_count"]).cast(pl.Int64, strict=False),
        )

        dfs.append(df)
        logger.info(f"Done {file_path}")
        with contextlib.suppress(Exception):
            Path(file_path).unlink()
        if normalized_path is not None:
            with contextlib.suppress(Exception):
                normalized_path.unlink()

    if not dfs:
        logger.warning("No dataframes were downloaded.")
        return

    if failed_files:
        failed_list = ", ".join(str(path) for path in failed_files)
        raise RuntimeError(f"Failed to parse one or more assembly summary files: {failed_list}")

    combined = (
        pl.concat(dfs, how="vertical_relaxed").unique(subset=["assembly_accession"]).rechunk()
    )

    combined.write_parquet(output_path)
    logger.info(f"Saved combined parquet to {output_path}")


def download_assembly_summary_db(output_path: Path | None = None) -> Path:
    """Convenience wrapper to download and merge RefSeq + GenBank assembly summaries."""
    from importlib.resources import files

    if output_path is None:
        output_path = files("hoodini").joinpath("data", "assembly_summary.parquet")

    columns = [
        "assembly_accession",
        "refseq_category",
        "taxid",
        "species_taxid",
        "organism_name",
        "infraspecific_name",
        "isolate",
        "assembly_level",
        "genome_rep",
        "gbrs_paired_asm",
        "paired_asm_comp",
        "group",
        "ftp_path",
        "genome_size",
        "gc_percent",
        "replicon_count",
        "scaffold_count",
        "contig_count",
        "seq_rel_date",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    download_assembly_db(
        dbs=["refseq", "genbank"],
        output_path=output_path,
        columns_to_keep=columns,
        include_historical=True,
    )

    return output_path
