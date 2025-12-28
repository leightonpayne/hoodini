import collections
import polars as pl
import pyhmmer
from pyhmmer import easel
from pyhmmer.plan7 import HMMFile
from hoodini.utils.core import console
import os
from importlib.resources import files


def deduplicate_domains(df: pl.DataFrame, gap_threshold=10, max_overlap=5, per_database=False) -> pl.DataFrame:
    """Deduplicate domain hits in Polars (no pandas)."""
    if df is None or df.is_empty():
        return pl.DataFrame()

    if "bit_score*alignment_length" not in df.columns and {"bit_score", "alignment_length"}.issubset(df.columns):
        df = df.with_columns((pl.col("bit_score") * pl.col("alignment_length")).alias("bit_score*alignment_length"))

    group_cols = ["protein_id", "database"] if not per_database else ["protein_id"]
    result_rows = []

    for group in df.partition_by(group_cols, as_dict=False, maintain_order=True):
        g = group.sort(["domain_id_clean", "start"])
        merged_hits = []
        prev = None
        for row in g.iter_rows(named=True):
            if (
                prev is not None
                and str(row.get("domain_id_clean")) == str(prev.get("domain_id_clean"))
                and (int(row.get("start", 0)) - int(prev.get("end", 0))) <= gap_threshold
            ):
                prev_start = int(prev.get("start", 0))
                prev_end = int(prev.get("end", 0))
                row_start = int(row.get("start", 0))
                row_end = int(row.get("end", 0))
                prev["start"] = min(prev_start, row_start)
                prev["end"] = max(prev_end, row_end)

                prev_al = float(prev.get("alignment_length", 0) or 0)
                row_al = float(row.get("alignment_length", 0) or 0)
                prev_bs_al = float(prev.get("bit_score*alignment_length", 0) or 0)
                row_bs_al = float(row.get("bit_score*alignment_length", 0) or 0)

                new_al = prev_al + row_al
                new_bs_al = prev_bs_al + row_bs_al

                prev["alignment_length"] = new_al
                prev["bit_score*alignment_length"] = new_bs_al
                prev["bit_score"] = (new_bs_al / new_al) if new_al > 0 else float(prev.get("bit_score", 0) or 0)

                try:
                    prev_e = float(prev.get("e_value", float("inf")) or float("inf"))
                    row_e = float(row.get("e_value", float("inf")) or float("inf"))
                    prev["e_value"] = min(prev_e, row_e)
                except Exception:
                    prev["e_value"] = prev.get("e_value", row.get("e_value", prev.get("e_value")))
            else:
                if prev is not None:
                    merged_hits.append(prev)
                prev = dict(row)
        if prev is not None:
            merged_hits.append(prev)

        merged_hits = sorted(
            merged_hits,
            key=lambda x: float(x.get("bit_score*alignment_length", x.get("bit_score", 0)) or 0),
            reverse=True,
        )

        non_overlapping = []
        occupied = []
        for row in merged_hits:
            start = int(row.get("start", 0))
            end = int(row.get("end", 0))
            overlap = False
            for occ_start, occ_end in occupied:
                ov = min(end, occ_end) - max(start, occ_start) + 1
                if ov > max_overlap:
                    overlap = True
                    break
            if not overlap:
                non_overlapping.append(row)
                occupied.append((start, end))

        result_rows.extend(non_overlapping)

    return pl.DataFrame(result_rows) if result_rows else pl.DataFrame()


def run_domain(all_prots, output, valid_dbs, num_threads,
               *,
               deduplicate: bool = True,
               per_database: bool = False,
               gap_threshold: int = 10,
               max_overlap: int = 5):
    """
    Run domain annotation with HMMER using pre-validated MetaCerberus databases.
    """
    domains_data = pl.DataFrame()
    if not valid_dbs:
        return domains_data

    data_dir = files('hoodini').joinpath('data', 'metacerberus')
    from hoodini.download.metacerberus import list_db_files, get_db_groups, check_downloaded
    try:
        files_list = list_db_files()
        groups = get_db_groups(files_list)
        status = check_downloaded(groups)
    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not load MetaCerberus database info: {e}[/bold yellow]")
        return domains_data

    db_files = []
    for db in valid_dbs:
        file_statuses = status.get(db, [])
        hmm_file = None
        tsv_file = None
        for f, present in file_statuses:
            if present:
                if f["name"].endswith(".hmm.gz"):
                    hmm_file = data_dir / f["name"]
                elif f["name"].endswith(".tsv"):
                    tsv_file = data_dir / f["name"]
        if tsv_file:
            db_files.append((db, hmm_file, tsv_file))
    if not db_files:
        console.print("[bold yellow]No valid database files found.[/bold yellow]")
        return domains_data

    fasta_path = os.path.join(output, "results.fasta")
    if not os.path.exists(fasta_path) or os.path.getsize(fasta_path) == 0:
        console.print(f"[dim]Generating protein sequences file at {fasta_path}...[/dim]")
        try:
            all_prots[["protein_id", "sequence"]].drop_nulls().drop_duplicates("protein_id").to_fasta("protein_id", "sequence", fasta_path)
            console.print(f"[green]✔ Generated {fasta_path}[/green]")
        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not generate FASTA file: {e}[/bold yellow]")
            return domains_data

    alphabet = pyhmmer.easel.Alphabet.amino()
    try:
        with easel.SequenceFile(fasta_path, digital=True, alphabet=alphabet) as seq_file:
            sequences = list(seq_file)
    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not load sequences from {fasta_path}: {e}[/bold yellow]")
        return domains_data

    Result = collections.namedtuple(
        "Result", ["protein_id", "domain_id", "bit_score",
                   "alignment_length", "e_value", "start", "end", "cov", "database"])
    all_results = []

    # --- FIXED HMMER LOOP ---
    for db, hmm_path, tsv_path in db_files:
        console.print(f"🔍\tAnnotating domains for [bold]{db}[/bold] with HMMER...")
        if hmm_path and os.path.exists(hmm_path):
            try:
                with HMMFile(str(hmm_path)) as hmm_file:
                    hmms = list(hmm_file)
            except Exception as e:
                console.print(f"[bold yellow]Warning: Could not load HMM file for {db}: {e}. Skipping.[/bold yellow]")
                continue

            for hits in pyhmmer.hmmsearch(hmms, sequences, cpus=num_threads, E=1e-5):
                hmm_id = hits.query.name.decode()
                for hit in hits.included:  # only included hits
                    protein_id = hit.name.decode()
                    for domain in hit.domains.included:  # only included domains
                        start = int(domain.env_from)
                        end = int(domain.env_to)
                        length = end - start + 1
                        target_len = getattr(domain.alignment, "target_length", hit.length)
                        cov = length / float(target_len)
                        all_results.append(
                            Result(protein_id, hmm_id, domain.score, length,
                                   domain.c_evalue, start, end, cov, db)
                        )
        else:
            console.print(f"[dim]Database {db} has no HMM file, skipping HMMER search.[/dim]")

    if not all_results:
        console.print("[bold yellow]No domain matches found.[/bold yellow]")
        return domains_data

    domains_df = pl.DataFrame(all_results)
    # compute composite score and sort (protein asc, score desc)
    domains_df = domains_df.with_columns(
        (pl.col('bit_score') * pl.col('alignment_length')).alias('bit_score*alignment_length')
    ).sort(['protein_id', 'bit_score*alignment_length'], descending=[False, True])

    # --- Merge metadata ---
    all_domains_with_metadata = []
    for db, hmm_path, tsv_path in db_files:
        db_domains = domains_df.filter(pl.col('database') == db)
        if db_domains.height == 0:
            continue
        try:
            domain_metadata = pl.read_csv(str(tsv_path), separator="\t")
            db_domains = db_domains.with_columns(
                pl.col("domain_id")
                .map_elements(lambda v: v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v, return_dtype=pl.Utf8)
                .alias("domain_id_base")
            ).with_columns(
                pl.col("domain_id_base").cast(pl.Utf8).str.split(".").list.first().alias("domain_id_clean")
            )
            id_column = None
            for col in ["ID", "id", "domain_id", "Domain_ID"]:
                if col in domain_metadata.columns:
                    id_column = col
                    break
            if id_column:
                metadata_columns = {col: f"{col}_{db}" for col in domain_metadata.columns if col != id_column}
                domain_metadata = domain_metadata.rename(metadata_columns)
                merged = db_domains.join(domain_metadata, left_on="domain_id_clean", right_on=id_column, how="left")
                all_domains_with_metadata.append(merged)
            else:
                console.print(f"[bold yellow]Warning: Could not find ID column in metadata for {db}. Using domains without metadata.[/bold yellow]")
                all_domains_with_metadata.append(db_domains)
        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not load metadata for {db}: {e}. Using domains without metadata.[/bold yellow]")
            all_domains_with_metadata.append(db_domains)

    if all_domains_with_metadata:
        domains_data = pl.concat(all_domains_with_metadata, how="vertical")
        if deduplicate and domains_data.height > 0:
            before = domains_data.height
            if "domain_id_clean" not in domains_data.columns:
                domains_data = domains_data.with_columns(
                    pl.col("domain_id")
                    .map_elements(lambda v: v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v, return_dtype=pl.Utf8)
                    .alias("domain_id_base")
                ).with_columns(
                    pl.col("domain_id_base").cast(pl.Utf8).str.split(".").list.first().alias("domain_id_clean")
                )
            if "bit_score*alignment_length" not in domains_data.columns and {"bit_score","alignment_length"}.issubset(domains_data.columns):
                domains_data = domains_data.with_columns(
                    (pl.col("bit_score") * pl.col("alignment_length")).alias("bit_score*alignment_length")
                )

            domains_data = deduplicate_domains(
                domains_data,
                gap_threshold=gap_threshold,
                max_overlap=max_overlap,
                per_database=per_database,
            )
            console.print(
                f"✔️\tDomain annotation complete: {len(domains_data)} matches from {len(db_files)} databases (deduplicated from {before}).\n"
            )
        else:
            console.print(
                f"✔️\tDomain annotation complete: {len(domains_data)} matches from {len(db_files)} databases\n"
            )

    return domains_data
