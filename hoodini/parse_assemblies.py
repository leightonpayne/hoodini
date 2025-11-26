# hoodini/parse_assemblies.py

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
import hoodini  # Ensure hoodini is correctly installed and in PYTHONPATH
from multiprocessing import Pool
import requests
import requests.adapters
from pathlib import Path

import pandas as pd
import polars as pl

from hoodini.utils.core import console, extract_neighborhood

from hoodini.prefetch_links import get_prefetched_link_table


def in_jupyter():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        else:
            return False  # Other type (likely terminal)
    except Exception:
        return False

def run_assembly_parser(
    records_df: pd.DataFrame,
    *,
    output_dir: str = None,
    assembly_folder: str = None,
    ncrna: bool = False,
    cctyper: bool = False,
    genomad: bool = False,
    blast: str = None,
    apikey: str = "",
    max_concurrent_downloads: int = 8,
    img: str = None,
    num_threads: int = 10,
    mod: str = "win_nts",
    wn: int = 20000,
    sorfs: bool = False,
    minwin: int = None,
    minwin_type: str = "both"
) -> dict:
    """
    Take an existing `records_df` (pandas.DataFrame) and:
      1) Download assemblies (or use a local assembly folder) for all valid assembly_ids.
      2) Populate `gbf_path`, `gff_path`, `faa_path`, `fna_path` for each record.
      3) Run `extract_neighborhood(...)` in parallel for each valid record.
      4) Concatenate results into `all_gff` and `all_neigh` DataFrames.
      5) Enforce a minimum window size (`minwin`) and mark any short contigs as failed.
      6) Write out intermediate files: 
           - `{output}/assembly_list.txt`
           - `{output}/records.csv`
           - `{output}/neighborhood/neighborhoods.fasta`
           - `{output}/results.fasta`
    Returns a dict with keys:
      - "records":  updated DataFrame
      - "all_gff":   DataFrame of extracted GFF sequences (id, sequence)
      - "all_neigh": DataFrame of neighborhood sequences (seqid, sequence, etc.)
    """
    # Work on a copy to avoid mutating the caller’s DataFrame
    records = records_df.copy()

    # Infer output directory if not provided, using the same logic as initialize_inputs
    if output_dir is None:
        input_path = None
        if hasattr(records_df, 'input_path'):
            input_path = records_df.input_path
        elif 'input_path' in records_df.columns:
            input_path = records_df['input_path'].iloc[0]
        if input_path is not None:
            output_dir = str(Path(input_path).stem)
        else:
            raise ValueError("No output directory provided and could not infer from input_path. Please specify output_dir.")

    os.makedirs(output_dir, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────────────────
    # 1) DOWNLOAD ASSEMBLIES (or use local folder)
    # ───────────────────────────────────────────────────────────────────────────────
    def _download_assemblies(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        # Build list of unique assembly_ids where assembly_id.notnull() and failed.isnull()
        mask = df["assembly_id"].notnull() & df["failed"].isnull()
        assembly_list = df.loc[mask, "assembly_id"].dropna().unique().tolist()

        if not assembly_list:
            console.print("ℹ️  No assemblies to download.")
            return df, None

        # Write assembly_list.txt
        assembly_list_file = os.path.join(output_dir, "assembly_list.txt")
        with open(assembly_list_file, "w") as f:
            f.write("\n".join(assembly_list))

        #generate links results by calling get_prefetched_link_table with the assembly_list
        links_result = get_prefetched_link_table(assembly_list)

        # Keep only genomic GenBank (gbff) rows — other filetypes (faa, fna, gff)
        # may also be returned by get_prefetched_link_table; saving all kinds
        # to a single target filename caused FAA (protein FASTA) to appear
        # at the start of the `genomic.gbff` files. Filter here so we only
        # download the intended GenBank flat file.
        if "filetype" in links_result.columns:
            gbff_links = links_result[links_result["filetype"].str.lower() == "gbff"].copy()
        else:
            gbff_links = links_result

        #check those in which the assembly_id was not found in links_result
        missing_assemblies = set(assembly_list) - set(gbff_links["assembly_id"].unique())
        if missing_assemblies:
            console.print(f"[yellow]Warning:[/yellow] The following assembly_ids were not found in the download links: {missing_assemblies}")
                
        assembly_folder_path = os.path.join(output_dir, "assembly_folder")

        # 1) Create a single Session and mount adapters (tune pool sizes)
        session = requests.Session()
        total_files = len(gbff_links)
        use_jupyter = in_jupyter()
        if use_jupyter:
            from tqdm.notebook import tqdm
            pbar = tqdm(total=total_files, desc="Downloading assemblies")
        else:
            from rich.progress import Progress
            progress = Progress()
            task = progress.add_task("Downloading assemblies", total=total_files)
            progress.start()

        def download_gbff(row):
            assembly_id = row.get("assembly_id")
            url = row.get("url")
            target_folder = os.path.join(assembly_folder_path, assembly_id)
            target_file = os.path.join(target_folder, "genomic.gbff")

            if os.path.exists(target_file):
                return f"[SKIP] {assembly_id} already exists."

            try:
                os.makedirs(target_folder, exist_ok=True)

                # 2) Use session.get(...) + stream=True
                with session.get(url, timeout=60, stream=True) as resp:
                    resp.raise_for_status()
                    # 3) Write in chunks to avoid huge memory spike
                    with open(target_file, "wb") as fout:
                        for chunk in resp.iter_content(chunk_size=1024 * 32):  # 32 KB chunks
                            if not chunk:
                                continue
                            fout.write(chunk)

                return f"[OK] {assembly_id}"

            except Exception as e:
                return f"[FAIL] {assembly_id}: {e}"

        # 4) Limit worker count to the user-supplied concurrency and size the
        #    requests connection pool accordingly to avoid "Connection pool is full"
        MAX_THREADS = max(1, min(max_concurrent_downloads or 8, 32))
        adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_THREADS, pool_maxsize=MAX_THREADS)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Convert DataFrame to a lightweight list of dicts for safe iteration from threads
        link_records = gbff_links.to_dict(orient="records")

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(download_gbff, row) for row in link_records]
            for future in as_completed(futures):
                result = future.result()
                if use_jupyter:
                    pbar.update(1)
                else:
                    progress.advance(task)
        if use_jupyter:
            pbar.close()
        else:
            progress.stop()


        # Now populate gbf_path for each record
        cwd = os.getcwd()
        def _make_gbf_path(aid):
            return os.path.join(cwd, assembly_folder_path, aid, "genomic.gbff")

        df.loc[mask, "gbf_path"] = df.loc[mask, "assembly_id"].apply(_make_gbf_path)
        console.print(f"✔️  Downloaded or located assemblies (folder: {assembly_folder_path})")

        return df, assembly_folder_path

    records, actual_assembly_folder = _download_assemblies(records)

    # ───────────────────────────────────────────────────────────────────────────────
    # 2) EXTRACT NEIGHBORHOODS (parallel) → all_gff, all_neigh
    # ───────────────────────────────────────────────────────────────────────────────
    def _extract_neighborhoods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Add a unique_id column so that each parallel result can mark failures
        df = df.copy()
        df["unique_id"] = df.index

        # Ensure start, end, strand exist
        for col in ("start", "end", "strand"):
            if col not in df.columns:
                df[col] = None

        # Build list of arguments for extract_neighborhood
        file_list = []
        
        for idx, row in df.iterrows():
            has_gbf = bool(row.get("gbf_path"))
            has_gff_and_faa = bool(row.get("gff_path") and row.get("faa_path"))
            no_fail = pd.isnull(row.get("failed"))

            if (has_gbf or has_gff_and_faa) and no_fail:
                file_list.append((
                    row.get("protein_id"),
                    row.get("nucleotide_id"),
                    row.get("gbf_path"),
                    row.get("gff_path"),
                    row.get("faa_path"),
                    row.get("fna_path"),
                    mod,
                    wn,
                    row.get("strand"),
                    row.get("start"),
                    row.get("end"),
                    row["unique_id"],
                    row.get("input_type"),
                    sorfs
                ))

        if not file_list:
            console.print("ℹ️  No valid records to extract neighborhoods.")
            return df, pd.DataFrame(), pd.DataFrame()

        # Run in parallel
        with Pool(processes=num_threads) as pool:
            results = pool.starmap(extract_neighborhood, file_list)

        # Separate successes vs failures
        df_list      = [res[0] for res in results if res[0] is not None]
        df_list_neigh= [res[1] for res in results if res[1] is not None]
        failed_ids   = [res[2] for res in results if res[0] is None]
        failed_msgs  = [res[3] for res in results if res[0] is None]

        # Concatenate DataFrames
        all_prots = pd.concat(df_list,      ignore_index=True) if df_list else pd.DataFrame()
        all_prots = all_prots.assign(attributes="ID=" + all_prots["id"])
        gff_columns = [
            "seqid",       # 1. Sequence ID (e.g. contig name)
            "source",      # 2. Annotation source (e.g. Prokka, GenBank)
            "type",        # 3. Feature type (e.g. gene, CDS, rRNA)
            "start",       # 4. Start coordinate (1-based)
            "end",         # 5. End coordinate (inclusive)
            "score",       # 6. Score (e.g. alignment score or '.' if not used)
            "strand",      # 7. Strand ('+' or '-' or '.')
            "phase",       # 8. Phase (0, 1, 2 for CDS features or '.' otherwise)
            "attributes"   # 9. Semicolon-separated tag-value pairs (e.g. ID=gene1;Name=recA)
        ]
        all_gff = all_prots[gff_columns]
        #remove all_gff columns from all_prots
        all_prots = all_prots.drop(columns=gff_columns)

        all_neigh= pd.concat(df_list_neigh,ignore_index=True) if df_list_neigh else pd.DataFrame()
        # If we have any neighbors, check for minwin
        if not all_neigh.empty:
            # Compute “length” = end_win - start_win
            all_neigh["length"] = all_neigh["end_win"] - all_neigh["start_win"]
            #save all_neigh to tsv
            all_neigh.drop(columns=["sequence"]).to_csv(os.path.join(output_dir, "all_neigh.tsv"), sep="\t", index=False)
            short_contigs = []
            if minwin is not None:
                if minwin_type == "total":
                    short_contigs = all_neigh.loc[
                        all_neigh["length"] < minwin, "unique_id"
                    ].unique().tolist()
                elif minwin_type == "upstream":
                    short_contigs = all_neigh.loc[
                        (all_neigh["start_target"] - all_neigh["start_win"]) < minwin,
                        "unique_id"
                    ].unique().tolist()
                elif minwin_type == "downstream":
                    short_contigs = all_neigh.loc[
                        (all_neigh["end_win"] - all_neigh["end_target"]) < minwin,
                        "unique_id"
                    ].unique().tolist()
                elif minwin_type == "both":
                    short_contigs = all_neigh.loc[
                        ((all_neigh["start_target"] - all_neigh["start_win"]) < minwin)
                        | ((all_neigh["end_win"] - all_neigh["end_target"]) < minwin),
                        "unique_id"
                    ].unique().tolist()
            else:
                short_contigs = []

            # Create neighborhood output folder
            neigh_dir = os.path.join(output_dir, "neighborhood")
            os.makedirs(neigh_dir, exist_ok=True)

            # Report any extraction failures
            if failed_ids:
                console.print(f"[yellow]Failed extractions for {len(failed_ids)} records: {failed_ids}")

            # Mark short contigs as failed
            if short_contigs:
                df.loc[short_contigs, "failed"] = "Genomic context shorter than minimum window size"
                console.print(f"[yellow]{len(short_contigs)} contigs below min window size: {short_contigs}")

            # Mark any failed_ids in df
            for fid, msg in zip(failed_ids, failed_msgs):
                df.loc[fid, "failed"] = msg

            # Identify valid unique_ids (those with nucleotide_id not null & no failure)
            valid_uids = df.loc[
                df["nucleotide_id"].notnull() & df["failed"].isnull(),
                "unique_id"
            ].unique()

            # Build temp_seqid for each neighbor and write neighborhoods.fasta
            all_neigh["temp_seqid"] = (
                all_neigh["seqid"].astype(str) + "_" +
                all_neigh["start_win"].astype(str) + "_" +
                all_neigh["end_win"].astype(str)
            )
            subset_neigh = all_neigh.loc[
                all_neigh["unique_id"].isin(valid_uids),
                ["temp_seqid", "sequence"]
            ].dropna().drop_duplicates(subset=["temp_seqid"])

            neigh_fasta = os.path.join(neigh_dir, "neighborhoods.fasta")
            if not subset_neigh.empty:
                subset_neigh.to_fasta("temp_seqid", "sequence", neigh_fasta)

            # Save updated records to disk
            df.to_csv(os.path.join(output_dir, "records.csv"), index=False)

            # Write proteins to results.fasta
            subset_gff = all_prots.loc[:, ["id", "sequence"]].dropna().drop_duplicates(subset=["id"])
            results_fasta = os.path.join(output_dir, "results.fasta")
            if not subset_gff.empty:
                subset_gff.to_fasta("id", "sequence", results_fasta)

            console.print("✔️  Extracted neighborhoods and wrote output files.")
            return df, all_gff, all_prots, all_neigh, valid_uids

        else:
            # If no neighbors were extracted at all, mark all failed & exit
            for fid, msg in zip(failed_ids, failed_msgs):
                df.loc[fid, "failed"] = msg
            df.to_csv(os.path.join(output_dir, "records.csv"), index=False)
            console.print("🚫  [bold red]Aborting! All IDs failed to yield neighborhoods.")
            sys.exit(1)

    # Call the helper to extract neighborhoods
    
    updated_records, all_gff, all_prots, all_neigh, valid_uids = _extract_neighborhoods(records)
    add_data = all_prots[all_prots["protein_id"] == all_prots["target_prot"]][["unique_id","target_prot"]]
    all_neigh["unique_id"] = all_neigh["unique_id"].astype(str)
    add_data["unique_id"] = add_data["unique_id"].astype(str)
    all_neigh = all_neigh.merge(add_data, on = "unique_id", how = "left")

    return {
        "records": updated_records,
        "all_gff": all_gff,
        "all_prots": all_prots,
        "all_neigh": all_neigh,
        "valid_uids": valid_uids
    }
