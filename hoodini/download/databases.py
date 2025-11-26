import os
import subprocess
import tarfile
from pathlib import Path
from importlib.resources import files
from typing import Optional


from hoodini.utils.logging_utils import console, stage_header, stage_done
from hoodini.utils.downloader import download_with_aria2c


EMAPPER_URL = "http://eggnog6.embl.de/download/emapperdb-5.0.2/mmseqs.tar.gz"
EGGNOG_OG = "https://hoodini.bio/eggnog_og.parquet"
EGGNOG_PROTS = "https://hoodini.bio/eggnog_prots.parquet"
CONTIGS_URL = "https://hoodini.bio/contig_lengths.parquet"


def _run_cmd(cmd, cwd: Optional[Path] = None):
    try:
        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
        subprocess.run(cmd, check=True, cwd=(str(cwd) if cwd is not None else None))
        return True
    except FileNotFoundError:
        console.print(f"[yellow]Command not found: {cmd[0]} — skipping.[/yellow]")
        return False
    except subprocess.CalledProcessError as e:
        console.print(f"[yellow]Command failed ({cmd[0]}): {e} — continuing.[/yellow]")
        return False



# Use aria2p-based downloader for all downloads
def _download_url(url: str, dest: Path, num_threads: int = 0):
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Downloading {url} -> {dest}[/dim]")
    try:
        out_name = Path(dest).name
        result_files = download_with_aria2c([url], dest.parent, show_progress=True, out_names=[out_name], num_threads=num_threads)
        # aria2c returns the full path; check if our dest is present
        if any(str(dest) == f for f in result_files):
            return True
        # fallback: move the file if needed (aria2c may use original filename)
        import shutil
        for f in result_files:
            pf = Path(f)
            # if aria2c returned a file with the expected name, move it
            if pf.name == out_name and pf != dest:
                shutil.move(str(pf), str(dest))
                return True
            # if aria2c returned a directory, try to find the file inside
            if pf.is_dir():
                candidate = pf / out_name
                if candidate.exists() and candidate.is_file():
                    shutil.move(str(candidate), str(dest))
                    return True
                files_inside = [p for p in pf.iterdir() if p.is_file()]
                if files_inside:
                    shutil.move(str(files_inside[0]), str(dest))
                    return True
        return False
    except Exception as e:
        console.print(f"[yellow]Download failed: {e}[/yellow]")
        return False


def extract_tar(tar_path: Path, dest_dir: Path, threads: int = 0) -> bool:
    """Extract tar.gz using pigz if available (fast), else fall back to tar, else Python tarfile."""
    import shutil

    pigz = shutil.which("pigz")
    tar_bin = shutil.which("tar")

    if pigz and tar_bin:
        pigz_cmd = f"pigz -p {threads}" if threads > 0 else "pigz"
        cmd = [tar_bin, f"--use-compress-program={pigz_cmd}", "-xvf", str(tar_path), "-C", str(dest_dir)]
        try:
            console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]pigz extraction failed: {e}, falling back to tarfile[/yellow]")

    if tar_bin:
        try:
            subprocess.run([tar_bin, "-xzf", str(tar_path), "-C", str(dest_dir)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]tar extraction failed: {e}, falling back to tarfile[/yellow]")

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=dest_dir)
        return True
    except Exception as e:
        console.print(f"[red]tarfile extraction failed: {e}[/red]")
        return False

def main(
    force: bool = False,
    skip_padloc: bool = False,
    skip_deffinder: bool = False,
    skip_genomad: bool = False,
    skip_emapper: bool = False,
    skip_parquet: bool = False,
    skip_contig_lengths: bool = False,
    num_threads: int = 0,
):
    stage_header("Downloading databases and support files", "⬇️")

    data_dir = files("hoodini").joinpath("data")
    emapper_dir = data_dir.joinpath("emapper")
    contig_dir = data_dir.joinpath("contig_lengths")
    genomad_db = data_dir.joinpath("genomad_db")

    # 1) Run system installers / db updaters (best-effort)
    if not skip_padloc:
        _run_cmd(["padloc", "--db-update"])  # update padloc DB
    else:
        console.print("[dim]Skipping padloc DB update (--skip-padloc)[/dim]")

    if not skip_deffinder:
        _run_cmd(["defense-finder", "update"])  # install defense-finder models
    else:
        console.print("[dim]Skipping defense-finder model install (--skip-deffinder)[/dim]")

    # genomad expects current directory; run in emapper_dir to download into package data
    if not skip_genomad:
        genomad_db.mkdir(parents=True, exist_ok=True)
        _run_cmd(["genomad", "download-database", str(genomad_db)])
    else:
        console.print("[dim]Skipping genomad database download (--skip-genomad)[/dim]")

    # 2) Download eggNOG emapper mmseqs DB and extract
    emapper_dir.mkdir(parents=True, exist_ok=True)
    emapper_tar = emapper_dir.joinpath("mmseqs.tar.gz")
    mmseqs_folder = emapper_dir.joinpath("mmseqs")

    if skip_emapper:
        console.print("[dim]Skipping mmseqs/emapper DB download (--skip-emapper)[/dim]")
    else:
        if force or not mmseqs_folder.exists():
            console.print(f"[yellow]Downloading and extracting mmseqs; folder missing or --force is set[/yellow]")

            ok = _download_url(EMAPPER_URL, emapper_tar, num_threads=num_threads)
            if ok:
                ok_extract = extract_tar(emapper_tar, emapper_dir, threads=num_threads)
                if ok_extract:
                    console.print(f"[green]Extracted {emapper_tar.name} into {emapper_dir}[/green]")
                else:
                    console.print(f"[red]Failed to extract {emapper_tar.name}[/red]")
                #remove tar file to save space
                try:
                    emapper_tar.unlink()
                    console.print(f"[dim]Removed {emapper_tar} to save space[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Failed to remove {emapper_tar}: {e}[/yellow]")
                # After successful extraction, attempt to create a GPU-ready padded DB
                padded_prefix = mmseqs_folder.joinpath("mmseqs.db_pad")
                if not padded_prefix.exists():
                    try:
                        cmd = ["mmseqs", "makepaddedseqdb", "mmseqs.db", "mmseqs.db_pad"]
                        _run_cmd(cmd, cwd=mmseqs_folder)
                    except Exception as e:
                        console.print(f"[yellow]Failed to create padded mmseqs DB: {e}[yellow]")
            else:
                console.print(f"[red]Download failed from {EMAPPER_URL}[/red]")
        else:
            console.print(f"[dim]{mmseqs_folder} already exists; skipping download and extraction[/dim]")

    # 3) Download parquet support files
    if skip_parquet:
        console.print("[dim]Skipping eggNOG parquet support files (--skip-parquet)[/dim]")
    else:
        for url in (EGGNOG_OG, EGGNOG_PROTS):
            dest = emapper_dir.joinpath(Path(url).name)
            if force or not dest.exists():
                _download_url(url, dest, num_threads=num_threads)
            else:
                console.print(f"[dim]{dest} exists; use --force to re-download[/dim]")

    # 4) Download contig_lengths.parquet into data/contig_lengths
    contig_dir.mkdir(parents=True, exist_ok=True)
    contig_dest = contig_dir.joinpath("contig_lengths.parquet")
    if skip_contig_lengths:
        console.print("[dim]Skipping contig_lengths.parquet download (--skip-contig-lengths)[/dim]")
    else:
        if force or not contig_dest.exists():
            _download_url(CONTIGS_URL, contig_dest, num_threads=num_threads)
        else:
            console.print(f"[dim]{contig_dest} exists; use --force to re-download[/dim]")

    stage_done("Databases download complete")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(force=args.force)
