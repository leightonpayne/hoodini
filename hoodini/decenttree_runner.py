"""Run DecentTree on a pairwise distance table and return Newick string.

The table should contain rows with (query_id, target_id, distance).
This module will build a relaxed PHYLIP-like distance matrix (tab-separated,
no name length limits), run DecentTree (bundled in `hoodini/extra_tools`) and
return the resulting Newick string.

The function `run_decenttree_from_table` accepts column-name arguments so it
can be used with different input DataFrame schemas.
"""
from __future__ import annotations
from pathlib import Path
import platform
import subprocess
import tempfile
import csv
from typing import Iterable, Optional
from importlib.resources import files

import pandas as pd


def _choose_decenttree_binary() -> Path:
    sysname = platform.system().lower()
    if sysname.startswith("darwin"):
        p = files('hoodini').joinpath('extra_tools', 'decenttree_macos')
    else:
        p = files('hoodini').joinpath('extra_tools', 'decenttree_linux')
    if not p.exists():
        raise FileNotFoundError(f"DecentTree binary not found at {p}")
    return p


def _normalize_algorithm(name: str) -> str:
    """Map common algorithm aliases to DecentTree accepted names.

    Accepts case-insensitive inputs like 'nj', 'nj-r', 'bionj' and returns
    the proper DecentTree token (e.g. 'NJ', 'NJ-R', 'BIONJ').
    """
    if not name:
        return "NJ"
    n = str(name).strip().lower()
    mapping = {
        "nj": "NJ",
        "nj-r": "NJ-R",
        "nj_r": "NJ-R",
        "njr": "NJ-R",
        "nj-v": "NJ-V",
        "bionj": "BIONJ",
        "bionj-r": "BIONJ-R",
        "rapidnj": "RapidNJ",
        "unj": "UNJ",
        "upgma": "UPGMA",
        "auction": "AUCTION",
        "bionj-v": "BIONJ-V",
        "nj-r-d": "NJ-R-D",
    }
    if n in mapping:
        return mapping[n]
    # If user already passed a likely valid token (case-insensitive), try upper-casing
    cand = str(name).upper()
    return cand


def _build_relaxed_phylip(df: pd.DataFrame, qcol: str, tcol: str, dcol: str, out_path: Path, fill_missing=None) -> None:
    """Create a relaxed PHYLIP-like matrix (tab-separated) at out_path.

    The matrix will include all unique sequence ids found in qcol and tcol,
    and distances will be placed in the appropriate cells. Missing entries will
    be set to 0 (or a large value if you prefer).
    """
    # If ids is provided, use it as the full set of sequence IDs for the matrix
    # Otherwise, use all unique IDs found in the DataFrame
    if hasattr(fill_missing, "__iter__") and not isinstance(fill_missing, str):
        # If fill_missing is a tuple (fill_missing_value, ids)
        fill_value, ids = fill_missing
    else:
        ids = pd.unique(df[[qcol, tcol]].values.ravel())
        fill_value = fill_missing
    ids = list(ids)
    id_index = {idv: i for i, idv in enumerate(ids)}
    n = len(ids)

    # Determine fill value for missing entries
    if fill_value is None:
        fill_value = 0.0
    elif isinstance(fill_value, str):
        sval = fill_value.lower()
        vals = df[dcol].dropna().astype(float).values
        if vals.size == 0:
            fill_value = 0.0
        elif sval == "max":
            fill_value = float(vals.max())
        elif sval in ("max+2std", "max_plus_2std", "max+2*std"):
            # Use max + 2*std to emulate very large distances for missing pairs
            fill_value = float(vals.max()) + 2.0 * float(vals.std())
        elif sval == "min":
            fill_value = float(vals.min())
        else:
            fill_value = 0.0
    else:
        try:
            fill_value = float(fill_value)
        except Exception:
            fill_value = 0.0

    # Initialize matrix with fill_value
    mat = [[fill_value] * n for _ in range(n)]

    # Build a lookup for distances
    df_lookup = df.set_index([qcol, tcol])[dcol]

    # Fill matrix with provided distances for all pairs in ids
    for i, q in enumerate(ids):
        for j, t in enumerate(ids):
            # Try both (q, t) and (t, q) for symmetry
            v = None
            if (q, t) in df_lookup:
                v = df_lookup.get((q, t))
            elif (t, q) in df_lookup:
                v = df_lookup.get((t, q))
            if v is not None and not pd.isna(v):
                try:
                    d = float(v)
                    mat[i][j] = d
                except Exception:
                    pass

    # Ensure parent directory exists before writing the file
    out_path_parent = out_path.parent
    out_path_parent.mkdir(parents=True, exist_ok=True)

    # Write relaxed phylip: first line N, then rows: name \t val1 \t val2 ...
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        fh.write(f"{n}\n")
        for i, idv in enumerate(ids):
            row = [idv] + [str(x) for x in mat[i]]
            writer.writerow(row)


def run_decenttree_from_table(df: pd.DataFrame,
                              qcol: str = "query",
                              tcol: str = "target",
                              dcol: str = "distance",
                              algorithm: str = "nj",
                              threads: int = 1,
                              decenttree_bin: Optional[Path] = None,
                              fill_missing=None,
                              ids: Optional[Iterable[str]] = None) -> str:
    """Run DecentTree on the provided pairwise distance table and return Newick string.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing pairwise distances.
    qcol, tcol, dcol : str
        Column names for query id, target id, and distance.
    algorithm : str
        Algorithm to pass to DecentTree via `-t` (e.g., 'nj' or others supported).
    threads : int
        Number of threads to pass to DecentTree via `-nt`.
    decenttree_bin : Path, optional
        Explicit path to binary; if None the bundled extra_tools binary is used.
    ids : Optional[Iterable[str]], optional
        List of ids to ensure are present in the matrix.

    Returns
    -------
    str
        Newick tree string returned by DecentTree.
    """
    if decenttree_bin is None:
        decenttree_bin = _choose_decenttree_binary()

    # Prepare a working DataFrame dt_df that contains qcol, tcol, dcol (distance)
    dt_df = df.copy()

    # If the table contains AAI or ANI columns, convert to distance (100 - value).
    if 'AAI' in dt_df.columns and dcol not in dt_df.columns:
        dt_df['AAI'] = pd.to_numeric(dt_df['AAI'], errors='coerce')
        if dt_df['AAI'].max() <= 1.0:
            dt_df['AAI'] = dt_df['AAI'] * 100.0
        dt_df['distance'] = 100.0 - dt_df['AAI']
        dcol_use = 'distance'
    elif 'ANI' in dt_df.columns and dcol not in dt_df.columns:
        dt_df['ANI'] = pd.to_numeric(dt_df['ANI'], errors='coerce')
        if dt_df['ANI'].max() <= 1.0:
            dt_df['ANI'] = dt_df['ANI'] * 100.0
        dt_df['distance'] = 100.0 - dt_df['ANI']
        dcol_use = 'distance'
    else:
        dcol_use = dcol

    # ensure qcol/tcol exist
    if qcol not in dt_df.columns or tcol not in dt_df.columns:
        raise ValueError(f"Input DataFrame must contain columns: {qcol}, {tcol}")

    # Always use ids as the full set of sequence IDs for the matrix if provided
    # Otherwise, use all unique IDs found in the DataFrame
    if ids is not None:
        valid_uids = [str(x) for x in ids]
    else:
        valid_uids = list(pd.unique(dt_df[[qcol, tcol]].values.ravel()))

    with tempfile.TemporaryDirectory() as td:
        tdpth = Path(td)
        phylip_path = tdpth / "matrix.phylip"
        out_newick = tdpth / "tree.nwk"

    # Build relaxed phylip using all possible pairs from valid_uids
    # Missing pairs will be filled with the specified fill_value (e.g., max+2std)
    _build_relaxed_phylip(dt_df, qcol, tcol, dcol_use, phylip_path, fill_missing=(fill_missing, valid_uids))
    alg = _normalize_algorithm(algorithm)
    cmd = [str(decenttree_bin), "-in", str(phylip_path), "-out", str(out_newick), "-t", alg, "-nt", str(int(threads))]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"DecentTree failed: rc={proc.returncode}\nstderr={proc.stderr}\nstdout={proc.stdout}")

    if not out_newick.exists():
        raise RuntimeError("DecentTree did not produce output tree")

    newick = out_newick.read_text(encoding="utf-8")
    return newick.strip()


def run_decenttree_from_matrix(matrix_df: pd.DataFrame,
                                algorithm: str = "nj",
                                threads: int = 1,
                                decenttree_bin: Optional[Path] = None) -> str:
    """Run DecentTree on a square distance matrix (pandas DataFrame) and return Newick.

    matrix_df must be square with identical index and columns in the same order.
    """
    if not isinstance(matrix_df, pd.DataFrame):
        raise TypeError("matrix_df must be a pandas.DataFrame")
    if matrix_df.shape[0] != matrix_df.shape[1]:
        raise ValueError("matrix_df must be square")
    if not all(str(i) == str(c) for i, c in zip(matrix_df.index.astype(str), matrix_df.columns.astype(str))):
        # reorder columns to match index if possible
        cols = [c for c in matrix_df.index.astype(str) if c in matrix_df.columns]
        if len(cols) == matrix_df.shape[0]:
            matrix_df = matrix_df.loc[:, cols]
        else:
            raise ValueError("matrix_df index and columns must contain the same labels")

    if decenttree_bin is None:
        decenttree_bin = _choose_decenttree_binary()

    with tempfile.TemporaryDirectory() as td:
        tdpth = Path(td)
        phylip_path = tdpth / "matrix.phylip"
        out_newick = tdpth / "tree.nwk"

        # write relaxed phylip
        ids = [str(i) for i in matrix_df.index]
        n = len(ids)
        with open(phylip_path, "w", newline="") as fh:
            fh.write(f"{n}\n")
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
            for i, idv in enumerate(ids):
                row = [idv] + [str(x) for x in matrix_df.iloc[i].tolist()]
                writer.writerow(row)
        alg = _normalize_algorithm(algorithm)
        cmd = [str(decenttree_bin), "-in", str(phylip_path), "-out", str(out_newick), "-t", alg, "-nt", str(int(threads))]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"DecentTree failed: rc={proc.returncode}\nstderr={proc.stderr}\nstdout={proc.stdout}")

        if not out_newick.exists():
            raise RuntimeError("DecentTree did not produce output tree")

        newick = out_newick.read_text(encoding="utf-8")
        return newick.strip()
