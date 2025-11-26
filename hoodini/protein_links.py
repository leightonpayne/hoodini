import os
from pathlib import Path
from typing import Optional

import pandas as pd

from hoodini.utils.logging_utils import console


def run_protein_links(output_dir: str, all_prots: pd.DataFrame, threads: int = 4, evalue: float = 1e-5) -> pd.DataFrame:
    """Build Diamond DB from `all_prots` (DataFrame) and run all-vs-all blastp.

    Parameters
    - output_dir: base output folder where results.fasta will be read/written
    - all_prots: DataFrame containing at least columns ['id', 'sequence'] or ['protein_id', 'sequence']
    - threads: number of threads for Diamond
    - evalue: evalue cutoff for blastp

    Returns
    - pandas.DataFrame with columns [qseqid, sseqid, pident, length, evalue, bitscore]
      with self-hits removed (same genome prefix in qseqid and sseqid).
    """
    # Find sequence column names
    df = all_prots.copy()
    if 'sequence' not in df.columns:
        raise ValueError("all_prots must contain a 'sequence' column")

    if 'id' in df.columns:
        id_col = 'id'
    elif 'protein_id' in df.columns:
        id_col = 'protein_id'
    else:
        raise ValueError("all_prots must contain an 'id' or 'protein_id' column")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = out_dir / 'results.fasta'
    db_path = out_dir / 'results.dmnd'

    # Write FASTA if not present
    if not fasta_path.exists():
        console.print(f"Writing protein FASTA to {fasta_path}")
        with open(fasta_path, 'w') as fh:
            for _, r in df.iterrows():
                fh.write(f">{r[id_col]}\n")
                seq = r['sequence']
                # wrap at 80 chars
                for i in range(0, len(seq), 80):
                    fh.write(seq[i:i+80] + "\n")

    # Try to import Diamond wrapper
    try:
        from diamondonpy import Diamond
    except Exception as e:
        raise RuntimeError("diamondonpy Diamond wrapper not available; please install diamondonpy") from e

    diamond = Diamond()

    console.print("Building Diamond database...")
    diamond.makedb(db=str(db_path), input_file=str(fasta_path), threads=threads)
    console.print(f"Diamond database built: {db_path}")

    console.print("Running all-vs-all blastp (Diamond)...")
    results_df = diamond.blastp(
        db=str(db_path),
        query=str(fasta_path),
        evalue=evalue,
        threads=threads,
        outfmt="6 qseqid sseqid pident length evalue bitscore"
    )

    # Remove DB file(s)
    try:
        if db_path.exists():
            db_path.unlink()
    except Exception:
        console.print(f"[yellow]warning[/yellow] could not remove diamond DB {db_path}")

    # Ensure DataFrame columns and types
    expected = ['qseqid', 'sseqid', 'pident', 'length', 'evalue', 'bitscore']
    if not all(c in results_df.columns for c in expected):
        # Attempt to coerce if blastp returned a list
        results_df = pd.DataFrame(results_df, columns=expected)

    # Exclude self-hits based on genome prefix before the first '|'
    def same_genome(a: str, b: str) -> bool:
        try:
            return a.split("|")[0] == b.split("|")[0]
        except Exception:
            return a == b

    mask = results_df.apply(lambda row: same_genome(row['qseqid'], row['sseqid']), axis=1)
    filtered = results_df[~mask].reset_index(drop=True)

    console.print(f"Protein pairwise comparisons complete: {len(filtered)} hits (self-hits removed)")
    return filtered
