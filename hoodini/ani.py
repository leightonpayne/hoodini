import os
import subprocess
from pathlib import Path
import io
from typing import Optional

import pandas as pd

from hoodini.utils.logging_utils import console


def _write_fasta_from_df(df: pd.DataFrame, out_path: Path, id_col: str = 'temp_seqid', seq_col: str = 'sequence') -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        for _, r in df.iterrows():
            hdr = str(r.get(id_col, '')).strip()
            seq = str(r.get(seq_col, '')).strip()
            if not hdr or not seq:
                continue
            fh.write(f">{hdr}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")


def run_ani(output_dir: str, all_neigh: Optional[pd.DataFrame] = None, threads: int = 8, evalue: Optional[float] = None, mode: str = "fastani") -> pd.DataFrame:
    """Run pairwise ANI using either `skani` or `fastANI` and return a skani-like DataFrame.

    Parameters
    - output_dir: project output folder (expects neighborhood/neighborhoods.fasta)
    - all_neigh: optional DataFrame containing neighborhood sequences (columns 'temp_seqid' and 'sequence') if fasta missing
    - threads: number of threads to run
    - mode: 'skani' or 'fastani'

    Returns
    - pandas.DataFrame with columns: Ref_name, Query_name, ANI, Align_fraction_ref, Align_fraction_query
    """
    out = Path(output_dir)
    fasta_path = out / 'neighborhood' / 'neighborhoods.fasta'

    if not fasta_path.exists():
        if all_neigh is None:
            raise FileNotFoundError(f"neighborhood FASTA not found at {fasta_path} and no `all_neigh` provided")
        console.print(f"neighborhoods.fasta not found; writing from provided DataFrame to {fasta_path}")
        _write_fasta_from_df(all_neigh, fasta_path)

    if mode.lower() == 'skani':
        cmd = [
            'skani', 'triangle',
            '-i', str(fasta_path),
            '-E', '--small-genomes',
            '-t', str(threads)
        ]
        console.print(f"Running: {' '.join(cmd)}")
        # capture skani output to a log file for inspection instead of printing to terminal
        skani_log = out / 'skani_triangle.log'
        with open(skani_log, 'w') as logfh:
            proc = subprocess.run(cmd, check=True, stdout=logfh, stderr=logfh, text=True)
        # read produced log to parse skani's tabular output
        with open(skani_log, 'r') as logfh:
            out_text = logfh.read()
        # Parse skani output
        df = pd.read_csv(io.StringIO(out_text), sep='\t', header=0)
        # Expected columns include Ref_name, Query_name, ANI, Align_fraction_ref, Align_fraction_query
        # Normalize column names and ensure both align fraction columns exist
        if 'Align_fraction_query' not in df.columns and 'Align_fraction_ref' in df.columns:
            # sometimes skani repeats the same column; create a symmetric placeholder
            df['Align_fraction_query'] = df['Align_fraction_ref']
        df['Ref_name'] = df['Ref_name'].apply(lambda x: str(x).split('/')[-1])
        df['Query_name'] = df['Query_name'].apply(lambda x: str(x).split('/')[-1])
        # Ensure numeric conversions
        df['ANI'] = pd.to_numeric(df['ANI'], errors='coerce')
        df['Align_fraction_ref'] = pd.to_numeric(df.get('Align_fraction_ref'), errors='coerce')
        df['Align_fraction_query'] = pd.to_numeric(df.get('Align_fraction_query'), errors='coerce')
        df = df[df['Ref_name'] != df['Query_name']]
        return df[['Ref_name', 'Query_name', 'ANI', 'Align_fraction_ref', 'Align_fraction_query']]

    elif mode.lower() == 'fastani':
        # split multi-fasta into individual genome files
        split_dir = out / 'ani_split'
        split_dir.mkdir(parents=True, exist_ok=True)
        try:
            from Bio import SeqIO
        except Exception as exc:
            raise ImportError('Biopython is required for fastANI mode (SeqIO)') from exc

        genome_files = []
        for record in SeqIO.parse(str(fasta_path), 'fasta'):
            header = record.id
            filename = "".join(c if c.isalnum() or c in '-._' else '_' for c in header) + '.fasta'
            out_path = split_dir / filename
            SeqIO.write(record, str(out_path), 'fasta')
            genome_files.append(str(out_path))

        file_list_path = out / 'fastani_genome_list.txt'
        with open(file_list_path, 'w') as fh:
            for p in genome_files:
                fh.write(p + '\n')

        fastani_output = out / 'fastani_output.tsv'
        cmd = [
            'fastANI',
            '--ql', str(file_list_path),
            '--rl', str(file_list_path),
            '-o', str(fastani_output),
            '-t', str(threads)
        ]
        console.print(f"Running FastANI: {' '.join(cmd)}")
        fastani_log = out / 'fastani_all.log'
        with open(fastani_log, 'w') as logfh:
            subprocess.run(cmd, check=True, stdout=logfh, stderr=logfh, text=True)

        if not fastani_output.exists():
            raise FileNotFoundError(f"fastANI did not produce expected output at {fastani_output}")

        df = pd.read_csv(fastani_output, sep='\t', header=None,
                         names=['query', 'reference', 'ani', 'frags_matched', 'frags_total_query'])

        reverse_frag_total = df.set_index(['reference', 'query'])['frags_total_query'].to_dict()

        rows = []
        for _, row in df.iterrows():
            q = row['query']
            r = row['reference']
            ani = row['ani']
            frags_matched = row['frags_matched']
            frags_total_query = row['frags_total_query']

            frags_total_ref = reverse_frag_total.get((q, r), None)

            align_fraction_query = (frags_matched / frags_total_query) if frags_total_query else 0
            align_fraction_ref = (frags_matched / frags_total_ref) if frags_total_ref else None

            rows.append({
                'Ref_name': str(r).split('/')[-1],
                'Query_name': str(q).split('/')[-1],
                'ANI': float(ani),
                'Align_fraction_ref': (align_fraction_ref * 100.0) if align_fraction_ref is not None else None,
                'Align_fraction_query': float(align_fraction_query) * 100.0
            })

        skani_like_df = pd.DataFrame(rows)
        # remove self-hits
        skani_like_df = skani_like_df[skani_like_df['Ref_name'] != skani_like_df['Query_name']]

        # canonicalize reciprocal pairs (A,B) and (B,A) into a single key
        def pair_key(row):
            a, b = row['Ref_name'], row['Query_name']
            return tuple(sorted((a, b)))

        skani_like_df['pair_key'] = skani_like_df.apply(pair_key, axis=1)

        # For reciprocal pairs (A,B) and (B,A) compute average ANI and average align fractions
        agg = skani_like_df.groupby('pair_key').agg({
            'ANI': 'mean',
            'Align_fraction_ref': 'mean',
            'Align_fraction_query': 'mean'
        }).reset_index()

        # Expand pair_key into Ref_name and Query_name (sorted order)
        agg['Ref_name'] = agg['pair_key'].apply(lambda t: t[0])
        agg['Query_name'] = agg['pair_key'].apply(lambda t: t[1])

        result_df = agg[['Ref_name', 'Query_name', 'ANI', 'Align_fraction_ref', 'Align_fraction_query']]
        return result_df

    else:
        raise ValueError("mode must be 'skani' or 'fastani'")
