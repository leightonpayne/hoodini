import os
import subprocess
from pathlib import Path
import math
from collections import Counter
from typing import Optional, Tuple
import mappy as mp  # noqa


import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from hoodini.utils.logging_utils import console

def _fasta_ids(fasta_path: str):
    from Bio import SeqIO
    ids = [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]
    if not ids:
        raise ValueError("No sequences found.")
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise ValueError(f"Duplicate FASTA IDs: {dup[:10]}{'...' if len(dup) > 10 else ''}")
    return ids


def _fasta_lengths(path: str):
    from Bio import SeqIO
    idx = SeqIO.index(path, "fasta")
    try:
        return {k: len(idx[k].seq) for k in idx.keys()}
    finally:
        idx.close()


def _write_intergenic_fasta(all_gff, fasta_path: str, out_fasta: str, all_neigh: Optional[pd.DataFrame] = None):
    """
    Create a FASTA of intergenic regions from neighborhoods.fasta using a GFF (path or DataFrame).
    Returns a DataFrame with columns ['temp_seqid','seqid','start_win'] describing the mapping for each record.
    The FASTA record ids are created as "{temp_seqid}__inter_{start}_{end}" where start/end are 1-based coordinates
    relative to the neighborhood sequence.
    """
    from Bio import SeqIO
    import pandas as pd

    # load GFF into DataFrame with columns seqid,type,start,end
    if all_gff is None:
        raise ValueError("all_gff is required for intergenic FASTA generation")
    if isinstance(all_gff, pd.DataFrame):
        gff = all_gff.copy()
    else:
        # assume path-like
        cols = ['seqid','source','type','start','end','score','strand','phase','attributes']
        gff = pd.read_csv(str(all_gff), sep='\t', comment='#', header=None, names=cols, low_memory=False)

    # keep only coding features to exclude
    coding_types = set(['CDS', 'gene'])
    gff_coding = gff[gff['type'].isin(coding_types)].copy()

    # build mapping from fasta header -> canonical seqid and start_win (if available)
    temp_to_seqid = {}
    start_map_init = {}
    if all_neigh is not None and not all_neigh.empty:
        for _, nr in all_neigh.iterrows():
            temp = str(nr['temp_seqid']) if 'temp_seqid' in nr and pd.notna(nr['temp_seqid']) else None
            seqid = str(nr['seqid']) if 'seqid' in nr and pd.notna(nr['seqid']) else temp
            if temp:
                temp_to_seqid[temp] = seqid
            if seqid and 'start_win' in nr and pd.notna(nr['start_win']):
                start_map_init[seqid] = int(nr['start_win'])

    out_records = []
    meta_rows = []
    with open(out_fasta, 'w') as outfh:
        for rec in SeqIO.parse(str(fasta_path), 'fasta'):
            hdr = rec.id
            seq = str(rec.seq)
            seqlen = len(seq)
            canonical = temp_to_seqid.get(hdr, hdr)

            # get coding intervals for this canonical id (try exact and name/stem matches)
            cand = gff_coding[gff_coding['seqid'].isin([canonical, Path(canonical).name, Path(canonical).stem])]
            coding_intervals = []

            # determine start_win (global start of this neighborhood) if provided
            start_win = start_map_init.get(canonical, None)
            if start_win is not None:
                # GFF coords are genome/global coords -> convert to local by subtracting start_win
                for _, r in cand.iterrows():
                    try:
                        s_global = int(r['start']); e_global = int(r['end'])
                    except Exception:
                        continue
                    s = s_global - int(start_win) + 1
                    e = e_global - int(start_win) + 1
                    # clamp to sequence length
                    s = max(1, min(s, seqlen))
                    e = max(1, min(e, seqlen))
                    if e >= s:
                        coding_intervals.append((s, e))
            else:
                # fallback: assume GFF coords are already relative to the neighborhood
                console.log(f"Warning: no start_win for {canonical}; treating GFF coords as neighborhood-local")
                for _, r in cand.iterrows():
                    try:
                        s = int(r['start']); e = int(r['end'])
                    except Exception:
                        continue
                    # clamp
                    s = max(1, min(s, seqlen))
                    e = max(1, min(e, seqlen))
                    if e >= s:
                        coding_intervals.append((s, e))

            # merge coding intervals
            coding_intervals.sort()
            merged = []
            for iv in coding_intervals:
                if not merged:
                    merged.append(list(iv))
                else:
                    if iv[0] <= merged[-1][1] + 1:
                        merged[-1][1] = max(merged[-1][1], iv[1])
                    else:
                        merged.append([iv[0], iv[1]])

            # compute complement (intergenic) intervals
            intergenic = []
            pos = 1
            for s, e in merged:
                if pos < s:
                    intergenic.append((pos, s-1))
                pos = e + 1
            if pos <= seqlen:
                intergenic.append((pos, seqlen))

            # if no coding intervals, whole sequence is intergenic
            if not merged:
                intergenic = [(1, seqlen)]

            # write intergenic sequences
            for s, e in intergenic:
                if e < s:
                    continue
                sub = seq[s-1:e]
                temp_id = f"{hdr}__inter_{s}_{e}"
                outfh.write(f">{temp_id}\n")
                # wrap at 80
                for i in range(0, len(sub), 80):
                    outfh.write(sub[i:i+80] + "\n")

                # record mapping: temp_seqid -> canonical seqid and start (global within neighborhood)
                start_win = start_map_init.get(canonical, 0)
                meta_rows.append({'temp_seqid': temp_id, 'seqid': canonical, 'start_win': int(s) + int(start_win) - 1})

    meta_df = pd.DataFrame(meta_rows, columns=['temp_seqid','seqid','start_win'])
    return meta_df


def _skani_like_from_blast(hits_df: pd.DataFrame, seq_lengths: dict,
                           round_digits=3, overlap_on="query", overlap_tol=0) -> pd.DataFrame:
    import math
    def _norm_iv(a, b):
        a = int(a); b = int(b)
        lo, hi = (a - 1, b) if a <= b else (b - 1, a)
        return lo, hi

    def _overlap_len(iv1, iv2):
        a1, a2 = iv1; b1, b2 = iv2
        return max(0, min(a2, b2) - max(a1, b1))

    def _merge_coverage_len(intervals):
        if not intervals:
            return 0
        ints = sorted(intervals)
        total = 0
        s, e = ints[0]
        for x, y in ints[1:]:
            if x <= e:
                e = max(e, y)
            else:
                total += (e - s)
                s, e = x, y
        total += (e - s)
        return total

    def _select_non_overlapping_blast(hits, on="query", tol=0):
        key_iv = ("qstart", "qend") if on == "query" else ("sstart", "send")
        order = sorted(range(len(hits)), key=lambda i: (
            -(hits[i].get("bitscore") or 0.0),
            -(hits[i].get("length") or 0),
            -(hits[i].get("pident") or 0.0),
        ))
        selected, ivs = [], []
        for i in order:
            a = hits[i].get(key_iv[0]); b = hits[i].get(key_iv[1])
            if a is None or b is None:
                continue
            iv = _norm_iv(a, b)
            if all(_overlap_len(iv, siv) <= tol for siv in ivs):
                selected.append(i); ivs.append(iv)
        return selected

    if hits_df.empty:
        return pd.DataFrame(columns=["Ref_name", "Query_name", "ANI", "Align_fraction_ref", "Align_fraction_query"])

    df = hits_df.copy()
    for c in ["pident", "length", "qstart", "qend", "sstart", "send", "bitscore", "qlen", "slen"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    out = []
    for (q_id, t_id), g in df.groupby(["qseqid", "sseqid"], sort=False):
        hits = g.to_dict("records")
        sel_idx = _select_non_overlapping_blast(hits, on=overlap_on, tol=overlap_tol)
        sel = [hits[i] for i in sel_idx]

        sum_blen = sum(int(h["length"]) for h in sel if pd.notna(h.get("length")))
        sum_mlen = sum(int(round((h["pident"]/100.0) * h["length"])) for h in sel
                       if pd.notna(h.get("pident")) and pd.notna(h.get("length")))
        ani = (100.0 * sum_mlen / sum_blen) if sum_blen > 0 else float('nan')

        q_ivs = [_norm_iv(h["qstart"], h["qend"]) for h in sel if pd.notna(h.get("qstart")) and pd.notna(h.get("qend"))]
        t_ivs = [_norm_iv(h["sstart"], h["send"]) for h in sel if pd.notna(h.get("sstart")) and pd.notna(h.get("send"))]
        covered_q = _merge_coverage_len(q_ivs)
        covered_t = _merge_coverage_len(t_ivs)

        q_len = g["qlen"].iloc[0] if "qlen" in g.columns and pd.notna(g["qlen"].iloc[0]) else seq_lengths.get(q_id)
        t_len = g["slen"].iloc[0] if "slen" in g.columns and pd.notna(g["slen"].iloc[0]) else seq_lengths.get(t_id)

        frac_q = (covered_q / q_len) * 100 if q_len else float('nan')
        frac_t = (covered_t / t_len) * 100 if t_len else float('nan')

        out.append({
            "Ref_name": t_id,
            "Query_name": q_id,
            "ANI": None if math.isnan(ani) else round(ani, round_digits),
            "Align_fraction_ref": None if math.isnan(frac_t) else round(frac_t, round_digits),
            "Align_fraction_query": None if math.isnan(frac_q) else round(frac_q, round_digits),
        })

    return pd.DataFrame(out, columns=["Ref_name", "Query_name", "ANI", "Align_fraction_ref", "Align_fraction_query"])


def _run_mappy_target_block(args):
    import mappy as mp
    from Bio import SeqIO

    (fasta_path, preset, min_mapq, mm2_threads_per_worker, tid, q_ids) = args
    idx = SeqIO.index(fasta_path, "fasta")
    try:
        t_seq = str(idx[tid].seq)
        al = mp.Aligner(seq=t_seq, preset=preset, n_threads=mm2_threads_per_worker)
        if not al:
            return []

        out = []
        for qid in q_ids:
            q_seq = str(idx[qid].seq)
            for h in al.map(q_seq):
                if getattr(h, "mapq", 0) < min_mapq:
                    continue

                mlen = getattr(h, "mlen", None)
                blen = getattr(h, "blen", None)
                pid_mlen_blen = (100.0 * mlen / blen) if (mlen is not None and blen) else None

                NM = getattr(h, "NM", None)
                q_st = getattr(h, "q_st", None); q_en = getattr(h, "q_en", None)
                aln_len_q = (q_en - q_st) if (q_en is not None and q_st is not None) else None
                pid_nm = (100.0 * (1.0 - NM / aln_len_q)) if (NM is not None and aln_len_q and aln_len_q > 0) else None

                out.append({
                    "query_id": qid,
                    "target_id": tid,
                    "r_st": getattr(h, "r_st", None),
                    "r_en": getattr(h, "r_en", None),
                    "q_st": q_st,
                    "q_en": q_en,
                    "strand": getattr(h, "strand", 1),
                    "mapq": getattr(h, "mapq", None),
                    "cigar": getattr(h, "cigar_str", getattr(h, "cigar", None)),
                    "is_primary": getattr(h, "is_primary", None),
                    "ctg_len": getattr(h, "ctg_len", None),
                    "qry_len": getattr(h, "qry_len", None),
                    "blen": blen,
                    "mlen": mlen,
                    "NM": NM,
                    "pid": pid_mlen_blen if pid_mlen_blen is not None else pid_nm,
                })
        return out
    finally:
        idx.close()

def _skani_like_from_mappy(hits_df: pd.DataFrame, fasta_path: str,
                           round_digits: int = 3,
                           overlap_on: str = "query",
                           overlap_tol: int = 0,
                           require_primary: bool = True,
                           return_percent: bool = True) -> pd.DataFrame:
    """
    Columns out: Ref_name, Query_name, ANI, Align_fraction_ref, Align_fraction_query
    ANI = 100 * sum(mlen) / sum(blen) over non-overlapping HSP subset.
    Fractions in percent when return_percent=True.
    """
    if hits_df.empty:
        return pd.DataFrame(columns=["Ref_name","Query_name","ANI",
                                     "Align_fraction_ref","Align_fraction_query"])

    df = hits_df.copy()
    need = ["query_id","target_id","q_st","q_en","r_st","r_en",
            "mlen","blen","mapq","is_primary","NM"]
    for c in need:
        if c not in df.columns:
            df[c] = None
    for c in ["q_st","q_en","r_st","r_en","mlen","blen","mapq","NM"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if require_primary and "is_primary" in df.columns:
        df = df[df["is_primary"].fillna(True)]

    lengths = _fasta_lengths(fasta_path)

    # helpers
    def _norm_iv(a, b):
        a = int(a); b = int(b)
        return (a, b) if a <= b else (b, a)
    def _overlap_len(iv1, iv2):
        a1,a2 = iv1; b1,b2 = iv2
        return max(0, min(a2,b2) - max(a1,b1))
    def _merge_coverage_len(intervals):
        if not intervals: return 0
        ints = [_norm_iv(a,b) for a,b in intervals if a is not None and b is not None]
        if not ints: return 0
        ints.sort()
        total = 0; s,e = ints[0]
        for x,y in ints[1:]:
            if x <= e: e = max(e,y)
            else:
                total += (e - s)
                s,e = x,y
        total += (e - s)
        return total
    def _select_non_overlapping(hits, on="query", tol=0):
        key_iv = ("q_st","q_en") if on == "query" else ("r_st","r_en")
        order = sorted(range(len(hits)), key=lambda i: (
            -(hits[i].get("mapq") or 0),
            -(hits[i].get("mlen") or 0),
            -(hits[i].get("blen") or 0)
        ))
        selected, ivs = [], []
        for i in order:
            a = hits[i].get(key_iv[0]); b = hits[i].get(key_iv[1])
            if a is None or b is None: continue
            iv = _norm_iv(a,b)
            if all(_overlap_len(iv, siv) <= tol for siv in ivs):
                selected.append(i); ivs.append(iv)
        return selected

    out = []
    for (q_id, t_id), g in df.groupby(["query_id","target_id"], sort=False):
        hits = g.to_dict("records")
        sel_idx = _select_non_overlapping(hits, on=overlap_on, tol=overlap_tol)
        sel = [hits[i] for i in sel_idx]

        sum_blen, sum_mlen = 0, 0
        for h in sel:
            bl = h.get("blen")
            if pd.isna(bl) or bl is None:
                qlen = (h.get("q_en") - h.get("q_st")) if (h.get("q_en") is not None and h.get("q_st") is not None) else None
                rlen = (h.get("r_en") - h.get("r_st")) if (h.get("r_en") is not None and h.get("r_st") is not None) else None
                bl = qlen if (qlen is not None) else rlen
            ml = h.get("mlen")
            if (pd.isna(ml) or ml is None) and bl is not None and h.get("NM") is not None:
                ml = max(0, int(bl) - int(h.get("NM")))
            if bl is not None: sum_blen += int(bl)
            if ml is not None: sum_mlen += int(ml)

        ani = (100.0 * sum_mlen / sum_blen) if sum_blen > 0 else float("nan")

        q_ivs = [_norm_iv(h["q_st"], h["q_en"]) for h in sel if pd.notna(h.get("q_st")) and pd.notna(h.get("q_en"))]
        t_ivs = [_norm_iv(h["r_st"], h["r_en"]) for h in sel if pd.notna(h.get("r_st")) and pd.notna(h.get("r_en"))]
        covered_q = _merge_coverage_len(q_ivs)
        covered_t = _merge_coverage_len(t_ivs)

        q_len = lengths.get(q_id); t_len = lengths.get(t_id)
        frac_q = (covered_q / q_len) if q_len else float("nan")
        frac_t = (covered_t / t_len) if t_len else float("nan")
        if return_percent:
            frac_q *= 100.0; frac_t *= 100.0

        out.append({
            "Ref_name": t_id,
            "Query_name": q_id,
            "ANI": None if math.isnan(ani) else round(ani, round_digits),
            "Align_fraction_ref": None if math.isnan(frac_t) else round(frac_t, round_digits),
            "Align_fraction_query": None if math.isnan(frac_q) else round(frac_q, round_digits),
        })

    return pd.DataFrame(out, columns=["Ref_name","Query_name","ANI",
                                      "Align_fraction_ref","Align_fraction_query"])


def run_pairwise_nt(all_neigh: Optional[pd.DataFrame],
                all_gff: Optional[object] = None,
                output_dir: str = "pairwise_nt_out",
                nt_aln_mode: Optional[object] = None,
                ani_mode: Optional[str] = None,
                nt_links: bool = True,
                threads: Optional[int] = None,
                blast_task: str = "blastn",
                evalue: float = 1e-5,
                perc_identity: int = 0,
                word_size: Optional[int] = None,
                soft_masking: str = "false",
                dust: str = "no",
                overlap_on: str = "query",
                overlap_tol: int = 0,
                write_outputs: bool = True,
                mm2_preset: str = "asm20",
                mm2_min_mapq: int = 0,
                mm2_threads_per_worker: int = 1,
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run pairwise nucleotide comparisons from a single FASTA and return (skani_like_df, alignment_rows_df).

    alignment_rows_df columns: [query, query_start, query_end, ref, ref_start, ref_end, ani]
    ani for blastn = pident; for fastani this will be filled from ANI table; for mappy we use pid.
    """
    def _map_and_write_pairwise_ani_uid(df: pd.DataFrame, mode_suffix: Optional[str] = None):
        if df is None or df.empty:
            return None
        if all_neigh is None or all_neigh.empty:
            return None

        # build maps: temp_seqid -> unique_id, and seqid -> unique_id only when unambiguous
        temp_map = {}
        if 'temp_seqid' in all_neigh.columns and 'unique_id' in all_neigh.columns:
            temp_map = dict(zip(all_neigh['temp_seqid'].astype(str), all_neigh['unique_id']))

        seq_map = {}
        if 'seqid' in all_neigh.columns and 'unique_id' in all_neigh.columns:
            seq_counts = all_neigh.groupby('seqid')['unique_id'].nunique()
            good_seqids = set(seq_counts[seq_counts == 1].index)
            if good_seqids:
                seq_map = dict(all_neigh[all_neigh['seqid'].isin(good_seqids)].set_index('seqid')['unique_id'])

        def _map_name_to_uid(name):
            if pd.isna(name):
                return None
            s = str(name)
            if s in temp_map:
                return temp_map[s]
            if s in seq_map:
                return seq_map[s]
            return None

        dfc = df.copy()
        # ensure expected columns exist
        if 'Ref_name' not in dfc.columns or 'Query_name' not in dfc.columns:
            return None

        dfc['ref_uid'] = dfc['Ref_name'].map(_map_name_to_uid)
        dfc['qry_uid'] = dfc['Query_name'].map(_map_name_to_uid)

        mapped = dfc.dropna(subset=['ref_uid', 'qry_uid']).copy()
        if mapped.empty:
            return None

        # unordered pair and aggregate
        mapped['A'] = mapped[['ref_uid', 'qry_uid']].min(axis=1)
        mapped['B'] = mapped[['ref_uid', 'qry_uid']].max(axis=1)

        agg = mapped.groupby(['A', 'B']).agg({
            'ANI': 'mean',
            'Align_fraction_ref': 'mean',
            'Align_fraction_query': 'mean'
        }).reset_index()

        # write output
        if write_outputs:
            suf = mode_suffix or str(nt_aln_mode)
            out_path = out / f'pairwise_ani_uid_{suf}.tsv'
            try:
                agg.to_csv(out_path, sep='\t', index=False)
            except Exception:
                # best-effort write; ignore failures here
                pass

        return agg
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    threads = threads or max(1, os.cpu_count() or 1)

    # ensure neighborhoods.fasta exists
    nb_dir = out / 'neighborhood'
    nb_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = nb_dir / 'neighborhoods.fasta'
    if not fasta_path.exists():
        if all_neigh is None or all_neigh.empty:
            raise FileNotFoundError(f"neighborhood FASTA not found at {fasta_path} and no `all_neigh` provided")
        from hoodini.utils.core import to_fasta
        if 'temp_seqid' in all_neigh.columns and 'sequence' in all_neigh.columns:
            df_for_fasta = all_neigh[['temp_seqid', 'sequence']].dropna().drop_duplicates(subset=['temp_seqid'])
            df_for_fasta.to_fasta('temp_seqid', 'sequence', str(fasta_path))
        elif 'seqid' in all_neigh.columns and 'sequence' in all_neigh.columns:
            df_for_fasta = all_neigh[['seqid', 'sequence']].dropna().drop_duplicates(subset=['seqid'])
            df_for_fasta.to_fasta('seqid', 'sequence', str(fasta_path))

    # prepare ids & lengths
    ids = _fasta_ids(str(fasta_path))
    id_pos = {cid: i for i, cid in enumerate(ids)}
    seq_lengths = _fasta_lengths(str(fasta_path))

    # SKANI BRANCH
    if ani_mode is not None and str(ani_mode).lower() == 'skani':
        import io
        console.log(f"Running skani triangle on {fasta_path}")
        cmd = [
            'skani', 'triangle',
            '-i', str(fasta_path),
            '-E', '--small-genomes',
            '-t', str(threads)
        ]
        console.print(f"Running: {' '.join(cmd)}")
        skani_log = out / 'skani_triangle.log'
        with open(skani_log, 'w') as logfh:
            subprocess.run(cmd, check=True, stdout=logfh, stderr=subprocess.DEVNULL, text=True)

        try:
            df = pd.read_csv(skani_log, sep='\t', header=0)
        except Exception:
            empty_align = pd.DataFrame(columns=["query","query_start","query_end","ref","ref_start","ref_end","ani"])
            return pd.DataFrame(columns=['Ref_name', 'Query_name', 'ANI', 'Align_fraction_ref', 'Align_fraction_query']), empty_align

        # Expected columns: Ref_name, Query_name, ANI, Align_fraction_ref, Align_fraction_query
        # If Align_fraction_query missing, copy from Align_fraction_ref
        if 'Align_fraction_query' not in df.columns and 'Align_fraction_ref' in df.columns:
            df['Align_fraction_query'] = df['Align_fraction_ref']
        # Normalize Ref/Query names
        if 'Ref_name' in df.columns:
            df['Ref_name'] = df['Ref_name'].apply(lambda x: str(x).split('/')[-1])
        if 'Query_name' in df.columns:
            df['Query_name'] = df['Query_name'].apply(lambda x: str(x).split('/')[-1])
        # Ensure numeric conversions
        if 'ANI' in df.columns:
            df['ANI'] = pd.to_numeric(df['ANI'], errors='coerce')
        if 'Align_fraction_ref' in df.columns:
            df['Align_fraction_ref'] = pd.to_numeric(df.get('Align_fraction_ref'), errors='coerce')
        if 'Align_fraction_query' in df.columns:
            df['Align_fraction_query'] = pd.to_numeric(df.get('Align_fraction_query'), errors='coerce')
        # Remove self-hits
        if 'Ref_name' in df.columns and 'Query_name' in df.columns:
            df = df[df['Ref_name'] != df['Query_name']]
        # Map to unique_id pairs using all_neigh
        required = ['Ref_name', 'Query_name', 'ANI', 'Align_fraction_ref', 'Align_fraction_query']
        pairwise_ani = df[required].copy() if all(c in df.columns for c in required) else pd.DataFrame(columns=required)
        if pairwise_ani is not None and not pairwise_ani.empty and all_neigh is not None and not all_neigh.empty:
            temp_map = dict(zip(all_neigh['temp_seqid'].astype(str), all_neigh['unique_id'])) if 'temp_seqid' in all_neigh.columns and 'unique_id' in all_neigh.columns else {}
            seq_counts = all_neigh.groupby('seqid')['unique_id'].nunique() if 'seqid' in all_neigh.columns and 'unique_id' in all_neigh.columns else pd.Series()
            good_seqids = set(seq_counts[seq_counts == 1].index) if not seq_counts.empty else set()
            seq_map = dict(all_neigh[all_neigh['seqid'].isin(good_seqids)].set_index('seqid')['unique_id']) if good_seqids else {}
            def map_name_to_uid(name):
                if pd.isna(name): return None
                s = str(name)
                if s in temp_map: return temp_map[s]
                if s in seq_map: return seq_map[s]
                return None
            pairwise_ani['ref_uid'] = pairwise_ani['Ref_name'].map(map_name_to_uid)
            pairwise_ani['qry_uid'] = pairwise_ani['Query_name'].map(map_name_to_uid)
            mapped = pairwise_ani.dropna(subset=['ref_uid', 'qry_uid']).copy()
            if not mapped.empty:
                mapped['A'] = mapped[['ref_uid', 'qry_uid']].min(axis=1)
                mapped['B'] = mapped[['ref_uid', 'qry_uid']].max(axis=1)
                pairwise_ani_uid = mapped.groupby(['A', 'B']).agg({
                    'ANI': 'mean',
                    'Align_fraction_ref': 'mean',
                    'Align_fraction_query': 'mean'
                }).reset_index()
                pairwise_ani = pairwise_ani_uid

        # attempt UID mapping and writing
        try:
            agg_uid = _map_and_write_pairwise_ani_uid(pairwise_ani, mode_suffix='skani')
            if agg_uid is not None:
                pairwise_ani = agg_uid
        except Exception:
            pass

        # skani does not produce alignment rows; return empty nt_links regardless of nt_links
        empty_align = pd.DataFrame(columns=["query","query_start","query_end","ref","ref_start","ref_end","ani"])
        return pairwise_ani, empty_align
    
    # BLASTN BRANCH
    if nt_aln_mode.lower() == 'blastn' or  str(ani_mode).lower() == 'blastn':
        db_prefix = out / 'db'
        blast_tsv = out / 'allvsall.outfmt6.tsv'

        console.log(f"Making BLAST DB at {db_prefix}")
        subprocess.run([
            "makeblastdb",
            "-in", fasta_path,
            "-dbtype", "nucl",
            "-parse_seqids",
            "-out", str(db_prefix)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen"
        cmd = [
            "blastn",
            "-query", str(fasta_path),
            "-db", str(db_prefix),
            "-task", blast_task,
            "-evalue", str(evalue),
            "-soft_masking", soft_masking,
            "-dust", dust,
            "-outfmt", outfmt,
            "-num_threads", str(threads),
        ]
        if perc_identity and perc_identity > 0:
            cmd += ["-perc_identity", str(perc_identity)]
        if word_size:
            cmd += ["-word_size", str(word_size)]

        console.log("Running BLAST: " + " ".join(cmd))
        with open(blast_tsv, "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.PIPE)

        blast_cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                      "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen"]
        raw = pd.read_csv(blast_tsv, sep="\t", names=blast_cols, dtype={"qseqid": str, "sseqid": str}, low_memory=False)

        raw = raw[raw['qseqid'] != raw['sseqid']].copy()
        raw['q_idx'] = raw['qseqid'].map(id_pos)
        raw['s_idx'] = raw['sseqid'].map(id_pos)
        raw = raw[raw['q_idx'] < raw['s_idx']].copy()

        hits_blast_df = raw[blast_cols].reset_index(drop=True)
        hits_blast_df = hits_blast_df.merge(all_neigh[["temp_seqid","seqid"]], left_on="qseqid", right_on="temp_seqid", how="left")

        console.log(f"Raw BLAST HSPs (unique unordered, no self): {len(hits_blast_df)}")

        if write_outputs:
            hits_blast_df.to_csv(out / 'pairwise_hits_blast.tsv', sep='\t', index=False)

        skani_df = _skani_like_from_blast(hits_blast_df, seq_lengths, round_digits=3,
                                          overlap_on=overlap_on, overlap_tol=overlap_tol)
        if write_outputs:
            skani_df.to_csv(out / 'skani_like_blast.tsv', sep='\t', index=False)

        # If the caller did not request nucleotide alignment rows (nt_links off)
        # then return the skani summary and an empty alignment rows DataFrame.
        if not nt_links:
            empty_align = pd.DataFrame(columns=["query","query_start","query_end","ref","ref_start","ref_end","ani"])
            # still attempt UID mapping and file write
            try:
                agg_uid = _map_and_write_pairwise_ani_uid(skani_df, mode_suffix='blastn')
                if agg_uid is not None:
                    skani_df = agg_uid
            except Exception:
                pass
            return skani_df, empty_align

        align_rows = hits_blast_df.rename(columns={
            'qseqid': 'query', 'qstart': 'query_start', 'qend': 'query_end',
            'sseqid': 'ref', 'sstart': 'ref_start', 'send': 'ref_end', 'pident': 'ani'
        })[['query', 'query_start', 'query_end', 'ref', 'ref_start', 'ref_end', 'ani']].copy()

        # Normalize IDs and apply start offsets using `all_neigh` mapping (make outputs use canonical seqid)
        if all_neigh is not None and not all_neigh.empty:
            start_map, id_map = {}, {}
            for _, nr in all_neigh.iterrows():
                temp = str(nr['temp_seqid']) if 'temp_seqid' in nr and pd.notna(nr['temp_seqid']) else None
                seqid = str(nr['seqid']) if 'seqid' in nr and pd.notna(nr['seqid']) else temp
                start_win = int(nr['start_win']) if 'start_win' in nr and pd.notna(nr['start_win']) else 0
                for key in set([temp, seqid]):
                    if not key:
                        continue
                    start_map[key] = start_win
                    start_map[Path(key).name] = start_win
                    start_map[Path(key).stem] = start_win
                    id_map[key] = seqid
                    id_map[Path(key).name] = seqid
                    id_map[Path(key).stem] = seqid

            def _find_offset(name: str) -> int:
                if pd.isna(name):
                    return 0
                name = str(name)
                canonical = id_map.get(name) or id_map.get(Path(name).name) or id_map.get(Path(name).stem) or name
                return (start_map.get(canonical)
                        or start_map.get(Path(canonical).name)
                        or start_map.get(Path(canonical).stem)
                        or 0)

            def _map_to_seqid(name: str) -> str:
                if pd.isna(name):
                    return name
                name = str(name)
                return id_map.get(name) or id_map.get(Path(name).name) or id_map.get(Path(name).stem) or name

            # adjust HSP coordinates to global coords and map ids
            align_rows["q_offset"] = align_rows["query"].apply(_find_offset)
            align_rows["r_offset"] = align_rows["ref"].apply(_find_offset)

            align_rows["query_start"] = pd.to_numeric(align_rows["query_start"], errors="coerce") + align_rows["q_offset"]
            align_rows["query_end"]   = pd.to_numeric(align_rows["query_end"], errors="coerce") + align_rows["q_offset"]
            align_rows["ref_start"]   = pd.to_numeric(align_rows["ref_start"], errors="coerce") + align_rows["r_offset"]
            align_rows["ref_end"]     = pd.to_numeric(align_rows["ref_end"], errors="coerce") + align_rows["r_offset"]

            align_rows["query"] = align_rows["query"].apply(_map_to_seqid)
            align_rows["ref"]   = align_rows["ref"].apply(_map_to_seqid)

            align_rows = align_rows.drop(columns=["q_offset", "r_offset"])

            # map skani summary pairs from temp ids -> canonical seqid
            if not skani_df.empty:
                skani_df = skani_df.copy()
                skani_df['Ref_name'] = skani_df['Ref_name'].apply(_map_to_seqid)
                skani_df['Query_name'] = skani_df['Query_name'].apply(_map_to_seqid)

            # attempt to map skani summary to neighborhood unique_id and write aggregated UID table
            try:
                agg_uid = _map_and_write_pairwise_ani_uid(skani_df, mode_suffix='blastn')
                if agg_uid is not None:
                    skani_df = agg_uid
            except Exception:
                pass

        return skani_df, align_rows
    
    elif nt_aln_mode.lower() in ('intergenic_blast', 'intergenic_blastn'):

        # write intergenic fasta using provided GFF
        inter_fa = out / 'intergenic.fasta'
        console.log(f"Writing intergenic FASTA to {inter_fa}")
        meta_df = _write_intergenic_fasta(all_gff, str(fasta_path), str(inter_fa), all_neigh=all_neigh)
        if meta_df.empty:
            console.log("No intergenic sequences produced; exiting")
            return pd.DataFrame(columns=["Ref_name","Query_name","ANI","Align_fraction_ref","Align_fraction_query"]), pd.DataFrame()

        # make BLAST DB and run blastn-short
        db_prefix = out / 'intergenic_db'
        console.log(f"Making BLAST DB for intergenic sequences at {db_prefix}")
        subprocess.run([
            "makeblastdb",
            "-in", str(inter_fa),
            "-dbtype", "nucl",
            "-out", str(db_prefix)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        blast_tsv = out / 'intergenic_allvsall.outfmt6.tsv'
        outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen"
        cmd = [
            "blastn",
            "-query", str(inter_fa),
            "-db", str(db_prefix),
            "-task", "blastn-short",              # change from 'blastn-short' to 'blastn'
            "-evalue", "1e-3",              # more stringent, but still permissive
            "-word_size", "4",              # minimum allowed, most sensitive
            "-reward", "1",
            "-penalty", "-1",
            "-gapopen", "2",
            "-gapextend", "2",
            "-dust", "no",
            "-soft_masking", "false",
            "-outfmt", outfmt,
            "-num_threads", str(threads),
        ]
        console.log("Running intergenic BLAST: " + " ".join(cmd))
        with open(blast_tsv, "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.PIPE)

        blast_cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                      "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen"]
        raw = pd.read_csv(blast_tsv, sep="\t", names=blast_cols, dtype={"qseqid": str, "sseqid": str}, low_memory=False)

        # merge with meta to map temp ids -> canonical seqid and start offsets
        meta = meta_df.copy()
        meta = meta.rename(columns={'temp_seqid': 'temp_id', 'seqid': 'canonical_seqid', 'start_win': 'start_win'})
        raw = raw.merge(meta[['temp_id', 'canonical_seqid', 'start_win']], left_on='qseqid', right_on='temp_id', how='left')
        raw = raw.rename(columns={'canonical_seqid': 'q_canonical', 'start_win': 'q_start_win'})
        raw = raw.drop(columns=['temp_id'])
        raw = raw.merge(meta[['temp_id', 'canonical_seqid', 'start_win']], left_on='sseqid', right_on='temp_id', how='left')
        raw = raw.rename(columns={'canonical_seqid': 's_canonical', 'start_win': 's_start_win'})
        raw = raw.drop(columns=['temp_id'])

        # drop self hits mapped to same canonical seqid
        raw = raw[(raw['q_canonical'].notna()) & (raw['s_canonical'].notna())]
        raw = raw[raw['q_canonical'] != raw['s_canonical']]

        # compute global coordinates for HSPs
        raw['qstart_g'] = pd.to_numeric(raw['qstart'], errors='coerce') + pd.to_numeric(raw['q_start_win'], errors='coerce') - 1
        raw['qend_g'] = pd.to_numeric(raw['qend'], errors='coerce') + pd.to_numeric(raw['q_start_win'], errors='coerce') - 1
        raw['sstart_g'] = pd.to_numeric(raw['sstart'], errors='coerce') + pd.to_numeric(raw['s_start_win'], errors='coerce') - 1
        raw['send_g'] = pd.to_numeric(raw['send'], errors='coerce') + pd.to_numeric(raw['s_start_win'], errors='coerce') - 1

        # normalize ordering so q < s lexicographically to get unordered pair uniqueness
        def _orient_row(r):
            if r['q_canonical'] > r['s_canonical']:
                # swap fields
                return pd.Series({
                    'qseqid': r['s_canonical'], 'sseqid': r['q_canonical'],
                    'pident': r['pident'], 'length': r['length'],
                    'qstart': r['sstart_g'], 'qend': r['send_g'],
                    'sstart': r['qstart_g'], 'send': r['qend_g'],
                    'qlen': r.get('qlen', None), 'slen': r.get('slen', None),
                    'mismatch': r.get('mismatch', None), 'gapopen': r.get('gapopen', None),
                    'evalue': r.get('evalue', None), 'bitscore': r.get('bitscore', None)
                })
            else:
                return pd.Series({
                    'qseqid': r['q_canonical'], 'sseqid': r['s_canonical'],
                    'pident': r['pident'], 'length': r['length'],
                    'qstart': r['qstart_g'], 'qend': r['qend_g'],
                    'sstart': r['sstart_g'], 'send': r['send_g'],
                    'qlen': r.get('qlen', None), 'slen': r.get('slen', None),
                    'mismatch': r.get('mismatch', None), 'gapopen': r.get('gapopen', None),
                    'evalue': r.get('evalue', None), 'bitscore': r.get('bitscore', None)
                })

        norm = raw.apply(_orient_row, axis=1)
        hits_blast_df = norm.reset_index(drop=True)

        console.log(f"Intergenic BLAST HSPs (mapped to seqid): {len(hits_blast_df)}")
        if write_outputs:
            hits_blast_df.to_csv(out / 'pairwise_hits_intergenic_blast.tsv', sep='\t', index=False)

        # build seq_lengths map from all_neigh if available
        seq_lengths_map = {}
        if all_neigh is not None and not all_neigh.empty and 'seqid' in all_neigh.columns and 'sequence' in all_neigh.columns:
            for _, nr in all_neigh.iterrows():
                sid = str(nr['seqid']) if pd.notna(nr.get('seqid')) else None
                if not sid:
                    continue
                seq = nr.get('sequence') if 'sequence' in nr else None
                if pd.notna(seq) and seq is not None:
                    seq_lengths_map[sid] = len(seq)

        # compute skani-like summary
        skani_df = _skani_like_from_blast(hits_blast_df, seq_lengths_map or seq_lengths, round_digits=3,
                                          overlap_on=overlap_on, overlap_tol=overlap_tol)
        if write_outputs:
            skani_df.to_csv(out / 'skani_like_intergenic_blast.tsv', sep='\t', index=False)

        align_rows = hits_blast_df.rename(columns={
            'qseqid': 'query', 'qstart': 'query_start', 'qend': 'query_end',
            'sseqid': 'ref', 'sstart': 'ref_start', 'send': 'ref_end', 'pident': 'ani'
        })[['query', 'query_start', 'query_end', 'ref', 'ref_start', 'ref_end', 'ani']].copy()

        return skani_df, align_rows
    
    elif nt_aln_mode.lower() == 'fastani':
        split_dir = out / 'ani_split'
        split_dir.mkdir(parents=True, exist_ok=True)
        try:
            from Bio import SeqIO
        except Exception as exc:
            raise ImportError('Biopython is required for fastANI mode (SeqIO)') from exc

        genome_files = []
        file_to_seqid = {}
        temp_to_seqid = {}
        start_map_init = {}
        if all_neigh is not None and not all_neigh.empty:
            for _, nr in all_neigh.iterrows():
                temp = str(nr['temp_seqid']) if 'temp_seqid' in nr and pd.notna(nr['temp_seqid']) else None
                seqid = str(nr['seqid']) if 'seqid' in nr and pd.notna(nr['seqid']) else temp
                if temp:
                    temp_to_seqid[temp] = seqid
                if seqid and 'start_win' in nr and pd.notna(nr['start_win']):
                    start_map_init[seqid] = int(nr['start_win'])

        for idx, record in enumerate(SeqIO.parse(str(fasta_path), 'fasta')):
            header = record.id
            filename = "".join(c if c.isalnum() or c in '-._' else '_' for c in header) + f'_idx{idx}.fasta'
            out_path = split_dir / filename
            SeqIO.write(record, str(out_path), 'fasta')
            genome_files.append(str(out_path))
            canonical = temp_to_seqid.get(header, header)
            file_to_seqid[out_path.name] = canonical
            file_to_seqid[out_path.stem] = canonical
            san_hdr = "".join(c if c.isalnum() or c in '-._' else '_' for c in header)
            file_to_seqid[san_hdr] = canonical

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
            q = row['query']; r = row['reference']
            ani = row['ani']
            frags_matched = row['frags_matched']
            frags_total_query = row['frags_total_query']
            frags_total_ref = reverse_frag_total.get((q, r), None)

            frac_q = (frags_matched / frags_total_query) if frags_total_query else 0
            frac_r = (frags_matched / frags_total_ref) if frags_total_ref else frac_q

            rows.append({
                'Ref_name': Path(str(r)).name,
                'Query_name': Path(str(q)).name,
                'ANI': float(ani),
                'Align_fraction_ref': float(frac_r) * 100.0,
                'Align_fraction_query': float(frac_q) * 100.0
            })

        skani_like_df = pd.DataFrame(rows)
        skani_like_df = skani_like_df[skani_like_df['Ref_name'] != skani_like_df['Query_name']]

        def _pair_key(row):
            a, b = row['Ref_name'], row['Query_name']
            return tuple(sorted((a, b)))

        skani_like_df['pair_key'] = skani_like_df.apply(_pair_key, axis=1)
        agg = skani_like_df.groupby('pair_key').agg({
            'ANI': 'mean',
            'Align_fraction_ref': 'mean',
            'Align_fraction_query': 'mean'
        }).reset_index()
        agg['Ref_name'] = agg['pair_key'].apply(lambda t: t[0])
        agg['Query_name'] = agg['pair_key'].apply(lambda t: t[1])
        pairwise_ani = agg[['Ref_name', 'Query_name', 'ANI', 'Align_fraction_ref', 'Align_fraction_query']]

        if write_outputs:
            pairwise_ani.to_csv(out / 'pairwise_ani_fastani.tsv', sep='\t', index=False)

        # attempt to map fastANI results to neighborhood unique_id and write aggregated UID table
        try:
            agg_uid = _map_and_write_pairwise_ani_uid(pairwise_ani, mode_suffix='fastani')
            if agg_uid is not None:
                pairwise_ani = agg_uid
        except Exception:
            pass

        # If caller did not request nucleotide alignment rows (nt_links off),
        # return empty visual/alignments DataFrame
        if not nt_links:
            empty_vis = pd.DataFrame(columns=["query","ref","ani","query_start","query_end","ref_start","ref_end"])
            return pairwise_ani, empty_vis

        # parse .visual files per unordered pair
        work_dir = out / 'fastani_pairwise_visual'
        work_dir.mkdir(parents=True, exist_ok=True)

        def _resolve_path(name: str) -> Path:
            p = Path(name)
            if p.exists():
                return p
            candidates = [
                out / 'ani_split' / name,
                out / 'ani_split' / (name + '.fasta'),
                out / 'ani_split' / (name + '.fa'),
                out / 'neighborhood' / name,
                out / 'neighborhood' / (name + '.fasta'),
            ]
            for c in candidates:
                if c.exists():
                    return c
            raise FileNotFoundError(f"Could not resolve path for genome identifier '{name}'")

        dfp = pairwise_ani.copy()
        if 'Align_fraction_ref' in dfp.columns or 'Align_fraction_query' in dfp.columns:
            dfp = dfp.dropna(subset=[c for c in ['Align_fraction_ref', 'Align_fraction_query'] if c in dfp.columns])

        visual_rows = []
        seen_pairs = set()
        from rich.progress import Progress
        progress = Progress()
        task = progress.add_task("[cyan]Running pairwise FastANI visualize...", total=len(dfp))
        progress.start()

        for row in dfp.itertuples(index=False):
            q_name = getattr(row, 'Query_name'); r_name = getattr(row, 'Ref_name')
            pair_key = tuple(sorted([str(q_name), str(r_name)]))
            if pair_key in seen_pairs:
                progress.advance(task); continue
            seen_pairs.add(pair_key)

            raw0, raw1 = str(pair_key[0]), str(pair_key[1])
            out_file = work_dir / f"{raw0}__vs__{raw1}.visual"
            if not out_file.exists():
                q_path = _resolve_path(pair_key[0])
                r_path = _resolve_path(pair_key[1])
                temp_base = work_dir / f"{raw0}__vs__{raw1}.fastani"
                cmd = [
                    'fastANI',
                    '-q', str(q_path),
                    '-r', str(r_path),
                    '--visualize',
                    '-o', str(temp_base),
                    '-t', str(threads)
                ]
                temp_log = Path(str(temp_base) + '.fastani.log')
                try:
                    with open(temp_log, 'w') as logfh:
                        subprocess.run(cmd, check=True, stdout=logfh, stderr=logfh, text=True)
                except subprocess.CalledProcessError:
                    continue
                visual_file = Path(str(temp_base) + '.visual')
                if visual_file.exists():
                    visual_file.rename(out_file)

            if out_file.exists():
                parsed = pd.read_csv(
                    out_file, sep='\t', header=None,
                    names=['query','ref','ani','na1','na2','na3',
                           'query_start','query_end','ref_start','ref_end','na4','na5'],
                    dtype={'query': str, 'ref': str}
                )
                parsed = parsed[["query","ref","ani","query_start","query_end","ref_start","ref_end"]]
                parsed['query'] = parsed['query'].apply(lambda s: Path(str(s)).name if pd.notna(s) else s)
                parsed['ref'] = parsed['ref'].apply(lambda s: Path(str(s)).name if pd.notna(s) else s)
                visual_rows.append(parsed)
            progress.advance(task)

        progress.stop()
        visual_df = pd.concat(visual_rows, ignore_index=True) if visual_rows else pd.DataFrame(
            columns=["query","ref","ani","query_start","query_end","ref_start","ref_end"])

        if 'ani' in visual_df.columns:
            visual_df['ani'] = pd.to_numeric(visual_df['ani'], errors='coerce')

        # Map filenames/backed ids to canonical seqid and apply start offsets if all_neigh provided
        if all_neigh is not None and not all_neigh.empty:
            # build id_map/start_map from temp_to_seqid and start_map_init
            id_map = {}
            start_map = {}
            # include mappings we already have for split files
            for k, v in file_to_seqid.items():
                id_map[k] = v
                id_map[Path(k).name] = v
                id_map[Path(k).stem] = v
            for seqid, st in start_map_init.items():
                start_map[seqid] = st
                start_map[Path(seqid).name] = st
                start_map[Path(seqid).stem] = st

            # also include mappings from all_neigh rows (temp_seqid -> seqid)
            for _, nr in all_neigh.iterrows():
                temp = str(nr['temp_seqid']) if 'temp_seqid' in nr and pd.notna(nr['temp_seqid']) else None
                seqid = str(nr['seqid']) if 'seqid' in nr and pd.notna(nr['seqid']) else temp
                start_win = int(nr['start_win']) if 'start_win' in nr and pd.notna(nr['start_win']) else 0
                for key in set([temp, seqid]):
                    if not key:
                        continue
                    id_map[key] = seqid
                    id_map[Path(key).name] = seqid
                    id_map[Path(key).stem] = seqid
                    start_map[key] = start_win
                    start_map[Path(key).name] = start_win
                    start_map[Path(key).stem] = start_win

            def _map_to_seqid_fast(name: str) -> str:
                if pd.isna(name):
                    return name
                name = str(name)
                return id_map.get(name) or id_map.get(Path(name).name) or id_map.get(Path(name).stem) or name

            def _find_offset_fast(name: str) -> int:
                if pd.isna(name):
                    return 0
                name = str(name)
                canonical = _map_to_seqid_fast(name)
                return (start_map.get(canonical)
                        or start_map.get(Path(canonical).name)
                        or start_map.get(Path(canonical).stem)
                        or 0)

            # map pairwise_ani names
            pairwise_ani = pairwise_ani.copy()
            pairwise_ani['Ref_name'] = pairwise_ani['Ref_name'].apply(_map_to_seqid_fast)
            pairwise_ani['Query_name'] = pairwise_ani['Query_name'].apply(_map_to_seqid_fast)

            # apply offsets to visual_df coordinates and map ids
            if not visual_df.empty:
                visual_df = visual_df.copy()
                visual_df['q_off'] = visual_df['query'].apply(_find_offset_fast)
                visual_df['r_off'] = visual_df['ref'].apply(_find_offset_fast)
                visual_df['query_start'] = pd.to_numeric(visual_df['query_start'], errors='coerce') + visual_df['q_off']
                visual_df['query_end'] = pd.to_numeric(visual_df['query_end'], errors='coerce') + visual_df['q_off']
                visual_df['ref_start'] = pd.to_numeric(visual_df['ref_start'], errors='coerce') + visual_df['r_off']
                visual_df['ref_end'] = pd.to_numeric(visual_df['ref_end'], errors='coerce') + visual_df['r_off']
                visual_df['query'] = visual_df['query'].apply(_map_to_seqid_fast)
                visual_df['ref'] = visual_df['ref'].apply(_map_to_seqid_fast)
                visual_df = visual_df.drop(columns=['q_off', 'r_off'])

        return pairwise_ani, visual_df
    elif nt_aln_mode.lower() in ("minimap2", "mappy"):

        plans = [(str(fasta_path), mm2_preset, mm2_min_mapq, mm2_threads_per_worker,
                  tid, ids[:id_pos[tid]]) for tid in ids]

        rows = []
        from rich.progress import Progress
        with Progress() as progress:
            task = progress.add_task("[cyan]Running minimap2/mappy...", total=len(plans))
            with ProcessPoolExecutor(max_workers=max(1, min(threads, len(plans)))) as ex:
                futs = [ex.submit(_run_mappy_target_block, p) for p in plans]
                for fut in as_completed(futs):
                    rows.extend(fut.result())
                    progress.advance(task)

        hits_df = pd.DataFrame.from_records(rows)
        if write_outputs:
            hits_df.to_csv(out / "pairwise_hits_mappy.tsv", sep="\t", index=False)

        # skani-like summary (ANI + coverage)
        skani_df = _skani_like_from_mappy(
            hits_df, str(fasta_path),
            round_digits=3,
            overlap_on=overlap_on,
            overlap_tol=overlap_tol,
            require_primary=True,
            return_percent=True
        )
        if write_outputs:
            skani_df.to_csv(out / "skani_like_mappy.tsv", sep="\t", index=False)

        # attempt to map mappy skani summary to neighborhood unique_id and write aggregated UID table
        try:
            agg_uid = _map_and_write_pairwise_ani_uid(skani_df, mode_suffix='mappy')
            if agg_uid is not None:
                skani_df = agg_uid
        except Exception:
            pass

        # Alignment rows table (HSP-level)
        align_rows = hits_df.rename(columns={
            "query_id": "query",
            "q_st": "query_start",
            "q_en": "query_end",
            "target_id": "ref",
            "r_st": "ref_start",
            "r_en": "ref_end",
            "pid": "ani"
        })[['query', 'query_start', 'query_end',
            'ref', 'ref_start', 'ref_end', 'ani']].copy()

        # 🔥 Normalize IDs and add offsets like in BLAST/FastANI
        if all_neigh is not None and not all_neigh.empty:
            start_map, id_map = {}, {}
            for _, nr in all_neigh.iterrows():
                temp = str(nr['temp_seqid']) if 'temp_seqid' in nr and pd.notna(nr['temp_seqid']) else None
                seqid = str(nr['seqid']) if 'seqid' in nr and pd.notna(nr['seqid']) else temp
                start_win = int(nr['start_win']) if 'start_win' in nr and pd.notna(nr['start_win']) else 0
                for key in set([temp, seqid]):
                    if not key:
                        continue
                    start_map[key] = start_win
                    start_map[Path(key).name] = start_win
                    start_map[Path(key).stem] = start_win
                    id_map[key] = seqid
                    id_map[Path(key).name] = seqid
                    id_map[Path(key).stem] = seqid

            def _find_offset(name: str) -> int:
                if pd.isna(name):
                    return 0
                name = str(name)
                canonical = id_map.get(name) or id_map.get(Path(name).name) or id_map.get(Path(name).stem) or name
                return (start_map.get(canonical)
                        or start_map.get(Path(canonical).name)
                        or start_map.get(Path(canonical).stem)
                        or 0)

            def _map_to_seqid(name: str) -> str:
                if pd.isna(name):
                    return name
                name = str(name)
                return id_map.get(name) or id_map.get(Path(name).name) or id_map.get(Path(name).stem) or name

            align_rows["q_offset"] = align_rows["query"].apply(_find_offset)
            align_rows["r_offset"] = align_rows["ref"].apply(_find_offset)

            align_rows["query_start"] = pd.to_numeric(align_rows["query_start"], errors="coerce") + align_rows["q_offset"]
            align_rows["query_end"]   = pd.to_numeric(align_rows["query_end"], errors="coerce") + align_rows["q_offset"]
            align_rows["ref_start"]   = pd.to_numeric(align_rows["ref_start"], errors="coerce") + align_rows["r_offset"]
            align_rows["ref_end"]     = pd.to_numeric(align_rows["ref_end"], errors="coerce") + align_rows["r_offset"]

            align_rows["query"] = align_rows["query"].apply(_map_to_seqid)
            align_rows["ref"]   = align_rows["ref"].apply(_map_to_seqid)

            align_rows = align_rows.drop(columns=["q_offset", "r_offset"])

        return skani_df, align_rows
    else:
        raise ValueError("nt_aln_mode must be 'blastn', 'fastani', or 'mappy'")
