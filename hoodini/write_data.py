from pathlib import Path
from typing import Optional, Dict

import base64
import pandas as pd
from importlib.resources import files

def write_viz_outputs(
    *,
    output_dir: str,
    all_gff: pd.DataFrame,
    all_neigh: pd.DataFrame,
    all_prots: pd.DataFrame,
    den_data: pd.DataFrame,
    tree_str: str,
    nt_links: Optional[pd.DataFrame] = None,
    pairwise_aa: Optional[pd.DataFrame] = None,
    domains_data: Optional[pd.DataFrame] = None,
    write_domains: bool = False,
) -> Path:
    """
    Write hoodini visualization-ready files into a hoodini-viz folder.

    Returns the output directory path.
    """
    outdir = Path(output_dir) / "hoodini-viz"
    outdir.mkdir(parents=True, exist_ok=True)

    # GFF
    if all_gff is not None and not all_gff.empty:
        all_gff.to_csv(outdir / "defaultGFF.gff", index=False, sep="\t", header=False)
    else:
        # write empty placeholder
        pd.DataFrame().to_csv(outdir / "defaultGFF.gff", index=False, sep="\t", header=False)

    # Baselines
    if all_neigh is not None and not all_neigh.empty:
        neigh = all_neigh.copy()
        neigh = neigh.rename(
            columns={
                "unique_id": "hood_id",
                "start_win": "start",
                "end_win": "end",
                "target_prot": "align_gene",
            }
        )
        cols = [c for c in ["hood_id", "seqid", "start", "end", "align_gene"] if c in neigh.columns]
        pd.DataFrame(columns=["hood_id", "seqid", "start", "end", "align_gene"]) \
            if not cols else neigh[cols] \
            .to_csv(outdir / "defaultBaselines.txt", sep="\t", index=False)
    else:
        pd.DataFrame(columns=["hood_id", "seqid", "start", "end", "align_gene"]).to_csv(
            outdir / "defaultBaselines.txt", sep="\t", index=False
        )

    # Protein metadata
    if all_prots is not None and not all_prots.empty:
        prots = all_prots.copy()
        prots = prots.rename(columns={"fam_cluster": "cluster"})
        #exclude "protein_id" from the output
        prots = prots[[c for c in prots.columns if c != "protein_id"]]
        if "product" in prots.columns:
            prots["product"] = prots["product"].astype(str).str.replace("\n", " ")
        prots.to_csv(outdir / "defaultProteinMetadata.txt", sep="\t", index=False)
    else:
        pd.DataFrame(columns=["gene_id", "cluster", "product"]).to_csv(
            outdir / "defaultProteinMetadata.txt", sep="\t", index=False
        )

    # Tree metadata
    if den_data is not None and not den_data.empty:
        tree_meta = den_data.copy()
        tree_meta = tree_meta.rename(columns={"unique_id": "leaf_id"})
        if "leaf_id" in tree_meta.columns:
            # Extract numeric id if possible
            tree_meta["leaf_id"] = (
                tree_meta["leaf_id"].astype(str).str.extract(r"(\d+)")
            )
            # Keep as string if extraction fails
            try:
                tree_meta["leaf_id"] = tree_meta["leaf_id"].astype(int)
            except Exception:
                pass
        tree_meta.to_csv(outdir / "defaultTreeMetadata.txt", sep="\t", index=False)
    else:
        pd.DataFrame(columns=["leaf_id"]).to_csv(outdir / "defaultTreeMetadata.txt", sep="\t", index=False)

    # Newick tree
    (outdir / "defaultNewick.txt").write_text(tree_str or "", encoding="utf-8")

    # Domain metadata (optional)
    if write_domains and domains_data is not None and not domains_data.empty:
        # Order columns so the frontend parser (which expects token[4] to be evalue)
        # receives: protein_id, domain_id, start, end, e_value, cov, database
        cols = [c for c in ["protein_id", "domain_id", "start", "end", "database", "e_value", "cov"] if c in domains_data.columns]
        undesired_cols = ['protein_id', 'bit_score', 'alignment_length', 'e_value',
                          'start', 'end', 'cov', 'database', 'bit_score*alignment_length',
                          'domain_id_clean', 'ID']
        desired_cols = [c for c in domains_data.columns if c not in undesired_cols]

        def format_evalue(val):
            try:
                if pd.isna(val):
                    return ""
                f = float(val)
            except Exception:
                return str(val)
            # format in scientific notation with 2 decimals in mantissa, e.g. 4.24e-43
            return "{:.2e}".format(f)

        def format_float_2(val):
            try:
                if pd.isna(val):
                    return ""
                f = float(val)
            except Exception:
                return str(val)
            return "{:.2f}".format(f)

        if cols:
            df_domains = domains_data[cols].copy()
            # format e_value and cov if present
            if 'e_value' in df_domains.columns:
                df_domains['e_value'] = df_domains['e_value'].apply(format_evalue)
            if 'cov' in df_domains.columns:
                df_domains['cov'] = df_domains['cov'].apply(format_float_2)

            df_domains.to_csv(outdir / "defaultDomains.txt", sep="\t", index=False, header=False)

            # format metadata floats to two decimals (and e_value specially)
            df_meta = domains_data[desired_cols].copy()
            for c in df_meta.columns:
                if c == 'e_value':
                    df_meta[c] = df_meta[c].apply(format_evalue)
                elif pd.api.types.is_float_dtype(df_meta[c].dtype):
                    df_meta[c] = df_meta[c].apply(format_float_2)

            df_meta.to_csv(outdir / "defaultDomainsMetadata.txt", sep="\t", index=False)

    # Nucleotide links
    if nt_links is not None and isinstance(nt_links, pd.DataFrame) and not nt_links.empty:
        cols = [
            "query",
            "query_start",
            "query_end",
            "ref",
            "ref_start",
            "ref_end",
            "ani",
        ]
        present = [c for c in cols if c in nt_links.columns]
        if len(present) == len(cols):
            nt_links[cols].to_csv(outdir / "defaultNucleotideLinks.txt", sep="\t", index=False, header=False)
        else:
            pd.DataFrame(columns=cols).to_csv(outdir / "defaultNucleotideLinks.txt", sep="\t", index=False, header=False)
    else:
        pd.DataFrame(columns=["query", "query_start", "query_end", "ref", "ref_start", "ref_end", "ani"]).to_csv(
            outdir / "defaultNucleotideLinks.txt", sep="\t", index=False, header=False
        )

    # Protein links
    if pairwise_aa is not None and isinstance(pairwise_aa, pd.DataFrame) and not pairwise_aa.empty:
        cols = ["qseqid", "sseqid", "pident"]
        present = [c for c in cols if c in pairwise_aa.columns]
        if len(present) == len(cols):
            pairwise_aa[cols].to_csv(outdir / "defaultProteinLinks.txt", sep="\t", index=False, header=False)
        else:
            pd.DataFrame(columns=cols).to_csv(outdir / "defaultProteinLinks.txt", sep="\t", index=False, header=False)
    else:
        pd.DataFrame(columns=["qseqid", "sseqid", "pident"]).to_csv(
            outdir / "defaultProteinLinks.txt", sep="\t", index=False, header=False
        )


    # Prefer package resource; fall back to relative path if needed
    resource_template = files("hoodini").joinpath("template", "template.html")

    # Read template HTML
    template_html = resource_template.read_text(encoding="utf-8")


    # Prepare data for embedding
    def escape_js_string(s):
        """
        Escapes backticks, backslashes, and newlines for safe JS template literal embedding.
        """
        if not isinstance(s, str):
            return ""
        return (
            s.replace('\\', r'\\')
             .replace('`', r'\`')
             .replace('\r', '')
             .replace('\n', r'\n')
        )

    def read_file_text(path):
        try:
            raw = Path(path).read_text(encoding="utf-8")
            return escape_js_string(raw)
        except Exception:
            return ""


    viz_files = {
        "GFF_ANNOTATION_DATA": outdir / "defaultGFF.gff",
        "BASELINES_DATA": outdir / "defaultBaselines.txt",
        "PROTEIN_METADATA_DATA": outdir / "defaultProteinMetadata.txt",
        "TREE_METADATA_DATA": outdir / "defaultTreeMetadata.txt",
        "NEWICK_TREE_DATA": outdir / "defaultNewick.txt",
        "DOMAINS_DATA": outdir / "defaultDomains.txt",
        "NUCLEOTIDE_LINKS_DATA": outdir / "defaultNucleotideLinks.txt",
        "PROTEIN_LINKS_DATA": outdir / "defaultProteinLinks.txt",
        "DOMAINS_METADATA": outdir / "defaultDomainsMetadata.txt"
    }

    viz_data = {k: read_file_text(v) for k, v in viz_files.items()}

    # Render template with Jinja2
    from jinja2 import Template
    template = Template(template_html)
    rendered_html = template.render(**viz_data)


    # Write the rendered HTML to output dir
    (outdir / "hoodini-viz.html").write_text(rendered_html, encoding="utf-8")

    return outdir
