from pathlib import Path
from typing import Optional, Dict

import base64
import polars as pl
from importlib.resources import files
from hoodini.utils.polars_adapters import to_polars


def write_viz_outputs(
    *,
    output_dir: str,
    all_gff: pl.DataFrame,
    all_neigh: pl.DataFrame,
    all_prots: pl.DataFrame,
    den_data: pl.DataFrame,
    tree_str: str,
    records: Optional[pl.DataFrame] = None,
    nt_links: Optional[pl.DataFrame] = None,
    pairwise_aa: Optional[pl.DataFrame] = None,
    domains_data: Optional[pl.DataFrame] = None,
    write_domains: bool = False,
) -> Path:
    """
    Write hoodini visualization-ready files into a hoodini-viz folder.

    Returns the output directory path.
    """
    outdir = Path(output_dir) / "hoodini-viz"
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure Polars DataFrames for consistent operations
    all_gff = to_polars(all_gff) if all_gff is not None else pl.DataFrame()
    all_neigh = to_polars(all_neigh) if all_neigh is not None else pl.DataFrame()
    all_prots = to_polars(all_prots) if all_prots is not None else pl.DataFrame()
    den_data = to_polars(den_data) if den_data is not None else pl.DataFrame()
    nt_links = to_polars(nt_links) if nt_links is not None else None
    pairwise_aa = to_polars(pairwise_aa) if pairwise_aa is not None else None
    domains_data = to_polars(domains_data) if domains_data is not None else None

    # GFF
    if all_gff is not None and all_gff.height > 0:
        all_gff.write_csv(outdir / "defaultGFF.gff", include_header=False, separator="\t")
    else:
        # write empty placeholder
        pl.DataFrame().write_csv(outdir / "defaultGFF.gff", include_header=False, separator="\t")

    # Baselines
    baseline_headers = "hood_id\tseqid\tstart\tend\talign_gene\n"
    if all_neigh is not None and all_neigh.height > 0:
        neigh = all_neigh.clone()
        mapping = {
            "unique_id": "hood_id",
            "start_win": "start",
            "end_win": "end",
            "target_prot": "align_gene",
        }
        present_map = {k: v for k, v in mapping.items() if k in neigh.columns}
        if present_map:
            neigh = neigh.rename(present_map)
        cols = [c for c in ["hood_id", "seqid", "start", "end", "align_gene"] if c in neigh.columns]
        if cols:
            csv_data = neigh.select(cols).write_csv(separator="\t", include_header=False)
            (outdir / "defaultBaselines.txt").write_text(
                baseline_headers + csv_data, encoding="utf-8"
            )
        else:
            (outdir / "defaultBaselines.txt").write_text(baseline_headers, encoding="utf-8")
    else:
        (outdir / "defaultBaselines.txt").write_text(baseline_headers, encoding="utf-8")

    # Protein metadata
    # Build dynamic headers: base 7 columns + any extra columns from extra tools
    base_headers = [
        "id",
        "sequence",
        "product",
        "target_prot",
        "target_nuc",
        "unique_id",
        "cluster",
    ]
    if all_prots is not None and all_prots.height > 0:
        prots = all_prots.clone()
        if "fam_cluster" in prots.columns:
            prots = prots.rename({"fam_cluster": "cluster"})

        # Ensure all required base columns exist, filling missing ones with empty strings
        for col in ["target_prot", "target_nuc", "unique_id"]:
            if col not in prots.columns:
                prots = prots.with_columns(pl.lit("").alias(col))

        # Ensure cluster is integer (or null) - round float values
        if "cluster" not in prots.columns:
            prots = prots.with_columns(pl.lit(None).alias("cluster"))
        else:
            prots = prots.with_columns(pl.col("cluster").round().cast(pl.Int64))

        # Select columns in order: base columns first, then any additional columns from extra tools
        base_cols = [
            "id",
            "sequence",
            "product",
            "target_prot",
            "target_nuc",
            "unique_id",
            "cluster",
        ]
        base_cols_present = [c for c in base_cols if c in prots.columns]
        extra_cols = [c for c in prots.columns if c not in base_cols]
        prots = prots.select(base_cols_present + extra_cols)

        # Sanitize string columns: remove newlines, carriage returns, and dangerous chars
        # Sanitize all string-type columns to prevent JS injection/errors
        for col in prots.columns:
            col_dtype = prots.schema.get(col)
            if col_dtype == pl.Utf8 or col_dtype == pl.String:
                prots = prots.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .str.replace_all("\r\n", " ")  # Windows newlines
                    .str.replace_all("\n", " ")  # Unix newlines
                    .str.replace_all("\r", " ")  # Old Mac newlines
                    .str.replace_all("`", "'")  # Replace backticks with single quotes
                    .str.replace_all('"', "'")  # Collapse double quotes to single quotes
                    .alias(col)
                )
        csv_data = prots.write_csv(separator="\t", include_header=False)
        # Build dynamic header from all columns in the dataframe
        protein_headers = "\t".join(prots.columns) + "\n"
        (outdir / "defaultProteinMetadata.txt").write_text(
            protein_headers + csv_data, encoding="utf-8"
        )
    else:
        # If empty, write header with base columns only
        protein_headers = "\t".join(base_headers) + "\n"
        (outdir / "defaultProteinMetadata.txt").write_text(protein_headers, encoding="utf-8")

    # Tree metadata
    tree_headers = "leaf_id\tog_index\tsuperkingdom\tkingdom\tphylum\tclass\torder\tfamily\tgenus\tspecies\tstart_win\tend_win\tstrand_win\tstart_target\tend_target\n"
    if den_data is not None and den_data.height > 0:
        tree_meta = den_data.clone()
        if "unique_id" in tree_meta.columns:
            tree_meta = tree_meta.rename({"unique_id": "leaf_id"})
        csv_data = tree_meta.write_csv(separator="\t", include_header=False)
        (outdir / "defaultTreeMetadata.txt").write_text(tree_headers + csv_data, encoding="utf-8")
    else:
        (outdir / "defaultTreeMetadata.txt").write_text(tree_headers, encoding="utf-8")

    # Newick tree
    (outdir / "defaultNewick.txt").write_text(tree_str or "", encoding="utf-8")

    # Domain metadata (optional)
    if write_domains and domains_data is not None and domains_data.height > 0:
        # Order columns so the frontend parser (which expects token[4] to be evalue)
        # receives: protein_id, domain_id, start, end, e_value, cov, database
        cols = [
            c
            for c in ["protein_id", "domain_id", "start", "end", "database", "e_value", "cov"]
            if c in domains_data.columns
        ]
        undesired_cols = [
            "protein_id",
            "bit_score",
            "alignment_length",
            "e_value",
            "start",
            "end",
            "cov",
            "database",
            "bit_score*alignment_length",
            "domain_id_clean",
            "ID",
        ]
        desired_cols = [c for c in domains_data.columns if c not in undesired_cols]

        def format_evalue(val):
            try:
                if val is None:
                    return ""
                f = float(val)
            except Exception:
                return str(val)
            # format in scientific notation with 2 decimals in mantissa, e.g. 4.24e-43
            return "{:.2e}".format(f)

        def format_float_2(val):
            try:
                if val is None:
                    return ""
                f = float(val)
            except Exception:
                return str(val)
            return "{:.2f}".format(f)

        if cols:
            df_domains = domains_data.select(cols)
            # format e_value and cov if present
            if "e_value" in df_domains.columns:
                df_domains = df_domains.with_columns(
                    pl.col("e_value").map_elements(format_evalue).alias("e_value")
                )
            if "cov" in df_domains.columns:
                df_domains = df_domains.with_columns(
                    pl.col("cov").map_elements(format_float_2).alias("cov")
                )

            df_domains.write_csv(
                outdir / "defaultDomains.txt", separator="\t", include_header=False
            )

            # format metadata floats to two decimals (and e_value specially)
            df_meta = domains_data.select(desired_cols)

            def _format_any(val, col):
                if col == "e_value":
                    return format_evalue(val)
                try:
                    if isinstance(val, float):
                        return format_float_2(val)
                except Exception:
                    pass
                return val

            for c in df_meta.columns:
                if c == "e_value":
                    df_meta = df_meta.with_columns(pl.col(c).map_elements(format_evalue).alias(c))
                else:
                    df_meta = df_meta.with_columns(pl.col(c).map_elements(format_float_2).alias(c))

            # Write metadata with headers so downstream consumers can parse column names
            df_meta.write_csv(
                outdir / "defaultDomainsMetadata.txt", separator="\t", include_header=True
            )

    # Nucleotide links
    if nt_links is not None and isinstance(nt_links, pl.DataFrame) and nt_links.height > 0:
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
            nt_links.select(cols).write_csv(
                outdir / "defaultNucleotideLinks.txt", separator="\t", include_header=False
            )
        else:
            pl.DataFrame({c: [] for c in cols}).write_csv(
                outdir / "defaultNucleotideLinks.txt", separator="\t", include_header=False
            )
    else:
        pl.DataFrame(
            {
                c: []
                for c in ["query", "query_start", "query_end", "ref", "ref_start", "ref_end", "ani"]
            }
        ).write_csv(outdir / "defaultNucleotideLinks.txt", separator="\t", include_header=False)

    # Protein links
    if pairwise_aa is not None and isinstance(pairwise_aa, pl.DataFrame) and pairwise_aa.height > 0:
        cols = ["qseqid", "sseqid", "pident"]
        present = [c for c in cols if c in pairwise_aa.columns]
        if len(present) == len(cols):
            pairwise_aa.select(cols).write_csv(
                outdir / "defaultProteinLinks.txt", separator="\t", include_header=False
            )
        else:
            pl.DataFrame({c: [] for c in cols}).write_csv(
                outdir / "defaultProteinLinks.txt", separator="\t", include_header=False
            )
    else:
        pl.DataFrame({c: [] for c in ["qseqid", "sseqid", "pident"]}).write_csv(
            outdir / "defaultProteinLinks.txt", separator="\t", include_header=False
        )

    # Prefer package resource; fall back to relative path if needed
    resource_template = files("hoodini").joinpath("template", "template.html")

    # Read template HTML
    template_html = resource_template.read_text(encoding="utf-8")

    # Prepare data for embedding - escape for JS template literals
    def escape_js_string(s):
        """
        Escapes backticks, backslashes, and newlines for safe JS template literal embedding.
        """
        if not isinstance(s, str):
            return ""
        return (
            s.replace("\\", r"\\")
            .replace('"', r"\"")
            .replace("`", r"\`")
            .replace("\r", "")
            .replace("\n", r"\n")
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
        "DOMAINS_METADATA": outdir / "defaultDomainsMetadata.txt",
    }

    viz_data = {k: read_file_text(v) for k, v in viz_files.items()}

    # Render template with Jinja2
    from jinja2 import Template

    template = Template(template_html)
    rendered_html = template.render(**viz_data)

    # Write the rendered HTML to output dir
    (outdir / "hoodini-viz.html").write_text(rendered_html, encoding="utf-8")

    return outdir
