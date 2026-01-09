from pathlib import Path
from typing import Optional
import json

import polars as pl
import base64
from importlib.resources import files
from jinja2 import Environment
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

    all_gff = to_polars(all_gff) if all_gff is not None else pl.DataFrame()
    all_neigh = to_polars(all_neigh) if all_neigh is not None else pl.DataFrame()
    all_prots = to_polars(all_prots) if all_prots is not None else pl.DataFrame()
    den_data = to_polars(den_data) if den_data is not None else pl.DataFrame()
    nt_links = to_polars(nt_links) if nt_links is not None else None
    pairwise_aa = to_polars(pairwise_aa) if pairwise_aa is not None else None
    domains_data = to_polars(domains_data) if domains_data is not None else None

    if all_gff is not None and all_gff.height > 0:
        gff_df = all_gff.clone()

        def normalize_attributes(val):
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                # Flatten dict to "key=value" pairs; preserve ordering best-effort.
                parts = []
                for k, v in val.items():
                    if v is None:
                        continue
                    parts.append(f"{k}={v}")
                return ";".join(parts)
            # Fallback to simple string conversion (avoid double-JSON encoding)
            return str(val)

        if "attributes" in gff_df.columns:
            gff_df = gff_df.with_columns(
                pl.col("attributes")
                .map_elements(normalize_attributes)
                .cast(pl.Utf8, strict=False)
            )
        gff_df.write_csv(outdir / "defaultGFF.gff", include_header=False, separator="\t")
        gff_df.write_parquet(outdir / "defaultGFF.parquet")
    else:
        empty = pl.DataFrame()
        empty.write_csv(outdir / "defaultGFF.gff", include_header=False, separator="\t")
        empty.write_parquet(outdir / "defaultGFF.parquet")

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
            neigh.select(cols).write_parquet(outdir / "defaultBaselines.parquet")
        else:
            (outdir / "defaultBaselines.txt").write_text(baseline_headers, encoding="utf-8")
            pl.DataFrame().write_parquet(outdir / "defaultBaselines.parquet")
    else:
        (outdir / "defaultBaselines.txt").write_text(baseline_headers, encoding="utf-8")
        pl.DataFrame().write_parquet(outdir / "defaultBaselines.parquet")

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

        # Drop redundant aliases if present
        drop_cols = []
        if "protein_id" in prots.columns:
            drop_cols.append("protein_id")
        if "clusterID" in prots.columns:
            drop_cols.append("clusterID")
        if drop_cols:
            prots = prots.drop(drop_cols)

        for col in ["target_prot", "target_nuc", "unique_id"]:
            if col not in prots.columns:
                prots = prots.with_columns(pl.lit("").alias(col))

        if "cluster" not in prots.columns:
            prots = prots.with_columns(pl.lit(None).alias("cluster"))
        else:
            prots = prots.with_columns(pl.col("cluster").round().cast(pl.Int64))

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

        for col in prots.columns:
            col_dtype = prots.schema.get(col)
            if col_dtype == pl.Utf8 or col_dtype == pl.String:
                prots = prots.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .str.replace_all("\r\n", " ")  
                    .str.replace_all("\n", " ")  
                    .str.replace_all("\r", " ")  
                    .str.replace_all("`", "'")  
                    .str.replace_all('"', "'")  
                    .alias(col)
                )
            elif col_dtype in (pl.Float64, pl.Float32):
                prots = prots.with_columns(pl.col(col).round(2).alias(col))
        csv_data = prots.write_csv(separator="\t", include_header=False)
        protein_headers = "\t".join(prots.columns) + "\n"
        (outdir / "defaultProteinMetadata.txt").write_text(
            protein_headers + csv_data, encoding="utf-8"
        )
        prots.write_parquet(outdir / "defaultProteinMetadata.parquet")
    else:
        protein_headers = "\t".join(base_headers) + "\n"
        (outdir / "defaultProteinMetadata.txt").write_text(protein_headers, encoding="utf-8")
        pl.DataFrame(schema={c: pl.Utf8 for c in base_headers}).write_parquet(
            outdir / "defaultProteinMetadata.parquet"
        )

    tree_headers = "leaf_id\tog_index\tsuperkingdom\tkingdom\tphylum\tclass\torder\tfamily\tgenus\tspecies\tstart_win\tend_win\tstrand_win\tstart_target\tend_target\n"
    if den_data is not None and den_data.height > 0:
        tree_meta = den_data.clone()
        if "unique_id" in tree_meta.columns:
            tree_meta = tree_meta.rename({"unique_id": "leaf_id"})
        csv_data = tree_meta.write_csv(separator="\t", include_header=False)
        (outdir / "defaultTreeMetadata.txt").write_text(tree_headers + csv_data, encoding="utf-8")
        tree_meta.write_parquet(outdir / "defaultTreeMetadata.parquet")
    else:
        (outdir / "defaultTreeMetadata.txt").write_text(tree_headers, encoding="utf-8")
        pl.DataFrame().write_parquet(outdir / "defaultTreeMetadata.parquet")

    (outdir / "defaultNewick.txt").write_text(tree_str or "", encoding="utf-8")

    if write_domains and domains_data is not None and domains_data.height > 0:
        # Normalize to the exact schema expected by hoodini-viz:
        # gene_id, domainName, start, end, source, evalue, coverage
        df = domains_data.clone()

        def c(name: str):
            return pl.col(name) if name in df.columns else pl.lit(None)

        df = df.with_columns(
            [
                pl.coalesce([c("gene_id"), c("protein_id")]).alias("gene_id_norm"),
                pl.coalesce([c("domainName"), c("domain_id")]).alias("domain_id_norm"),
                pl.coalesce([c("source"), c("database")]).alias("source_norm"),
                pl.coalesce([c("evalue"), c("e_value")]).alias("evalue_norm"),
                pl.coalesce([c("coverage"), c("cov")]).alias("coverage_norm"),
            ]
        )

        # Cast numeric fields; keep nulls when casting fails so we can drop incomplete rows.
        df = df.with_columns(
            [
                c("start").cast(pl.Float64, strict=False).alias("start"),
                c("end").cast(pl.Float64, strict=False).alias("end"),
                pl.col("evalue_norm").cast(pl.Float64, strict=False),
                pl.col("coverage_norm").cast(pl.Float64, strict=False),
            ]
        )

        df_domains = (
            df.select(
                [
                    pl.col("gene_id_norm").alias("gene_id"),
                    pl.col("domain_id_norm").alias("domainName"),
                    pl.col("start"),
                    pl.col("end"),
                    pl.col("source_norm").alias("source"),
                    pl.col("evalue_norm").alias("evalue"),
                    pl.col("coverage_norm").alias("coverage"),
                ]
            )
            .drop_nulls(["gene_id", "domainName", "start", "end"])
            .with_columns(
                [
                    pl.col("start").round(2),
                    pl.col("end").round(2),
                    pl.col("evalue").round(2),
                    pl.col("coverage").round(2),
                ]
            )
        )

        if df_domains.height == 0:
            df_domains = pl.DataFrame(
                {
                    "gene_id": [],
                    "domainName": [],
                    "start": [],
                    "end": [],
                    "source": [],
                    "evalue": [],
                    "coverage": [],
                }
            )

        df_domains.select(["gene_id", "domainName", "start", "end", "source", "evalue", "coverage"]).write_csv(
            outdir / "defaultDomains.txt",
            separator="\t",
            include_header=False,
        )
        df_domains.write_parquet(outdir / "defaultDomains.parquet")

        # Build a compact metadata table keyed by domain_id (domainName) if extra columns exist.
        metadata_exclude = {
            "gene_id",
            "protein_id",
            "domain_id",
            "domainName",
            "start",
            "end",
            "database",
            "source",
            "e_value",
            "evalue",
            "cov",
            "coverage",
            "gene_id_norm",
            "domain_id_norm",
            "source_norm",
            "evalue_norm",
            "coverage_norm",
        }
        drop_meta = {"bit_score*alignment_length", "domain_id_base", "domain_id_clean"}
        meta_candidates = [c for c in df.columns if c not in metadata_exclude and c not in drop_meta]
        if meta_candidates:
            df_meta = df.select(
                [pl.col("domain_id_norm").alias("domain_id")] + [pl.col(c) for c in meta_candidates]
            ).unique(subset=["domain_id"])
            numeric_cols = [c for c, dt in df_meta.schema.items() if getattr(dt, "is_numeric", lambda: False)()]
            df_meta = df_meta.with_columns([pl.col(c).round(2).alias(c) for c in numeric_cols])
            df_meta.write_csv(outdir / "defaultDomainsMetadata.txt", separator="\t", include_header=True)
            df_meta.write_parquet(outdir / "defaultDomainsMetadata.parquet")
        else:
            pl.DataFrame().write_csv(
                outdir / "defaultDomainsMetadata.txt", separator="\t", include_header=True
            )
            pl.DataFrame().write_parquet(outdir / "defaultDomainsMetadata.parquet")
    else:
        # Always emit empty parquet/text files so the front-end receives a valid data URL.
        empty_domains = pl.DataFrame(
            {"protein_id": [], "domain_id": [], "start": [], "end": [], "database": [], "e_value": [], "cov": []}
        )
        empty_domains.write_csv(outdir / "defaultDomains.txt", separator="\t", include_header=False)
        empty_domains.write_parquet(outdir / "defaultDomains.parquet")

        empty_meta = pl.DataFrame()
        empty_meta.write_csv(outdir / "defaultDomainsMetadata.txt", separator="\t", include_header=True)
        empty_meta.write_parquet(outdir / "defaultDomainsMetadata.parquet")

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
            nt_links.select(cols).write_parquet(outdir / "defaultNucleotideLinks.parquet")
        else:
            empty_nt = pl.DataFrame({c: [] for c in cols})
            empty_nt.write_csv(
                outdir / "defaultNucleotideLinks.txt", separator="\t", include_header=False
            )
            empty_nt.write_parquet(outdir / "defaultNucleotideLinks.parquet")
    else:
        empty_nt = pl.DataFrame(
            {
                c: []
                for c in ["query", "query_start", "query_end", "ref", "ref_start", "ref_end", "ani"]
            }
        )
        empty_nt.write_csv(outdir / "defaultNucleotideLinks.txt", separator="\t", include_header=False)
        empty_nt.write_parquet(outdir / "defaultNucleotideLinks.parquet")

    if pairwise_aa is not None and isinstance(pairwise_aa, pl.DataFrame) and pairwise_aa.height > 0:
        cols = ["qseqid", "sseqid", "pident"]
        present = [c for c in cols if c in pairwise_aa.columns]
        if len(present) == len(cols):
            pairwise_aa.select(cols).write_csv(
                outdir / "defaultProteinLinks.txt", separator="\t", include_header=False
            )
            pairwise_aa.select(cols).write_parquet(outdir / "defaultProteinLinks.parquet")
        else:
            empty_prot = pl.DataFrame({c: [] for c in cols})
            empty_prot.write_csv(
                outdir / "defaultProteinLinks.txt", separator="\t", include_header=False
            )
            empty_prot.write_parquet(outdir / "defaultProteinLinks.parquet")
    else:
        empty_prot = pl.DataFrame({c: [] for c in ["qseqid", "sseqid", "pident"]})
        empty_prot.write_csv(
            outdir / "defaultProteinLinks.txt", separator="\t", include_header=False
        )
        empty_prot.write_parquet(outdir / "defaultProteinLinks.parquet")

    # Render standalone HTML by injecting base64 parquet data into the template placeholders.
    resource_template = files("hoodini").joinpath("template", "template.html")
    template_html = resource_template.read_text(encoding="utf-8")

    def b64(path: Path) -> str:
        try:
            return base64.b64encode(Path(path).read_bytes()).decode()
        except Exception:
            return ""

    parquet_map = {
        "PARQUET_GFF_B64": outdir / "defaultGFF.parquet",
        "PARQUET_PROT_LINKS_B64": outdir / "defaultProteinLinks.parquet",
        "PARQUET_NUC_LINKS_B64": outdir / "defaultNucleotideLinks.parquet",
        "PARQUET_DOMAINS_B64": outdir / "defaultDomains.parquet",
        "PARQUET_BASELINES_B64": outdir / "defaultBaselines.parquet",
        "PARQUET_PROT_META_B64": outdir / "defaultProteinMetadata.parquet",
        "PARQUET_DOM_META_B64": outdir / "defaultDomainsMetadata.parquet",
        "PARQUET_TREE_META_B64": outdir / "defaultTreeMetadata.parquet",
    }

    env = Environment(autoescape=False, variable_start_string="%%", variable_end_string="%%")
    template = env.from_string(template_html)
    rendered_html = template.render(
        DEFAULT_NEWICK_B64=base64.b64encode((tree_str or "").encode()).decode(),
        **{k: b64(v) for k, v in parquet_map.items()},
    )

    (outdir / "hoodini-viz.html").write_text(rendered_html, encoding="utf-8")

    return outdir
