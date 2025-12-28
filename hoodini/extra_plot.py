import polars as pl
import zlib
import base64
import json
import os
from importlib.resources import files
from hoodini.utils.core import console

def run_extra_plotter(
    all_gff: pl.DataFrame,
    den_data: pl.DataFrame,
    dendrogram: pl.DataFrame,
    nc_data: pl.DataFrame,
    records: pl.DataFrame,
    output: str,
    domains_data: pl.DataFrame = pl.DataFrame(),
    genomad_df: pl.DataFrame = pl.DataFrame(),
    columns: list = None,
    include_domains: bool = False,
    include_genomad: bool = False,
    ticks: pl.DataFrame = pl.DataFrame(),
    tick_text: pl.DataFrame = pl.DataFrame(),
):
    all_gff = all_gff.copy()
    den_data = den_data.copy()

    # Use default column order if none provided
    if columns is None:
        columns = ["id", "start", "end", "strand", "fam_cluster", "fillcolor", "linecolor", 
                   "text_x", "text_y", "text_coordinates", "coordinates", "assembly_id", 
                   "species", "genus", "family", "order", "class", "phylum", "kingdom", 
                   "superkingdom", "prevalence"]

    all_gff = all_gff[columns]
    # Merge leaf labels into den_data
    den_data = den_data.join(
        records[["assembly_id", "unique_id"]],
        left_on="leaf_labels",
        right_on="unique_id",
        how="left",
    ).drop(["unique_id"])

    den_data = den_data.with_columns(
        [
            pl.col("assembly_id").str.split(".").list.first().alias("temp"),
            pl.lit([255, 228, 184, 255]).alias("bckg_color"),
        ]
    )

    if not include_domains:
        domains_data = pl.DataFrame()

    if not include_genomad:
        genomad_df = pl.DataFrame()

    os.makedirs(os.path.join(output, "plotdata"), exist_ok=True)
    # Write all outputs to CSV
    all_gff.write_csv(f"{output}/plotdata/all_gff.csv", include_header=False, encoding="utf-8")
    dendrogram.write_csv(f"{output}/plotdata/dendrogram.csv", include_header=False, encoding="utf-8")
    den_data.write_csv(f"{output}/plotdata/den_data.csv", include_header=False, encoding="utf-8")
    ticks.write_csv(f"{output}/plotdata/ticks.csv", include_header=False, encoding="utf-8")
    tick_text.write_csv(f"{output}/plotdata/tick_text.csv", include_header=False, encoding="utf-8")
    domains_data.write_csv(f"{output}/plotdata/domains.csv", include_header=False, encoding="utf-8")
    genomad_df.write_csv(f"{output}/plotdata/genomad.csv", include_header=False, encoding="utf-8")
    nc_data.write_csv(f"{output}/plotdata/nc_data.csv", include_header=False, encoding="utf-8")

    # Serialize all to CSV strings and compress
    csvdata = {
        "results": all_gff.write_csv(include_header=False),
        "dendrogram": dendrogram.write_csv(include_header=False),
        "dend_data": den_data.write_csv(include_header=False),
        "ticks": ticks.write_csv(include_header=False),
        "tick_text": tick_text.write_csv(include_header=False),
        "domains": domains_data.write_csv(include_header=False),
        "genomad": genomad_df.write_csv(include_header=False),
        "nc_data": nc_data.write_csv(include_header=False),
    }

    jsondata = json.dumps(csvdata)
    compressed = zlib.compress(jsondata.encode("utf-8"))
    base64data = base64.b64encode(compressed).decode("utf-8")

    template_path = files('hoodini').joinpath('data', 'template.html')
    with template_path.open("r") as f:
        html_template = f.read()

    html_output = html_template.replace("{{ base64data }}", base64data)

    html_path = os.path.join(output, f"{os.path.basename(output)}_plot.html")
    with open(html_path, "w") as f:
        f.write(html_output)

    return {
        "all_gff": all_gff,
        "den_data": den_data,
        "dendrogram": dendrogram,
        "domains_data": domains_data,
        "genomad_df": genomad_df,
        "nc_data": nc_data,
    }
