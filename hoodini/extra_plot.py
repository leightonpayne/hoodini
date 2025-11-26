import pandas as pd
import zlib
import base64
import json
import os
from importlib.resources import files
from hoodini.utils.core import console

def run_extra_plotter(
    all_gff: pd.DataFrame,
    den_data: pd.DataFrame,
    dendrogram: pd.DataFrame,
    nc_data: pd.DataFrame,
    records: pd.DataFrame,
    output: str,
    domains_data: pd.DataFrame = pd.DataFrame(),
    genomad_df: pd.DataFrame = pd.DataFrame(),
    columns: list = None,
    include_domains: bool = False,
    include_genomad: bool = False,
    ticks: pd.DataFrame = pd.DataFrame(),
    tick_text: pd.DataFrame = pd.DataFrame(),
):
    all_gff = all_gff.copy()
    den_data = den_data.copy()

    # Use default column order if none provided
    if columns is None:
        columns = ["id", "start", "end", "strand", "fam_cluster", "fillcolor", "linecolor", 
                   "text_x", "text_y", "text_coordinates", "coordinates", "assembly_id", 
                   "species", "genus", "family", "order", "class", "phylum", "kingdom", 
                   "superkingdom", "prevalence"]

    all_gff["temp"] = all_gff["assembly_id"].str.split(".").str[0]

    all_gff = all_gff[columns]
    # Merge leaf labels into den_data
    den_data = pd.merge(
        den_data,
        records[["assembly_id", "unique_id"]],
        left_on="leaf_labels", right_on="unique_id", how="left"
    ).drop(columns=["unique_id"])
    
    den_data["temp"] = den_data["assembly_id"].str.split(".").str[0]
    den_data['bckg_color'] = [[255, 228, 184, 255]] * len(den_data)

    if not include_domains:
        domains_data = pd.DataFrame()

    if not include_genomad:
        genomad_df = pd.DataFrame()

    os.makedirs(os.path.join(output, "plotdata"), exist_ok=True)
    # Write all outputs to CSV
    all_gff.to_csv(f"{output}/plotdata/all_gff.csv", index=False, encoding="utf-8")
    dendrogram.to_csv(f"{output}/plotdata/dendrogram.csv", index=False, encoding="utf-8")
    den_data.to_csv(f"{output}/plotdata/den_data.csv", index=False, encoding="utf-8")
    ticks.to_csv(f"{output}/plotdata/ticks.csv", index=False, encoding="utf-8")
    tick_text.to_csv(f"{output}/plotdata/tick_text.csv", index=False, encoding="utf-8")
    domains_data.to_csv(f"{output}/plotdata/domains.csv", index=False, encoding="utf-8")
    genomad_df.to_csv(f"{output}/plotdata/genomad.csv", index=False, encoding="utf-8")
    nc_data.to_csv(f"{output}/plotdata/nc_data.csv", index=False, encoding="utf-8")

    # Serialize all to CSV strings and compress
    csvdata = {
        "results": all_gff.to_csv(index=False),
        "dendrogram": dendrogram.to_csv(index=False),
        "dend_data": den_data.to_csv(index=False),
        "ticks": ticks.to_csv(index=False),
        "tick_text": tick_text.to_csv(index=False),
        "domains": domains_data.to_csv(index=False),
        "genomad": genomad_df.to_csv(index=False),
        "nc_data": nc_data.to_csv(index=False),
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
