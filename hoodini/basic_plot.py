from bz2 import compress
import polars as pl
import zlib,base64
import json
from importlib.resources import files
import hoodini
import importlib.resources as pkg_resources


class Plotter:
    def __init__(self, obj):
        self.master = obj
        for key, val in vars(obj).items():
            setattr(self, key, val)

    def run(self):       

        results = self.results
        den_data = self.den_data
        dendrogram = self.dendrogram
        if self.domains:
            domains = self.domains_data
        ticks = self.ticks
        tick_text = self.tick_text

        results = results.with_columns(
            pl.col("assembly_accession").str.split(".").list.first().alias("temp")
        )
        results = results.join(
            self.type_strains[["bacdive_id", "strain_number_header", "type_strain_assembly"]],
            left_on="temp",
            right_on="type_strain_assembly",
            how="left",
        ).drop(["temp", "type_strain_assembly"])
        results = results[
            [
                "seqid",
                "id",
                "strand",
                "species",
                "rel_start",
                "rel_end",
                "product",
                "bacdive_id",
                "strain_number_header",
                "fam_cluster",
                "coordinates",
                "fillcolor",
                "linecolor",
                "text_coordinates",
                "prevalence",
                "sequence",
                "assembly_accession",
            ]
        ]

        den_data = den_data.with_columns(
            pl.col("assembly_accession").str.split(".").list.first().alias("temp")
        )
        den_data = den_data.join(
            self.type_strains[["bacdive_id", "strain_number_header", "type_strain_assembly"]],
            left_on="temp",
            right_on="type_strain_assembly",
            how="left",
        ).drop(["temp"])
        den_data = den_data.with_columns(
            pl.when(pl.col("strain_number_header").is_null())
            .then(pl.lit([255, 255, 255, 0]))
            .otherwise(pl.lit([255, 228, 184, 255]))
            .alias("bckg_color")
        )
        
        #save to csv files
    
        
        
        csvdata = {
            "results": results.write_csv(include_header=False, encoding='utf-8'),
            "dendrogram": dendrogram.write_csv(include_header=False, encoding='utf-8'),
            "dend_data": den_data.write_csv(include_header=False, encoding='utf-8'),
            "ticks": ticks.write_csv(include_header=False, encoding='utf-8'),
            "tick_text": tick_text.write_csv(include_header=False, encoding='utf-8')
        }

        # Convert the csvdata to JSON and compress it
        jsondata = json.dumps(csvdata)
        compressed_data = zlib.compress(jsondata.encode('utf-8'))
        base64data = base64.b64encode(compressed_data).decode('utf-8')
        template_path = files('hoodini').joinpath('data', 'template.html')
        with template_path.open('r') as f:
            html_template = f.read()

        html_output = html_template.replace('{{ base64data }}', base64data)
        
        with open(self.output+"/"+self.output+".html","w+") as f:
            f.write(html_output)
