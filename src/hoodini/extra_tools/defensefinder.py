import os
import polars as pl
import subprocess
from hoodini.utils.logging_utils import console


def run_defensefinder(all_gff, all_prots, output):
    console.print("🛡️\tRunning DefenseFinder...")
    deffinder_df = pl.DataFrame()
    try:
        # Convert to Polars if needed
        if not isinstance(all_gff, pl.DataFrame):
            all_gff = pl.from_pandas(all_gff)
        if not isinstance(all_prots, pl.DataFrame):
            all_prots = pl.from_pandas(all_prots)

        temp_gff = all_gff.clone()

        # Extract ID from attributes and join with proteins to get sequences
        temp_gff = temp_gff.with_columns(
            pl.col("attributes").str.extract(r"ID=([^;]+)").alias("id")
        )
        temp_gff = temp_gff.join(
            all_prots.select(["id", "unique_id", "sequence"]), on="id", how="left"
        )

        # Select and add temp_index for mapping results back
        temp_gff = temp_gff.select(
            [
                "seqid",
                "source",
                "type",
                "start",
                "end",
                "score",
                "strand",
                "phase",
                "id",
                "sequence",
            ]
        ).with_row_count("temp_index")

        temp_gff = temp_gff.with_columns(
            (pl.col("seqid") + "_" + pl.col("temp_index").cast(pl.Utf8)).alias("fasta_id")
        )

        # Write FASTA for defensefinder
        with open(f"{output}/proteome.fasta", "w") as f:
            for row in temp_gff.iter_rows(named=True):
                if row["sequence"] is not None:
                    f.write(f">{row['fasta_id']}\n{row['sequence']}\n")

        # Run defensefinder
        command = [
            "defense-finder",
            "run",
            f"{output}/proteome.fasta",
            "-a",
            "--db-type",
            "gembase",
            "-o",
            f"{output}/defense_finder",
        ]
        subprocess.run(command, check=True)

        # Parse results
        result_file = f"{output}/defense_finder/proteome_defense_finder_genes.tsv"
        if os.path.exists(result_file):
            deffinder_result = pl.read_csv(result_file, separator="\t")
            if deffinder_result.height > 0:
                # Extract temp_index from hit_id (last part after _)
                deffinder_result = deffinder_result.with_columns(
                    pl.col("hit_id").str.split("_").list.last().cast(pl.Int32).alias("temp_index")
                )

                # Add deffinder columns with type and subtype
                deffinder_result = deffinder_result.with_columns(
                    pl.col("type").alias("deffinder_type"),
                    pl.col("subtype").alias("deffinder_subtype"),
                )

                # Extract gene name: for Cas, take first part; otherwise take second part after __
                # Use safe extraction with error handling
                def extract_gene(gene_name, def_type):
                    if gene_name is None:
                        return ""
                    try:
                        if def_type == "Cas":
                            return gene_name.split("_")[0]
                        else:
                            parts = gene_name.split("__")
                            return parts[1] if len(parts) > 1 else ""
                    except:
                        return ""

                deffinder_result = deffinder_result.with_columns(
                    pl.struct(["gene_name", "deffinder_type"])
                    .map_elements(lambda s: extract_gene(s["gene_name"], s["deffinder_type"]))
                    .alias("deffinder_gene")
                )

                # Join back with temp_gff to get original protein IDs
                temp_gff = temp_gff.join(deffinder_result, on="temp_index", how="left")

                # Select and deduplicate
                deffinder_df = (
                    temp_gff.select(["id", "deffinder_type", "deffinder_subtype", "deffinder_gene"])
                    .filter(pl.col("deffinder_type").is_not_null())
                    .unique(subset=["id"])
                )
            else:
                console.print("[yellow]No defensefinder results found")
        else:
            console.print("[yellow]DefenseFinder output file not found")

    except Exception as e:
        console.print(f"[red]DefenseFinder failed: {e}")

    return deffinder_df
