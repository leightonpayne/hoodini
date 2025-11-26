import sys
import argparse
import polars as pl
from importlib.resources import files

def run_nuc2asmlen(accessions):
    """
    Fetch assembly accession and sequence length for nuccore IDs
    using a bundled Parquet file in `hoodini/data/contig_lengths`.

    Parameters:
        accessions (str or list): list of accession strings or path to file

    Returns:
        Polars DataFrame with columns: NucleotideAccession, AssemblyAccession, length
    """

    # Step 1: Read accession list
    if isinstance(accessions, str):
        with open(accessions) as f:
            query_accessions = [line.strip() for line in f if line.strip()]
    elif isinstance(accessions, (list, tuple)):
        query_accessions = list(accessions)
    else:
        raise ValueError("Expected list of accessions or path to file")

    if not query_accessions:
        raise ValueError("No accessions provided to run_nuc2asmlen")

    # Step 2: Locate bundled Parquet file
    parquet_path = files("hoodini").joinpath("data", "contig_lengths")

    # Step 3: Lazy scan and filter
    df_lazy = pl.scan_parquet(parquet_path, allow_missing_columns=True)

    filtered = df_lazy.filter(
        pl.col("genbankAccession").is_in(query_accessions) |
        pl.col("refseqAccession").is_in(query_accessions)
    ).select([
        "genbankAccession",
        "refseqAccession",
        "assemblyAccession",
        "length"
    ])

    matches = filtered.collect()

    # Step 4: Prefer GCF matches from refseq
    ref_matches = matches.filter(
        pl.col("refseqAccession").is_in(query_accessions) &
        pl.col("assemblyAccession").str.starts_with("GCF")
    ).with_columns([
        pl.col("refseqAccession").alias("NucleotideAccession")
    ])

    # Step 5: Fallback to GCA matches from genbank
    gbk_matches = matches.filter(
        pl.col("genbankAccession").is_in(query_accessions) &
        pl.col("assemblyAccession").str.starts_with("GCA")
    ).with_columns([
        pl.col("genbankAccession").alias("NucleotideAccession")
    ])

    # Step 6: Combine and deduplicate (prefer GCF)
    combined = pl.concat([ref_matches, gbk_matches], how="vertical") \
                 .unique(subset=["NucleotideAccession"])

    # Step 7: Join with full query list to preserve order
    query_df = pl.DataFrame({"NucleotideAccession": query_accessions})
    result = query_df.join(
        combined.select(["NucleotideAccession", "assemblyAccession", "length"]),
        on="NucleotideAccession",
        how="left"
    ).rename({
        "assemblyAccession": "AssemblyAccession"
    })

    return result.to_pandas()


# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="Fetch assembly + length info for nuccore accessions from bundled Parquet")
    parser.add_argument("input_file", help="File with one accession per line")
    parser.add_argument("-o", "--output", help="Output TSV file (default: stdout)", default=None)
    args = parser.parse_args()

    df = run_nuc2asmlen(args.input_file)

    if args.output:
        df.to_csv(args.output, sep="\t", index=False)
    else:
        sys.stdout.write(df.to_csv(sep="\t", index=False))

if __name__ == "__main__":
    main()
