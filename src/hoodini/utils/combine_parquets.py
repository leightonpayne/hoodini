import polars as pl
from pathlib import Path

# Define input and output paths
hive_dir = Path(
    "/home/klaupaucius/software/hoodini/hoodini/data/contig_lengths"
)  # e.g., "hoodini/data/contig_lengths"
output_file = Path(
    "/home/klaupaucius/software/hoodini/hoodini/data/contig_lengths/contig_lengths.parquet"
)

# Read all part-*.parquet files lazily
lazy_df = pl.scan_parquet(str(hive_dir / "part-*.parquet"), allow_missing_columns=True)

# Collect into a single in-memory DataFrame
df = lazy_df.collect()

# Write to a single Parquet file (can compress too)
df.write_parquet(output_file, compression="zstd")

print(f"Combined parquet written to: {output_file}")
