"""Debug utilities for IPG selection.

Usage (run as a script):
    python example/ipg_debug_notebook.py

This will:
- Load `ipg_debug.csv` (requires HOODINI_IPG_DEBUG=1 run beforehand)
- Show stage counts and example rows for the IDs of interest
- Print candidate rows per og_index for APL18794.1 and WP_061357476.1
"""

from __future__ import annotations

import polars as pl
from pathlib import Path

DEBUG_PATH = Path("ipg_debug.csv")

TARGET_PROTEINS = ["APL18794.1", "WP_061357476.1"]


def main() -> None:
    if not DEBUG_PATH.exists():
        print("ipg_debug.csv not found. Run with HOODINI_IPG_DEBUG=1 first.")
        return

    df = pl.read_csv(DEBUG_PATH)
    print("Stages:", df["stage"].unique().to_list())
    print("\nCounts by stage:\n", df.group_by("stage").len())

    for prot in TARGET_PROTEINS:
        subset = df.filter(pl.col("protein_id") == prot)
        if subset.is_empty():
            print(f"\nNo rows for {prot}")
            continue
        print(f"\nRows for {prot} by stage:")
        print(subset)
        for stage in subset["stage"].unique():
            print(f"\n[{prot}] stage={stage}")
            print(subset.filter(pl.col("stage") == stage))

    # Show top candidates per og_index in after_select_best
    final = df.filter(pl.col("stage") == "after_select_best")
    print("\nFinal selections (after_select_best):")
    print(final.sort(["og_index"]))


if __name__ == "__main__":
    main()
