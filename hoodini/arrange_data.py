import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from hoodini.utils.core import flat, desaturate, darken

def run_arranger(
    *,
    records: pd.DataFrame,
    all_gff: pd.DataFrame,
    all_neigh: pd.DataFrame,
    den_data: pd.DataFrame,
    domains_data: pd.DataFrame = None,
    nc_data: pd.DataFrame = None,
    genomad_df: pd.DataFrame = None,
    leaf_labels: list[str],
    dendro: dict,
    output: str,
    min_prevalence: float = 0.05,
    domains: bool = True,
    genomad: bool = True,
) -> dict:
    # ─────────────────────── Arrange genes ───────────────────────
    for i in range(1, len(den_data['y'])):
        if den_data['y'][i] - den_data['y'][i-1] != 0:
            y_step = den_data['y'][i] - den_data['y'][i-1]
            break
    y_half_height = y_step / 6
    y_quarter_height = y_step / 12

    swap_mask = all_gff["start"] > all_gff["end"]
    all_gff.loc[swap_mask, ["start", "end"]] = all_gff.loc[swap_mask, ["end", "start"]].values

    all_neigh["unique_id"] = all_neigh["unique_id"].astype(str)
    all_gff = all_gff.merge(
        all_neigh[["start_target", "end_target", "start_win", "end_win", "strand_win", "unique_id"]],
        on="unique_id",
        how="left"
    )
    all_gff["rel_start"] = all_gff["start"] - all_gff["start_target"]
    all_gff["rel_end"] = all_gff["end"] - all_gff["start_target"]

    delta = all_gff["end_target"] - all_gff["start_target"]
    all_gff["flip_strand"] = np.where(
        all_gff["strand_win"] == "-",
        np.where(all_gff["strand"] == "+", "-", "+"),
        all_gff["strand"]
    )
    all_gff["flipped"] = np.where(all_gff["strand_win"] == "-", "flipped", "")

    all_gff["rel_start"] = np.where(
        all_gff["strand_win"] == "-",
        delta - all_gff["rel_end"],
        all_gff["rel_start"]
    )
    all_gff["rel_end"] = np.where(
        all_gff["strand_win"] == "-",
        delta - all_gff["rel_start"],
        all_gff["rel_end"]
    )

    neg_strand = all_gff["flip_strand"] == "-"
    all_gff["dx"] = all_gff["rel_end"] - all_gff["rel_start"]
    gene_x_tail = np.where(neg_strand, all_gff["rel_end"], all_gff["rel_start"])
    gene_dx = np.where(neg_strand, -1 * all_gff["dx"], all_gff["dx"])
    gene_x_head = gene_x_tail + gene_dx
    gene_x_head_start = np.where(neg_strand, gene_x_head + 100, gene_x_head - 150)
    text_x = np.where(
        neg_strand,
        gene_x_tail - (gene_x_tail - gene_x_head_start) / 2,
        gene_x_tail + (gene_x_head_start - gene_x_tail) / 2
    )

    all_gff["text_x"] = text_x / 100
    all_gff["text_y"] = all_gff["y"] + y_half_height
    all_gff["xs"] = [list(n) for n in zip(gene_x_tail, gene_x_tail, gene_x_head_start, gene_x_head, gene_x_head_start)]
    all_gff["xs"] = all_gff["xs"].apply(lambda lst: [x / 100 for x in lst])
    all_gff["ys"] = [list(n) for n in zip(
        all_gff["y"] - y_half_height, all_gff["y"] + y_half_height,
        all_gff["y"] + y_half_height, all_gff["y"], all_gff["y"] - y_half_height
    )]
    all_gff["coordinates"] = [[n] for n in [list(zip(x, y)) for x, y in zip(all_gff["xs"], all_gff["ys"])]]
    all_gff["text_coordinates"] = [[x, y] for x, y in zip(all_gff["text_x"], all_gff["text_y"])]

    # ─────────── Dendrogram path and coordinates ───────────
    min_x = min(n[0] for n in flat(flat(all_gff["coordinates"])))
    max_x = max(n[0] for n in flat(flat(all_gff["coordinates"])))
    max_y = max(n[1] for n in flat(flat(all_gff["coordinates"])))
    max_text_len = max(len(n) for n in den_data["species"]) * 4
    path = [
        [list(n) for n in zip(
            [-v * math.sqrt(len(leaf_labels)) + min_x - max_text_len for v in dcoord],
            icoord
        )]
        for dcoord, icoord in zip(dendro["dcoord"], dendro["icoord"])
    ]
    dendrogram = pd.DataFrame({'path': path, "color": [[0, 0, 0, 100]] * len(path)})
    dendrogram[["Ax", "Ay", "Bx", "By", "Cx", "Cy", "Dx", "Dy"]] = pd.DataFrame(
        [[*p[0], *p[1], *p[2], *p[3]] for p in path], index=dendrogram.index
    )

    den_data["coordinates"] = [[x + min_x - max_text_len, y] for x, y in zip(den_data["x"], den_data["y"])]
    den_data["start"] = den_data["start_win"]
    den_data["end"] = den_data["end_win"]
    den_data["rel_start"] = den_data["start_win"] - den_data["start_target"]
    den_data["rel_end"] = den_data["end_win"] - den_data["start_target"]
    den_data["delta"] = den_data["end_target"] - den_data["start_target"]

    upstream = den_data["rel_start"] > 0
    downstream = den_data["rel_start"] < 0
    flipped = den_data["strand_win"] == "-"
    den_data["rel_start"] = np.where(upstream & flipped, -(den_data["rel_end"] - den_data["delta"]),
                             np.where(downstream & flipped, -(den_data["rel_end"]) + den_data["delta"], den_data["rel_start"]))
    den_data["rel_end"] = np.where(upstream & flipped, -(den_data["rel_start"] - den_data["delta"]),
                           np.where(downstream & flipped, -(den_data["rel_start"]) + den_data["delta"], den_data["rel_end"]))
    den_data["rel_start"] = den_data["rel_start"] / 100
    den_data["rel_end"] = den_data["rel_end"] / 100
    den_data["start_line"] = [[x, y] for x, y in zip(den_data["rel_start"], den_data["y"])]
    den_data["end_line"] = [[x, y] for x, y in zip(den_data["rel_end"], den_data["y"])]

    # ─────────── Genome ruler ticks ───────────
    min_bp = all_gff["rel_start"].min()
    max_bp = all_gff["rel_end"].max()
    min_thousand = math.floor(abs(min_bp) / 1000) * 1000
    max_thousand = math.floor(max_bp / 1000) * 1000
    baseline_y = max_y + y_step
    lines_start = [[x / 100, baseline_y - y_quarter_height] for x in range(-min_thousand, max_thousand + 1000, 1000)]
    lines_end = [[x / 100, baseline_y + y_quarter_height] for x in range(-min_thousand, max_thousand + 1000, 1000)]
    ticks = pd.DataFrame({'start': lines_start + [[min_x, baseline_y]], 'end': lines_end + [[max_x, baseline_y]]})
    tick_text = pd.DataFrame({
        'text': [str(x) for x in range(-min_thousand, max_thousand + 1000, 1000)],
        'coordinates': [[x / 100, baseline_y + y_quarter_height * 2] for x in range(-min_thousand, max_thousand + 1000, 1000)]
    })

    # ─────────── Add prevalence color to clusters ───────────
    prevalence = all_gff.groupby("fam_cluster")["unique_id"].nunique() / all_gff["unique_id"].nunique()
    prevalence = prevalence.round(2).to_dict()
    all_gff["prevalence"] = all_gff["fam_cluster"].map(prevalence)

    families = all_gff["fam_cluster"].dropna().unique().tolist()
    colors_rgb = [plt.cm.gist_ncar(random.random()) for _ in families]
    colors_dic = {
        num: desaturate(list(color), 1 - prevalence[num] * 0.6, 1)
        if prevalence[num] >= min_prevalence else [230, 230, 230, 255]
        for num, color in zip(families, colors_rgb)
    }
    all_gff["fillcolor"] = all_gff["fam_cluster"].map(colors_dic).apply(lambda d: d if isinstance(d, list) else [230, 230, 230, 255])
    all_gff["linecolor"] = all_gff["fillcolor"].apply(lambda x: darken(x, 0.5, 1))

    all_gff = all_gff.drop_duplicates(subset=['start', 'end', "seqid", "target_prot"])
    all_gff = all_gff.merge(
        records[['assembly_id', 'taxid', 'superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'unique_id']],
        on="unique_id",
        how="left"
    )
    all_gff.drop_duplicates(subset=['start', 'end', "seqid"]).to_csv(os.path.join(output, "results.txt"), sep="\t", index=False)

    # Domain, ncRNA, and genomad will be handled in next message due to length limits. Shall I continue with those chunks now?
    results = {
        "records": records,
        "all_gff": all_gff,
        "den_data": den_data,
        "dendrogram": dendrogram,
        "ticks": ticks,
        "tick_text": tick_text,
    }

    # ─────────── Domains ───────────
    if domains and domains_data is not None and not domains_data.empty:
        domains_data = domains_data.merge(
            all_gff[["id", "rel_start", "rel_end", "flip_strand", "y"]],
            left_on="protein_id",
            right_on="id",
            how="left"
        ).dropna(subset=["y"])

        start = np.where(domains_data["flip_strand"] == "+",
                         domains_data["start"] * 3 + domains_data["rel_start"],
                         domains_data["rel_end"] - domains_data["end"] * 3)
        end = np.where(domains_data["flip_strand"] == "+",
                       domains_data["end"] * 3 + domains_data["rel_start"],
                       domains_data["rel_end"] - domains_data["start"] * 3)
        domains_data["start"] = start
        domains_data["end"] = end

        tmp_evals = domains_data["e_value"].copy()
        domains_data["e_value"] = domains_data["e_value"].clip(lower=1e-100).astype(float)
        norm = matplotlib.colors.LogNorm(vmin=domains_data["e_value"].min(), vmax=domains_data["e_value"].max())
        cmap = plt.cm.plasma
        domains_data["colors"] = domains_data["e_value"].map(lambda v: list(cmap(norm(v))))
        domains_data["colors"] = domains_data["colors"].apply(lambda x: [int(255 * i) for i in x[:-1]] + [100])

        def norm_xs(xs): return [x / 100 for x in xs]

        domains_data["xs"] = [list(pair) for pair in zip(domains_data["start"], domains_data["start"], domains_data["end"], domains_data["end"])]
        domains_data["xs"] = domains_data["xs"].apply(norm_xs)
        domains_data["ys"] = [list(n) for n in zip(
            domains_data["y"] - y_half_height - y_quarter_height * domains_data["y_pos"],
            (domains_data["y"] - y_half_height - y_quarter_height * domains_data["y_pos"]) - y_quarter_height,
            (domains_data["y"] - y_half_height - y_quarter_height * domains_data["y_pos"]) - y_quarter_height,
            domains_data["y"] - y_half_height - y_quarter_height * domains_data["y_pos"]
        )]
        domains_data["coordinates"] = [[list(zip(x, y))] for x, y in zip(domains_data["xs"], domains_data["ys"])]
        domains_data = domains_data[domains_data["y_pos"] < 2]
        domains_data["e_value"] = tmp_evals.astype(str)
        domains_data.to_csv(os.path.join(output, "domains.txt"), sep="\t", index=False)
        results["domains_data"] = domains_data

    # ─────────── ncRNA ───────────
    if nc_data is not None and not nc_data.empty:
        nc_data["rel_start"] = nc_data["start"] - nc_data["start_target"]
        nc_data["rel_end"] = nc_data["end"] - nc_data["start_target"]
        nc_data["delta"] = nc_data["end_target"] - nc_data["start_target"]

        upstream = nc_data["rel_start"] > 0
        downstream = nc_data["rel_start"] < 0
        flipped = nc_data["strand_win"] == "-"

        new_start = np.where(upstream & flipped, -(nc_data["rel_end"] - nc_data["delta"]),
                             np.where(downstream & flipped, -(nc_data["rel_end"]) + nc_data["delta"], nc_data["rel_start"]))
        new_end = np.where(upstream & flipped, -(nc_data["rel_start"] - nc_data["delta"]),
                           np.where(downstream & flipped, -(nc_data["rel_start"]) + nc_data["delta"], nc_data["rel_end"]))
        nc_data["rel_start"], nc_data["rel_end"] = new_start, new_end

        nc_data["dx"] = nc_data["rel_end"] - nc_data["rel_start"]
        gene_x_tail = nc_data["rel_start"]
        gene_x_head = gene_x_tail + nc_data["dx"]
        gene_x_head_start = gene_x_head
        text_x = gene_x_tail + (gene_x_head_start - gene_x_tail) / 2

        nc_data["text_x"] = text_x / 100
        nc_data["text_y"] = nc_data["y"] + y_half_height
        nc_data["xs"] = [list(x) for x in zip(gene_x_tail, gene_x_tail, gene_x_head_start, gene_x_head, gene_x_head_start)]
        nc_data["xs"] = nc_data["xs"].apply(lambda x: [i / 100 for i in x])
        nc_data["ys"] = [list(y) for y in zip(
            nc_data["y"] - y_half_height,
            nc_data["y"] + y_half_height,
            nc_data["y"] + y_half_height,
            nc_data["y"],
            nc_data["y"] - y_half_height
        )]
        nc_data["coordinates"] = [[list(zip(x, y))] for x, y in zip(nc_data["xs"], nc_data["ys"])]

        families = nc_data["nc_feature"].dropna().unique().tolist()
        colors_rgb = [plt.cm.rainbow(random.random()) for _ in families]
        colors_dic = {f: desaturate(list(c), 0.6, 1) for f, c in zip(families, colors_rgb)}
        nc_data["fillcolor"] = nc_data["nc_feature"].map(colors_dic).apply(
            lambda d: d if isinstance(d, list) else [230, 230, 230, 255]
        )

        nc_data.to_csv(os.path.join(output, "ncdata.txt"), sep="\t", index=False)
        results["nc_data"] = nc_data

    # ─────────── GenoMAD ───────────
    if genomad and genomad_df is not None and not genomad_df.empty:
        genomad_df["rel_start"] = genomad_df["start"] - genomad_df["start_target"]
        genomad_df["rel_end"] = genomad_df["end"] - genomad_df["start_target"]
        genomad_df["delta"] = genomad_df["end_target"] - genomad_df["start_target"]

        upstream = genomad_df["rel_start"] > 0
        downstream = genomad_df["rel_start"] < 0
        flipped = genomad_df["strand_win"] == "-"

        genomad_df["rel_start"] = np.where(upstream & flipped, -(genomad_df["rel_end"] - genomad_df["delta"]),
                                           np.where(downstream & flipped, -(genomad_df["rel_end"]) + genomad_df["delta"], genomad_df["rel_start"]))
        genomad_df["rel_end"] = np.where(upstream & flipped, -(genomad_df["rel_start"] - genomad_df["delta"]),
                                         np.where(downstream & flipped, -(genomad_df["rel_start"]) + genomad_df["delta"], genomad_df["rel_end"]))

        genomad_df["rel_start"] = genomad_df["rel_start"] / 100
        genomad_df["rel_end"] = genomad_df["rel_end"] / 100
        genomad_df["y"] = genomad_df["y"] + 1.05 * y_half_height
        genomad_df["start_line"] = [list(x) for x in zip(genomad_df["rel_start"], genomad_df["y"])]
        genomad_df["end_line"] = [list(x) for x in zip(genomad_df["rel_end"], genomad_df["y"])]
        genomad_df["fillcolor"] = genomad_df["mge_type"].map({
            "plasmid": [42, 157, 143, 255],
            "virus": [244, 162, 97, 255]
        })

        genomad_df.to_csv(os.path.join(output, "genomad.txt"), sep="\t", index=False)
        results["genomad_df"] = genomad_df

    return results
