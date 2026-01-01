# add all imports
import os
import gb_io


def extract_neighborhood(
    protein_id,
    nucleotide_id,
    gbf_file,
    gff_file,
    faa_file,
    fna_file,
    mode="win_nts",
    window=None,
    strand=None,
    start=None,
    end=None,
    unique_id=None,
    input_type=None,
    sorfs=None,
):

    neighborhood = {}
    if gbf_file:

        record_found = False

        if not os.path.exists(gbf_file):
            return (
                None,
                None,
                unique_id,
                "GenBank file not found",
            )  # Return None and index for failed extractions

        # Stream GenBank records to avoid loading everything in memory
        try:
            record_iter = gb_io.iter(gbf_file)
        except Exception:
            return None, None, unique_id, "gb_io failed to open GenBank iterator"

        if nucleotide_id:
            # Iterate records and stop when version matches the target nucleotide_id
            record_found = False
            for record in record_iter:
                record_version = getattr(record, "version", None)
                if record_version and nucleotide_id in record_version:
                    record_found = True
                    break
        if record_found:
            feature_data = process_features(record.features, record_version)
            feature_data = pl.DataFrame(feature_data)
            if "attributes" in feature_data.columns:
                attributes_df = pl.DataFrame(feature_data["attributes"].to_list())
                attributes_df.drop(["protein_id"])
                feature_data = pl.concat(
                    [feature_data.drop(["attributes"]), attributes_df], how="horizontal"
                )
                feature_data = feature_data.rename({"translation": "sequence"})
            else:
                return None, None, unique_id, "GenBank file is not annotated"
        else:
            return None, None, unique_id, "GenBank record not found"

        if input_type == "protein":

            if "protein_id" in feature_data.columns:
                if not (start and end):
                    # get the start from the row in which protein_id matches the input protein_id
                    start = feature_data[feature_data["protein_id"] == protein_id]["start"].iloc[0]
                    end = feature_data[feature_data["protein_id"] == protein_id]["end"].iloc[0]
                    strand = feature_data[feature_data["protein_id"] == protein_id]["strand"].iloc[
                        0
                    ]
            start, end = int(start), int(end)

            if mode == "win_nts":
                start_win = start - window
                end_win = end + window
                if start_win < 0:
                    start_win = 0
                if end_win > len(record.sequence):
                    end_win = len(record.sequence)
                subgff = feature_data.query("start>=@start_win & end<=@end_win")

            elif mode == "win_ngen":
                subgff = feature_data.reset_index(drop=True)
                # get the index of the row in which protein_id matches the input protein_id
                prot_index = subgff[subgff["protein_id"] == protein_id].index.to_list()[0]
                subgff = subgff[prot_index - window : prot_index + window]
                start_win = subgff["start"].min()
                end_win = subgff["end"].max()

        elif input_type == "nucleotide":

            if nucleotide_id and (start and end) and window:
                start = int(start)
                end = int(end)
                if mode == "win_nts":
                    start_win = start - window
                    end_win = end + window
                    if not strand:
                        strand = "+" if end > start else "-"
                    subgff = feature_data.query("start>=@start_win & end<=@end_win")

                elif mode == "win_ngen":
                    subgff = feature_data.reset_index(drop=True)
                    # get the index of the first and last feature in the subgff dataframe between start and end
                    start_index = subgff[subgff["start"] >= start].index.to_list()[0]
                    end_index = subgff[subgff["end"] <= end].index.to_list()[-1]
                    # get start_index - window and end_index + window
                    subgff = subgff[start_index - window : end_index + window]
                    start_win = subgff["start"].min()
                    end_win = subgff["end"].max()
                    if not strand:
                        strand = "+" if end > start else "-"

            elif not window and nucleotide_id and (start and end):
                start, end = int(start), int(end)
                if not strand:
                    strand = "-" if end < start else "+"
                subgff = feature_data.query(
                    "seqid == @nucleotide_id & type =='CDS' & start>=@start & end<=@end"
                )
                start_win = subgff["start"].min()
                end_win = subgff["end"].max()

            elif not (start and end):
                start = end = 0
                subgff = feature_data
                start_win = subgff["start"].min()
                end_win = subgff["end"].max()
                strand = "+"

        if start_win < 0:
            start_win = 0
        if end_win > len(record.sequence):
            end_win = len(record.sequence)
        subgff["id"] = subgff["protein_id"]
        header = [
            "seqid",
            "source",
            "type",
            "start",
            "end",
            "score",
            "strand",
            "phase",
            "protein_id",
            "id",
            "sequence",
        ]
        if "product" in subgff.columns:
            header.append("product")
        else:
            subgff["product"] = None
        subgff = subgff[header]
        neighborhood = {
            "seqid": record_version,
            "start_target": start,
            "end_target": end,
            "start_win": start_win,
            "end_win": end_win,
            "strand_win": strand,
            "sequence": record.sequence[start_win:end_win].decode("utf-8"),
            "unique_id": unique_id,
        }
        if sorfs:
            # annotate genes in the neighborhood with pyrodigal:
            orf_finder = pyrodigal.GeneFinder(meta=True, min_gene=10, max_overlap=9)
            new_genes = []

            for i, pred in enumerate(
                orf_finder.find_genes(record.sequence[start_win:end_win].decode("utf-8"))
            ):
                overlap_flag = False
                for row in subgff.iter_rows(named=True):
                    overlap_percentage = calculate_overlap(
                        row["start"], row["end"], pred.begin + start_win, pred.end + start_win
                    )
                    if overlap_percentage > 10:
                        overlap_flag = True
                        break
                if not overlap_flag:
                    new_genes.append(
                        {
                            "seqid": nucleotide_id,
                            "source": "pyrodigal",
                            "type": "CDS",
                            "start": pred.begin + start_win,
                            "end": pred.end + start_win,
                            "score": pred.score,
                            "strand": "-" if pred.strand == "-1" else "+",
                            "phase": ".",
                            "protein_id": f"sORF_{unique_id}_{i}",
                            "id": f"sORF_{unique_id}_{i}",
                            "sequence": pred.translate(),  # Assuming you want to store the translated protein sequence
                        }
                    )

            # Convert new genes to DataFrame and concatenate with existing subgff
            if new_genes:
                new_genes_df = pl.DataFrame(new_genes)
                subgff = pl.concat([subgff, new_genes_df], how="vertical")

            new_genes = []
            seq = record.sequence[start_win:end_win].decode("utf-8").upper()
            for i, (start, stop, strand, description) in enumerate(
                orfipy_core.orfs(seq, minlen=100, maxlen=1000, partial3=False, between_stops=False)
            ):
                overlap_flag = False
                for row in subgff.iter_rows(named=True):
                    overlap_percentage = calculate_overlap(
                        row["start"], row["end"], start + start_win, stop + start_win
                    )
                    if overlap_percentage > 0:
                        overlap_flag = True
                        break

                if not overlap_flag:
                    orf_sequence = Seq(
                        record.sequence[start_win:end_win][start:stop]
                    )  # Extract the ORF sequence
                    if strand == "-":  # If the strand is negative
                        orf_sequence = (
                            orf_sequence.reverse_complement()
                        )  # Get the reverse complement
                    protein_sequence = orf_sequence.translate(
                        table=11, to_stop=True
                    )  # Translate the DNA to protein

                    new_genes.append(
                        {
                            "seqid": nucleotide_id,
                            "source": "orfipy",
                            "type": "CDS",
                            "start": start + start_win,
                            "end": stop + start_win,
                            "score": ".",
                            "strand": "-" if strand == "-" else "+",
                            "phase": ".",
                            "protein_id": f"sORF_orfipy_{unique_id}_{i}",
                            "id": f"sORF_orfipy_{unique_id}_{i}",
                            "sequence": protein_sequence,
                        }
                    )

            if new_genes:
                new_genes_df = pl.DataFrame(new_genes)
                subgff = pl.concat([subgff, new_genes_df], how="vertical")

        neighborhood = pl.DataFrame(neighborhood, index=[0])

    elif gff_file and faa_file:

        # print arguments
        console.print(f"✔️\tExtracting neighborhood {unique_id}")
        # print unique id and nucleotide_id and protein_id
        gff_header = [
            "seqid",
            "source",
            "type",
            "start",
            "end",
            "score",
            "strand",
            "phase",
            "attributes",
        ]
        # Check if GFF and FAA files exist
        if not os.path.exists(gff_file):
            return (
                None,
                None,
                unique_id,
                "GFF file not found",
            )  # Return None and index for failed extractions
        if not os.path.exists(faa_file):
            return None, None, unique_id, "FAA file not found"

        # Read GFF and FAA files
        try:
            gff = pl.read_csv(
                filepath_or_buffer=gff_file,
                separator="\t",
                comment="#",
                names=gff_header,
                engine="c",
            )
        except:
            return None, None, unique_id, "Failed to read GFF file"
        try:
            faa_df = read_fasta(faa_file)
        except:
            return None, None, unique_id, "Failed to read FAA file"

        # Implement the logic to process the GFF and FAA files
        flip = False
        if input_type == "protein":
            if protein_id and window:
                query = f"={protein_id}"
                start = gff[gff["attributes"].str.contains(query)]["start"].to_list()
                if not start:
                    return None, None, unique_id, "Protein not found in GFF file"
                else:
                    start = start[0]
                end = gff[gff["attributes"].str.contains(query)]["end"].to_list()[0]
                strand = gff[gff["attributes"].str.contains(query)]["strand"].to_list()[0]
                nucleotide_id = gff[gff["attributes"].str.contains(query)]["seqid"].to_list()[0]

                if mode == "win_nts":
                    start_win = start - window
                    end_win = end + window
                    if start_win < 0:
                        start_win = 0
                    gff_nuc = gff.query("seqid == @nucleotide_id")
                    if end_win > gff_nuc["end"].max():
                        end_win = gff_nuc["end"].max()
                    subgff = gff.query(
                        "seqid == @nucleotide_id & type =='CDS' & start>=@start_win & end<=@end_win"
                    )
                elif mode == "win_ngen":
                    subgff = gff.query("seqid == @nucleotide_id & type =='CDS'").reset_index(
                        drop=True
                    )
                    prot_index = subgff[subgff["attributes"].str.contains(query)].index.to_list()[0]
                    subgff = subgff[prot_index - window : prot_index + window]
                    start_win = subgff["start"].min()
                    end_win = subgff["end"].max()

                if strand == "-":
                    flip = True

        elif input_type == "nucleotide":
            if nucleotide_id and (start and end) and window:
                start, end = int(start), int(end)
                if not strand:
                    strand = "-" if end < start else "+"
                target_nuc = nucleotide_id

                if mode == "win_nts":
                    start_win = start - window
                    end_win = end + window
                    if start_win < 0:
                        start_win = 0
                    subgff = gff.query(
                        "seqid == @nucleotide_id & type =='CDS' & start>=@start_win & end<=@end_win"
                    )
                elif mode == "win_ngen":
                    subgff = gff.query("seqid == @nucleotide_id & type =='CDS'").reset_index(
                        drop=True
                    )
                    prot_index = subgff[subgff["attributes"].str.contains(query)].index.to_list()[0]
                    subgff = subgff[prot_index - window : prot_index + window]
                    start_win = subgff["start"].min()
                    end_win = subgff["end"].max()

                if strand == "-":
                    flip = True

            elif not window and nucleotide_id and (start and end):
                start, end = int(start), int(end)
                if not strand:
                    strand = "-" if end < start else "+"
                subgff = gff.query(
                    "seqid == @nucleotide_id & type =='CDS' & start>=@start & end<=@end"
                )
                start_win = start
                end_win = end
                if strand == "-":
                    flip = True

            elif not window and nucleotide_id and not (start and end):
                start = end = 0
                subgff = gff.query("seqid == @nucleotide_id & type =='CDS'")
                flip = False
                start_win = 0
                end_win = subgff["end"].max()
                strand = "+"

            elif window and nucleotide_id and not (start and end):
                start = end = 0
                subgff = gff.query("seqid == @nucleotide_id & type =='CDS'")
                flip = False
                start_win = 0
                end_win = subgff["end"].max()
                strand = "+"

            else:
                return (
                    None,
                    None,
                    unique_id,
                    "Invalid usage of parameters",
                )  # Invalid usage of parameters, return None and index

        # Process the attributes in GFF and merge with FAA
        subgff = unwrap_attributes(subgff)
        if "protein_id" in subgff.columns:
            key_join = "protein_id"
        else:
            key_join = "ID"

        subgff = subgff.join(
            faa_df[["id", "sequence"]], left_on=key_join, right_on="id", how="left"
        )

        if fna_file:
            fna_df = read_fasta(fna_file)
            nucleotide_id = str(nucleotide_id)
            faa_df["id"] = faa_df["id"].astype(str)
            # check if nucleotide id in fna_df["id"]
            if nucleotide_id in fna_df["id"].to_list():
                sequence = fna_df[fna_df["id"] == nucleotide_id]["sequence"].to_list()[0]
                end_win = end + window
                if end_win > len(sequence):
                    end_win = len(sequence)
                if sorfs:
                    # annotate genes in the neighborhood with pyrodigal:
                    orf_finder = pyrodigal.GeneFinder(meta=True)
                    new_genes = []

                    for i, pred in enumerate(orf_finder.find_genes(sequence.encode())):
                        overlap_flag = False
                        for row in subgff.iter_rows(named=True):
                            overlap_percentage = calculate_overlap(
                                row["start"], row["end"], pred.begin, pred.end
                            )
                            if overlap_percentage > 5:
                                overlap_flag = True
                                break

                        if not overlap_flag:
                            new_genes.append(
                                {
                                    "seqid": nucleotide_id,
                                    "source": "pyrodigal",
                                    "type": "CDS",
                                    "start": pred.begin + start_win,
                                    "end": pred.end + start_win,
                                    "score": pred.score,
                                    "strand": "-" if pred.strand == "-1" else "+",
                                    "phase": ".",
                                    key_join: f"{key_join}=sORF_{unique_id}_{i}",
                                    "sequence": pred.translate(),  # Assuming you want to store the translated protein sequence
                                }
                            )

                    # Convert new genes to DataFrame and concatenate with existing subgff
                    if new_genes:
                        new_genes_df = pl.DataFrame(new_genes)
                        subgff = pl.concat([subgff, new_genes_df], how="vertical")
        else:
            # wnd win should be the end of the last ORF in the window
            sequence = None
        neighborhood = {
            "seqid": nucleotide_id,
            "start_target": start,
            "end_target": end,
            "start_win": start_win,
            "end_win": end_win,
            "strand_win": strand,
            "sequence": sequence[start_win:end_win],
            "unique_id": unique_id,
        }
        neighborhood = pl.DataFrame(neighborhood, index=[0])

    subgff["target_prot"] = protein_id
    subgff["target_nuc"] = nucleotide_id
    subgff["unique_id"] = str(unique_id)

    # Normalize identifier columns so callers always see a single canonical 'id'
    try:
        if isinstance(subgff, pl.DataFrame):
            if "id" not in subgff.columns:
                for cand in ("protein_id", "ID", "gene_id"):
                    if cand in subgff.columns:
                        subgff["id"] = subgff[cand]
                        break

            for redundant in ("ID", "gene_id"):
                if redundant in subgff.columns and redundant != "id":
                    try:
                        subgff.drop([redundant])
                    except Exception:
                        pass

            if "id" in subgff.columns:
                try:
                    subgff["id"] = subgff["id"].astype(str)
                except Exception:
                    pass
    except Exception:
        pass

    if "product" not in subgff.columns:
        subgff["product"] = None

    return subgff, neighborhood, unique_id
