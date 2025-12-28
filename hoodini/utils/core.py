# Utility: check if nuccore ID is RefSeq or GenBank/INSDC
def is_refseq_nuccore(nuc_id):
    """
    Return True if the nuccore accession is a RefSeq accession (NC_, NZ_, NM_, NR_, XM_, XR_, AP_, YP_, XP_, WP_, etc.), else False (GenBank/INSDC).
    """
    refseq_prefixes = (
        "NC_", "NZ_", "NM_", "NR_", "XM_", "XR_", "AP_", "YP_", "XP_", "WP_"
    )
    return isinstance(nuc_id, str) and nuc_id.startswith(refseq_prefixes)

# Utility: switch GCA_ <-> GCF_ for an assembly accession
def switch_assembly_prefix(asm_id):
    if not isinstance(asm_id, str):
        return asm_id
    if asm_id.startswith("GCA_"):
        return "GCF_" + asm_id[4:]
    elif asm_id.startswith("GCF_"):
        return "GCA_" + asm_id[4:]
    return asm_id
from enum import unique
import xml.etree.ElementTree as ET
import polars as pl
import os
import re
import rich_click as click
import itertools
import ete3
import requests
import concurrent.futures
import numpy as np
import glob
from Bio import SeqIO
import subprocess
from networkx.utils.union_find import UnionFind
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
import pyhmmer
from functools import partial
from tqdm import tqdm
from rich.progress import Progress
import sqlite3
import csv
from dataclasses import dataclass
from typing import Optional
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from UniProtMapper import ProtMapper
import time
import sys
import xml.etree.ElementTree as ET
import polars as pl
import taxoniq 
import gb_io
import pyrodigal
import orfipy_core
from Bio.Seq import Seq

from rich.console import Console
from rich.logging import RichHandler
import logging

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")
console = Console()

def validate_input_file(ctx, param, value):
    """Validate the input file based on its type, either single-column or TSV format."""
    if value is None:
        return None  # Skip validation if no file path is provided.

    if not os.path.isfile(value):
        raise click.BadParameter(f"File '{value}' does not exist.")

    try:
        with open(value, 'r') as file:
            if param.name == 'input_path':
                lines = [line.strip() for line in file.readlines() if line.strip()]
                if len(lines) <= 1:
                    raise click.BadParameter("File must contain multiple lines.")

                for line in lines:
                    if ',' in line or '\t' in line:
                        raise click.BadParameter("File must be a single-column text file without delimiters like commas or tabs.")

                special_char_pattern = re.compile(r'[^A-Za-z0-9\s._]+')
                for i, line in enumerate(lines):
                    match = special_char_pattern.search(line)
                    if match and not line.startswith("IMG"):
                        raise click.BadParameter(f"Invalid character '{match.group()}' found in line {i+1}: \"{line}\"")

            elif param.name == 'inputsheet':
                first_line = file.readline()
                delimiter = '\t' if '\t' in first_line else None
                if delimiter is None:
                    raise click.BadParameter("The file is not in TSV format.")
                file.seek(0)  # Reset the file pointer to the beginning to read with csv.DictReader
                reader = csv.DictReader(file, delimiter=delimiter)
                required_columns = ['nucleotide_id', 'protein_id', 'gff_path', 'fna_path', 'faa_path']
                if not all(col in reader.fieldnames for col in required_columns):
                    raise click.BadParameter("TSV file must contain columns: nucleotide_id, protein_id, gff_path, fna_path, faa_path")
                
                # Ensure there's at least one row and it has valid content
                found_valid_row = False
                for row in reader:
                    if (row['nucleotide_id'].strip() or row['protein_id'].strip()):
                        found_valid_row = True
                        break
                if not found_valid_row:
                    raise click.BadParameter("The TSV file must contain at least one valid row with required data.")

    except Exception as e:
        raise click.BadParameter(f"Error reading file: {e}")

    return value


def validate_domains(ctx, param, value):
    """Validate domain database names against MetaCerberus availability."""
    import click
    # Treat None or empty/blank string as no domains requested (optional)
    if not value or (isinstance(value, str) and value.strip() == ""):
        return None
    
    # Basic format validation - should be comma-separated database names
    db_names = [d.strip().lower() for d in value.split(",") if d.strip()]
    if not db_names:
        raise click.BadParameter("Domain parameter must contain at least one database name.")
    
    # Validate database names contain only alphanumeric characters and underscores
    import re
    for db in db_names:
        if not re.match(r'^[a-zA-Z0-9_-]+$', db):
            raise click.BadParameter(f"Invalid database name '{db}'. Only letters, numbers, underscores, and hyphens are allowed.")
    
    # Now validate against MetaCerberus availability
    try:
        from hoodini.download.metacerberus import list_db_files, get_db_groups, check_downloaded
        from importlib.resources import files
        
        # Get available databases
        files_list = list_db_files()
        groups = get_db_groups(files_list)
        status = check_downloaded(groups)
        
        # Check each requested database
        valid_dbs = []
        invalid_dbs = []
        missing_files_dbs = []
        
        for db in db_names:
            if db not in groups:
                invalid_dbs.append(db)
                continue
            
            # Check if both HMM and TSV files are present
            file_statuses = status.get(db, [])
            hmm_present = any(present for f, present in file_statuses if f["name"].endswith(".hmm.gz"))
            tsv_present = any(present for f, present in file_statuses if f["name"].endswith(".tsv"))
            
            # Some databases don't have HMM files (like FOAM, KEGG)
            has_hmm_file = any(f["name"].endswith(".hmm.gz") for f, _ in file_statuses)
            has_tsv_file = any(f["name"].endswith(".tsv") for f, _ in file_statuses)
            
            if has_tsv_file and not tsv_present:
                missing_files_dbs.append(f"{db} (missing TSV)")
            elif has_hmm_file and not hmm_present:
                missing_files_dbs.append(f"{db} (missing HMM)")
            elif not has_hmm_file and not has_tsv_file:
                invalid_dbs.append(db)
            elif (has_hmm_file and hmm_present and has_tsv_file and tsv_present) or (not has_hmm_file and has_tsv_file and tsv_present):
                valid_dbs.append(db)
            else:
                missing_files_dbs.append(db)
        
        # Report errors
        if invalid_dbs:
            raise click.BadParameter(f"Unknown MetaCerberus databases: {', '.join(invalid_dbs)}. Run 'hoodini download metacerberus' to see available databases.")
        
        if missing_files_dbs:
            raise click.BadParameter(f"MetaCerberus databases not downloaded: {', '.join(missing_files_dbs)}. Run 'hoodini download metacerberus {','.join(db_names)}' to download them.")
        
        if not valid_dbs:
            raise click.BadParameter("No valid MetaCerberus databases found.")
        
        # Return only valid databases as a list (not comma-separated string)
        return valid_dbs
        
    except ImportError as e:
        # Fallback validation if metacerberus module not available
        console.print(f"[yellow]Warning: Could not validate MetaCerberus databases: {e}[/yellow]")
        return value
    except Exception as e:
        # Don't fail completely on validation errors, but warn
        console.print(f"[yellow]Warning: Could not validate MetaCerberus databases: {e}[/yellow]")
        return value


def categorize_protein_ids(ids):
    """
    Categorize accession IDs into NCBI nucleotide IDs, NCBI protein IDs, UniProt IDs, IMG IDs, and unmatched IDs.

    Parameters:
        ids (list): A list of accession ID strings.

    Returns:
        tuple: Five lists containing NCBI nucleotide IDs, NCBI protein IDs, UniProt IDs, IMG IDs, and unmatched IDs respectively.
               Returns None for any empty list.
    """
    import re

    ncbi_nucleotide_ids = []
    ncbi_protein_ids = []
    uniprot_ids = []
    img_ids = []
    unmatched_ids = []

    # Regular expressions for matching ID formats

    # UniProt IDs (6 or 10 characters)
    uniprot_pattern = re.compile(
        r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|'          # Old format: e.g., P12345
        r'[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]'     # New format: e.g., A0A0B4J2D5
        r'([A-Z][A-Z0-9]{2}[0-9])?)$'             # Optional additional segment for 10-char IDs
    )

    # IMG IDs
    img_pattern = re.compile(r'^(IMGVR|IMGPR)\S*$')

    # NCBI nucleotide IDs
    nucleotide_prefixes = [
        'NC', 'NG', 'NM', 'NR', 'NT', 'NW', 'NZ', 'AC', 'AP',
        'MT', 'PP', 'OR', 'OZ', 'LR', 'LN', 'KX'
    ]
    nucleotide_pattern = re.compile(
        r'^(' + '|'.join(nucleotide_prefixes) + r')[_\d]\d+(\.\d+)?(:\d+-\d+)?$'
    )
    genbank_nuc_pattern = re.compile(r'^[A-Z]{1,2}\d{5,8}(\.\d+)?$')

    # WGS accession numbers
    wgs_pattern = re.compile(
        r'^('
        r'[A-Z]{4}\d{8}'          # Option 1: 4 letters + 8 digits
        r'|'
        r'[A-Z]{6}\d{9,}'         # Option 2: 6 letters + 9 or more digits
        r')'
        r'(\.\d+)?'               # Optional version number applies to both options
        r'$'
    )
    # NCBI protein IDs
    protein_prefixes = ['NP', 'XP', 'YP', 'WP', 'ZP']
    protein_pattern = re.compile(
        r'^(' + '|'.join(protein_prefixes) + r')_\d+(\.\d+)?$'
    )
    genbank_prot_pattern = re.compile(r'^[A-Z]{3}\d{5}(\.\d+)?$')
    protein_no_prefix_pattern = re.compile(r'^[A-Z]{3}\d{4,8}(\.\d+)?$')

    for id_ in ids:
        if (re.match(nucleotide_pattern, id_) or
            re.match(genbank_nuc_pattern, id_) or
            re.match(wgs_pattern, id_)):
            ncbi_nucleotide_ids.append(id_)
        elif (re.match(protein_pattern, id_) or
              re.match(genbank_prot_pattern, id_) or
              re.match(protein_no_prefix_pattern, id_)):
            ncbi_protein_ids.append(id_)
        elif re.match(uniprot_pattern, id_):
            uniprot_ids.append(id_)
        elif re.match(img_pattern, id_):
            img_ids.append(id_)
        else:
            unmatched_ids.append(id_)

    # Replace empty lists with None
    ncbi_nucleotide_ids = ncbi_nucleotide_ids or None
    ncbi_protein_ids = ncbi_protein_ids or None
    uniprot_ids = uniprot_ids or None
    img_ids = img_ids or None
    unmatched_ids = unmatched_ids or None

    return ncbi_nucleotide_ids, ncbi_protein_ids, uniprot_ids, img_ids, unmatched_ids
# Parsing data

import re


def categorize_id(id_):
    # Split the id_ on ':' if present (for nucleotide_id:protein_id cases)
    parts = id_.split(':')
    id_part = parts[0]  # Use the first part for categorization

    # Define patterns for UniProt, IMG, NCBI nucleotide, and protein IDs
    uniprot_pattern = re.compile(
        r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|'
        r'[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]'
        r'(?:[A-Z][A-Z0-9]{2}[0-9])?)$'
    )
    img_pattern = re.compile(r'^(IMGVR|IMGPR)\S*$')
    nucleotide_patterns = [
        re.compile(r'^(' + '|'.join(['NC', 'NG', 'NM', 'NR', 'NT', 'NW', 'NZ', 'AC', 'AP', 'MT', 'PP', 'OR', 'OZ', 'LR', 'LN', 'KX']) + r')(_[A-Z]+\d+|\d+)(\.\d+)?(:\d+-\d+)?$'),
        re.compile(r'^[A-Z]{1,2}\d{5,8}(\.\d+)?$'),
        re.compile(r'^[A-Z]{4,6}\d{8,}(\.\d+)?$')
    ]
    protein_patterns = [
        re.compile(r'^(' + '|'.join(['NP', 'XP', 'YP', 'WP', 'ZP']) + r')_\d+(\.\d+)?$'),
        re.compile(r'^[A-Z]{3}\d{5,8}(\.\d+)?$')
    ]

    # Matching against the patterns
    if re.match(uniprot_pattern, id_part):
        return {'type': 'uniprot', 'id': id_part, 'protein_id': None}
    elif re.match(img_pattern, id_part):
        return {'type': 'img', 'id': id_part, 'protein_id': None}
    elif any(re.match(pattern, id_part) for pattern in nucleotide_patterns):
        return {
            'type': 'nucleotide',
            'id': id_part,
            'protein_id': parts[1] if len(parts) > 1 else None
        }
    elif any(re.match(pattern, id_part) for pattern in protein_patterns):
        return {'type': 'protein', 'id': id_part, 'protein_id': None}   
    else:
        return {'type': 'unmatched', 'id': id_part, 'protein_id': None}

def read_input_list(filename):
    """
    Reads input from a list (e.g., from a text file) and returns a pandas DataFrame.
    """
    input_list = Path(filename).read_text().splitlines()
    data = []
    for index, id_ in enumerate(input_list):
        if not id_ or id_.strip() == '':
            continue  # Skip empty lines
        
        category = categorize_id(id_)
        record = {
            'og_index': index,
            'protein_id': None,
            'nucleotide_id': None,
            'uniprot_id': None,
            'img': None,
            'failed': None,
            'gff_path': None,
            'faa_path': None,
            'fna_path': None,
            'strand': None,
            'start': None,
            'end': None,
            'gbf_path': None,
            'taxid': None,
            'assembly_id': None,
            'input_type': None,
            'premade': None
        }
        
        # Assign IDs based on category
        if category['type'] == 'protein':
            record['protein_id'] = category['id']
            record['input_type'] = "protein"
        elif category['type'] == 'nucleotide':
            record['nucleotide_id'] = category['id']
            record['protein_id'] = category.get('protein_id')
            record['input_type'] = "nucleotide"
        elif category['type'] == 'uniprot':
            record['uniprot_id'] = category['id']
            record['input_type'] = "protein"
        elif category['type'] == 'img':
            record['img'] = True
            if "|" in category["id"]:
                record["protein_id"] = category["id"]
                record['input_type'] = "protein"
            else:
                record['nucleotide_id'] = category['id']
                record['input_type'] = "nucleotide"
        elif category['type'] == 'unmatched':
            record['failed'] = "not valid ID"
        else:
            if ":" in id_:
                
                #try to categorize the first part of the id and check if its a nucleotide id or img id
                category = categorize_id(id_.split(":")[0])
                if category['type'] == 'nucleotide' or (category['type'] == 'img' and "|" in category["id"]):
                    #if the second part contains two numbers separated by a dash:
                    if "-" in id_.split(":")[1] and id_.split(":")[1].split("-")[0].isdigit() and id_.split(":")[1].split("-")[1].isdigit():
                        record['nucleotide_id'] = id_.split(":")[0]
                        record['start'] = id.split(":")[1].split("-")[0]
                        record['end'] = id.split(":")[1].split("-")[1]
                        if record["start"] > record["end"]:
                            record["start"], record["end"] = record["end"], record["start"]
                            record["strand"] = "-"
                        else:
                            record["strand"] = "+"
                        record['input_type'] = "nucleotide"
                    else:
                        record['failed'] = "not valid ID"
                    record['nucleotide_id'] = category['id']
                    record['input_type'] = "nucleotide"
                    
            # For unmatched or other types
            record['protein_id'] = category['id']
            record['input_type'] = "protein"
        data.append(record)
    df = pl.DataFrame(data)
    return df

# Function to read input from a TSV file (e.g., with multiple columns)
def read_input_sheet(filename):
    df = pl.read_csv(filename, separator='\t', dtype=str)
    df = df.with_row_count('og_index').with_columns(pl.col('og_index').cast(pl.Utf8))

    # Initialize missing columns
    expected_columns = [
        'protein_id', 'nucleotide_id', 'uniprot_id', 'img', 'gff_path',
        'faa_path', 'fna_path', 'gbf_path', 'taxid', 'assembly_id', 'failed', 'input_type', 'premade'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
            
    # Apply categorization to 'protein_id' column
    def categorize_row(row):
        protein_id = row.get('protein_id', None)
        nucleotide_id = row.get('nucleotide_id', None)
        if protein_id:
            category = categorize_id(protein_id)
            row['input_type'] = "protein"
            if category['type'] == 'unmatched':
                if not ((row["gff_path"] and row["faa_path"]) or ( row["gbf_path"])):
                    row['failed'] = "not valid ID"
            elif category['type'] == 'protein':
                row['protein_id'] = category['id']
            elif category['type'] == 'nucleotide':
                row['failed'] = "ID provided is not a protein ID but a nucleotide ID"
                row['nucleotide_id'] = category['id']
                row['protein_id'] = category.get('protein_id')
                row['input_type'] = "nucleotide"
            elif category['type'] == 'uniprot':
                row['uniprot_id'] = category['id']
                row['protein_id'] = None  # Clear protein_id if it's a uniprot_id
            elif category['type'] == 'img':
                #if category_id
                row['img'] = True
        else:
            row['protein_id'] = None
            
        if nucleotide_id and not protein_id:
            category = categorize_id(nucleotide_id)
            row['input_type'] = "nucleotide"
            if category['type'] == 'unmatched':
                if not ((row["gff_path"] and row["faa_path"]) or ( row["gbf_path"])):
                    row['failed'] = "not valid ID"
            elif category['type'] == 'nucleotide':
                row['nucleotide_id'] = category['id']
                row['input_type'] = "nucleotide"   
            elif category['type'] == 'protein':
                row['failed'] = "ID provided is not a nucleotide ID but a protein ID"
                row['protein_id'] = category['id']
                row['nucleotide_id'] = category.get('nucleotide_id')
                row['input_type'] = "protein"
            elif category['type'] == 'uniprot':
                row['failed'] = "ID provided is not a nucleotide ID but a uniprot ID"
                row['uniprot_id'] = category['id']
                row['nucleotide_id'] = category.get('nucleotide_id')
            elif category['type'] == 'img':
                if "|" in category["id"]:
                    row['failed'] = "ID provided is not a nucleotide ID but a protein ID"
                    row['protein_id'] = category['id']
                    row['nucleotide_id'] = category.get('nucleotide_id')
                row['img'] = True
                
        elif nucleotide_id:
            category = categorize_id(nucleotide_id)
            if category['type'] == 'unmatched':
                if not ((row["gff_path"] and row["faa_path"]) or ( row["gbf_path"])):
                    row['failed'] = "not valid ID"
                
        else:
            row['nucleotide_id'] = None
                

        return row
    

    df = df.apply(categorize_row, how="horizontal")
    # mark IMG/VR/PR inputs as IMG
    df = df.with_columns(
        pl.when(pl.col("nucleotide_id").str.starts_with(("IMGVR", "IMGPR")).fill_null(False))
        .then(pl.lit(True))
        .otherwise(pl.col("img"))
        .alias("img")
    )
    
    return df

def uniprot2ncbi(df_records):
    """
    Maps 'uniprot_id's to 'protein_id's for records that have 'uniprot_id' but no 'protein_id',
    and do not have 'gff_path' and 'faa_path' (i.e., no local files).
    For 'uniprot_id's that map to multiple 'protein_id's, creates multiple records.
    For failed mappings, updates the 'failed' column with the message 'No associated NCBI found for the Uniprot entry'.
    
    Parameters:
        df_records (pl.DataFrame): The input DataFrame containing the records.
        
    Returns:
        pl.DataFrame: The modified DataFrame with 'protein_id's filled in where possible.
    """

    records_to_map = df_records.filter(
        pl.col('uniprot_id').is_not_null() &  # UniProt ID must be present
        pl.col('protein_id').is_null() &   # No protein ID yet
        pl.col('gff_path').is_null() &     # No gff_path
        pl.col('faa_path').is_null()       # No faa_path
    )
    
    if records_to_map.height == 0:
        return df_records

    # Extract unique uniprot_ids to map
    uniprot_ids_to_map = records_to_map['uniprot_id'].unique().to_list()

    # Perform the UniProt to NCBI mapping using ProtMapper
    mapper = ProtMapper()
    try:
        uniprot2ncbi_df, failed_uniprot_ids = mapper.get(
            ids=uniprot_ids_to_map,
            from_db="UniProtKB_AC-ID",
            to_db="EMBL-GenBank-DDBJ_CDS"
        )
    except:
        print("Failed to map Uniprot IDs to NCBI protein IDs.")
        failed_uniprot_ids = uniprot_ids_to_map
        failed_mask = df_records['uniprot_id'].isin(failed_uniprot_ids)
        df_records.loc[failed_mask, 'failed'] = "No associated NCBI found for the Uniprot entry."
        return df_records

    if uniprot2ncbi_df.height > 0:
        # Merge the mapping results directly into df_records
        df_records = df_records.join(
            uniprot2ncbi_df[['From', 'To']],
            left_on='uniprot_id',
            right_on='From',
            how='left'
        )

        # Update protein_id with the mapped values
        df_records['protein_id'].fillna(df_records['To'])

        # Drop the extra 'From' and 'To' columns after merging
        df_records.drop(['From', 'To'])
    
    # Handle failed mappings
    if failed_uniprot_ids:
        # Set 'failed' column for uniprot_ids that failed to map
        failed_mask = df_records['uniprot_id'].isin(failed_uniprot_ids)
        df_records.loc[failed_mask, 'failed'] = "No associated NCBI found for the Uniprot entry."

    return df_records

def unwrap_attributes(df):
    attribute_names = set()
    for attributes in df['attributes']:
        attributes_list = attributes.split(';')
        for attr in attributes_list:
            attribute_name = attr.split('=')[0]
            attribute_names.add(attribute_name)

    # Create new columns with attribute names and default values
    for attribute_name in attribute_names:
        df = df.assign(**{attribute_name: None})

    # Function to extract attribute values and assign them to columns
    def extract_attribute_values(row):
        attributes_list = row['attributes'].split(';')
        for attr in attributes_list:
            if "=" in attr:
                attribute_name, attribute_value = attr.split('=')
                row[attribute_name] = attribute_value
        return row

    # Apply the function to each row
    df = df.apply(lambda row: extract_attribute_values(row), how="horizontal")

    # Drop the original "attributes" columns
    df = df.drop(['attributes'])
    return df

def read_fasta(filename):
    with open(filename, "r") as file:
        records = file.read().split(">")[1:] # skip the first empty split
        records = [record.split("\n", 1) for record in records]
        records = [(t[0].split(" ")[0], "".join(t[1].split())) for t in records]
    return pl.DataFrame(records, columns=["id", "sequence"])

def to_fasta(df, id_col, seq_col, output_file):
    with open(output_file, 'w') as f:
        for row in df.select([id_col, seq_col]).iter_rows(named=True):
            seq_id = row[id_col]
            sequence = row[seq_col]
            if sequence is None:
                continue
            f.write(f'>{seq_id}\n')
            f.write(f'{sequence}\n')
# PandasObject.to_fasta = to_fasta  # Removed: migrated to polars

def process_features(features, record_accession):
    """Function to process features and extract data."""
    data = []
    for feature in features:
        if feature.kind == 'CDS':
            location = feature.location
            qualifiers = {q.key: q.value for q in feature.qualifiers}
            protein_id = qualifiers.get('protein_id', '')
            #location can be Range(1793069, 1794317), Complement(1793069, 1794317), or Join([Range(5277605, 5277702), Range(0, 431)]), please handle all cases. 
            # when is Range(1793069, 1794317), the strand is "+", start is 1793069, end is 1794317
            #when is Complement(1793069, 1794317), the strand is "-", start is 1793069, end is 1794317
            #when is Join([Range(5277605, 5277702), Range(0, 431)]), the strand is "+", start is 5277605, end is 43
            class_type = type(location).__name__
            
            if class_type == "Range":
                start = location.start
                end = location.end
                strand = "+"
            elif class_type == "Complement":
                start = location.start
                end = location.end
                strand = "-"
            elif class_type == "Join":
                start = location.locations[0].start
                end = location.locations[0].end
                if location.locations[-1].start in list(range(end-5, end+6,1)):                 
                    start = location.locations[0].start
                    end = location.locations[-1].end
                if type(location.locations[0]).__name__ == "Complement":
                    strand = "-"
                else:
                    strand = "+"
            else:
                continue
                    
            if end<start:
                start, end = end, start
                strand = "-"
                
            if not protein_id:
                continue
            data.append({
                'seqid': record_accession,
                'source': "hoodini",
                'type': 'CDS',
                'start': start,
                'end': end,
                'score': '.',
                'strand': strand,
                'phase': '.',
                'attributes': qualifiers,
                'protein_id': qualifiers.get('protein_id', ''),
            })
    return data

def calculate_overlap(coord1A, coord1B, coord2A, coord2B):
    # Sort each set of coordinates to ensure the start is smaller than the end
    start1, end1 = sorted([coord1A, coord1B])
    start2, end2 = sorted([coord2A, coord2B])
    
    # Find the max of starts and the min of ends
    max_start = max(start1, start2)
    min_end = min(end1, end2)
    
    # Calculate the actual overlap
    overlap = max(0, min_end - max_start)
    
    # Calculate the length of the second interval
    length_second_interval = end2 - start2
    
    # Calculate overlap as a percentage of the length of the second interval
    if length_second_interval == 0:
        return 0  # To avoid division by zero
    overlap_percentage = (overlap / length_second_interval) * 100
    
    return overlap_percentage

#add all imports
import os

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
            return None, None, unique_id, "GenBank file not found"
        # Load the GenBank records
        try:
            records = gb_io.load(gbf_file)
        except:
            return None, None, unique_id, "gb_io failed to load GenBank file"
        
        if nucleotide_id:
            # Process all CDS features in the record(s)
            record_found = False
            for record in records:
                #iterate over records and stop when record version matches nucleotide_id
                record_version = getattr(record, 'version', None)
                if nucleotide_id in record.version:
                    record_found = True
                    break
        if record_found:
            feature_data = process_features(record.features, record_version)
            feature_data = pl.DataFrame(feature_data)
            if "attributes" in feature_data.columns:
                attributes_df = pl.DataFrame(feature_data['attributes'].to_list())
                attributes_df = attributes_df.drop('protein_id')
                feature_data = pl.concat([feature_data.drop('attributes'), attributes_df], how="horizontal")
                feature_data = feature_data.rename({"translation":"sequence"})
            else:
                return None, None, unique_id, "GenBank file is not annotated"
        else:
            return None, None, unique_id, "GenBank record not found"  
                       
        if input_type == "protein":
                        
            if "protein_id" in feature_data.columns:
                if not (start and end):
                    # get the start from the row in which protein_id matches the input protein_id
                    match_row = feature_data.filter(pl.col("protein_id") == protein_id)
                    if match_row.height > 0:
                        first = match_row.row(0, named=True)
                        start = first["start"]
                        end = first["end"]
                        strand = first["strand"]
                    else:
                        return None, None, unique_id, "Protein ID not found in features"
            start, end = int(start), int(end)
            
            if mode == "win_nts": 
                start_win = start - window
                end_win = end + window
                if start_win < 0:
                    start_win = 0
                if end_win > len(record.sequence):
                    end_win = len(record.sequence)
                subgff = feature_data.filter((pl.col("start") >= start_win) & (pl.col("end") <= end_win))
                
            elif mode == "win_ngen":
                # Add row index for slicing
                feature_indexed = feature_data.with_row_count("_idx")
                prot_match = feature_indexed.filter(pl.col("protein_id") == protein_id)
                if prot_match.height == 0:
                    return None, None, unique_id, "Protein ID not found in features for win_ngen"
                prot_index = prot_match.row(0, named=True)["_idx"]
                start_idx = max(0, prot_index - window)
                end_idx = min(feature_indexed.height, prot_index + window + 1)
                subgff = feature_indexed.slice(start_idx, end_idx - start_idx).drop("_idx")
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
                    subgff = feature_data.filter((pl.col("start") >= start_win) & (pl.col("end") <= end_win))
                    
                elif mode == "win_ngen":
                    feature_indexed = feature_data.with_row_count("_idx")
                    start_matches = feature_indexed.filter(pl.col("start") >= start)
                    end_matches = feature_indexed.filter(pl.col("end") <= end)
                    if start_matches.height == 0 or end_matches.height == 0:
                        return None, None, unique_id, "No features found in specified range"
                    start_index = start_matches.row(0, named=True)["_idx"]
                    end_index = end_matches.row(-1, named=True)["_idx"]
                    slice_start = max(0, start_index - window)
                    slice_end = min(feature_indexed.height, end_index + window + 1)
                    subgff = feature_indexed.slice(slice_start, slice_end - slice_start).drop("_idx")
                    start_win = subgff["start"].min()
                    end_win = subgff["end"].max()
                    if not strand:
                        strand = "+" if end > start else "-"
                    
            elif not window and nucleotide_id and (start and end):
                start, end = int(start), int(end)
                if not strand:
                    strand = "-" if end < start else "+"
                subgff = feature_data.filter(
                    (pl.col("seqid") == nucleotide_id) & (pl.col("type") == "CDS") &
                    (pl.col("start") >= start) & (pl.col("end") <= end)
                )
                if subgff.height > 0:
                    start_win = subgff["start"].min()
                    end_win = subgff["end"].max()
                else:
                    start_win = start
                    end_win = end

            elif not (start and end):
                subgff = feature_data
                start_win = subgff["start"].min()
                end_win = subgff["end"].max()
                start = start_win
                end = end_win
                strand = "+"
        
        if start_win < 0:
            start_win = 0
        if end_win > len(record.sequence):
            end_win = len(record.sequence)
        subgff = subgff.with_columns(pl.col("protein_id").alias("id"))
        header = ['seqid', 'source', 'type','start','end','score','strand','phase','protein_id','id','sequence']
        if "product" in subgff.columns:
            header.append("product")
        else:
            subgff = subgff.with_columns(pl.lit(None).alias("product"))
        subgff = subgff.select([c for c in header if c in subgff.columns])
        neighborhood = {
            "seqid": record_version,
            "start_target": start,
            "end_target": end,
            "start_win": start_win,
            "end_win": end_win,
            "strand_win": strand,
            "sequence": record.sequence[start_win:end_win].decode("utf-8"),
            "unique_id": unique_id
            
        }
        if sorfs:
            #annotate genes in the neighborhood with pyrodigal:
            orf_finder = pyrodigal.GeneFinder(meta=True,min_gene=10, max_overlap=9)
            new_genes = []

            for i,pred in enumerate(orf_finder.find_genes(record.sequence[start_win:end_win].decode("utf-8"))):
                overlap_flag = False
                for row in subgff.iter_rows(named=True):
                    overlap_percentage = calculate_overlap(row['start'], row['end'], pred.begin+start_win, pred.end+start_win)
                    if overlap_percentage > 10:
                        overlap_flag = True
                        break
                if not overlap_flag:
                    new_genes.append({
                        'seqid': nucleotide_id,
                        'source': 'pyrodigal',
                        'type': 'CDS',
                        'start': pred.begin + start_win,
                        'end': pred.end + start_win,
                        'score': pred.score,
                        'strand': "-" if pred.strand == "-1" else "+",
                        'phase': '.',
                        "protein_id": f"sORF_{unique_id}_{i}",
                        "id": f"sORF_{unique_id}_{i}",
                        'sequence': pred.translate()  # Assuming you want to store the translated protein sequence
                    })
                    
            # Convert new genes to DataFrame and concatenate with existing subgff
            if new_genes:
                new_genes_df = pl.DataFrame(new_genes)
                subgff = pl.concat([subgff, new_genes_df], how="vertical") 
                
                
            new_genes = []
            seq = record.sequence[start_win:end_win].decode("utf-8").upper()
            for i, (start, stop, strand, description) in enumerate(orfipy_core.orfs(seq, minlen=100, maxlen=1000, partial3=False, between_stops=False)):
                overlap_flag = False
                for row in subgff.iter_rows(named=True):
                    overlap_percentage = calculate_overlap(row['start'], row['end'], start+start_win, stop+start_win)
                    if overlap_percentage > 0:
                        overlap_flag = True
                        break
                    
                    
                if not overlap_flag:
                    orf_sequence = Seq(record.sequence[start_win:end_win][start:stop])  # Extract the ORF sequence
                    if strand == '-':  # If the strand is negative
                        orf_sequence = orf_sequence.reverse_complement()  # Get the reverse complement
                    protein_sequence = orf_sequence.translate(table=11,to_stop=True)  # Translate the DNA to protein
                    
                    
                    new_genes.append({
                        'seqid': nucleotide_id,
                        'source': 'orfipy',
                        'type': 'CDS',
                        'start': start + start_win,
                        'end': stop + start_win,
                        'score': '.',
                        'strand': "-" if strand == "-" else "+",
                        'phase': '.',
                        "protein_id": f"sORF_orfipy_{unique_id}_{i}",
                        "id": f"sORF_orfipy_{unique_id}_{i}",
                        'sequence': protein_sequence
                    }) 
            
            if new_genes:
                new_genes_df = pl.DataFrame(new_genes)
                subgff = pl.concat([subgff, new_genes_df], how="vertical")
                
        neighborhood = pl.DataFrame(neighborhood, index=[0])
        
    elif gff_file and faa_file:
        
        #print arguments
        console.print(f"✔️\tExtracting neighborhood {unique_id}")
        #print unique id and nucleotide_id and protein_id        
        gff_header = ['seqid', 'source', 'type','start','end','score','strand','phase','attributes']
        # Check if GFF and FAA files exist
        if not os.path.exists(gff_file):
            return None, None, unique_id, "GFF file not found"  # Return None and index for failed extractions
        if not os.path.exists(faa_file):
            return None, None, unique_id, "FAA file not found"
        if not os.path.exists(fna_file) and fna_file:
            return None, None, unique_id, "FNA file not found"

        # Read GFF and FAA files
        try:
            gff = pl.read_csv(filepath_or_buffer=gff_file, separator="\t", comment="#", names=gff_header, engine="c")
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
                start = gff[gff['attributes'].str.contains(query)]["start"].to_list()
                if not start:
                    return None, None, unique_id, "Protein not found in GFF file"
                else:
                    start = start[0]
                end = gff[gff['attributes'].str.contains(query)]["end"].to_list()[0]
                strand = gff[gff['attributes'].str.contains(query)]["strand"].to_list()[0]
                nucleotide_id = gff[gff['attributes'].str.contains(query)]["seqid"].to_list()[0]
                
                if mode == "win_nts":
                    start_win = start - window
                    end_win = end + window
                    if start_win < 0:
                        start_win = 0
                    gff_nuc = gff.query("seqid == @nucleotide_id")
                    if end_win > gff_nuc["end"].max():
                        end_win = gff_nuc["end"].max()
                    subgff = gff.query("seqid == @nucleotide_id & type =='CDS' & start>=@start_win & end<=@end_win")
                elif mode == "win_ngen":
                    subgff = gff.query("seqid == @nucleotide_id & type =='CDS'").reset_index(drop=True)
                    prot_index = subgff[subgff['attributes'].str.contains(query)].index.to_list()[0]
                    subgff = subgff[prot_index-window:prot_index+window]
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
                    subgff = gff.query("seqid == @nucleotide_id & type =='CDS' & start>=@start_win & end<=@end_win")
                elif mode == "win_ngen":
                    subgff = gff.query("seqid == @nucleotide_id & type =='CDS'").reset_index(drop=True)
                    prot_index = subgff[subgff['attributes'].str.contains(query)].index.to_list()[0]
                    subgff = subgff[prot_index-window:prot_index+window]
                    start_win = subgff["start"].min()
                    end_win = subgff["end"].max()

                if strand == "-":
                    flip = True

            elif not window and nucleotide_id and (start and end):
                start, end = int(start), int(end)
                if not strand:
                    strand = "-" if end < start else "+"
                subgff = gff.query("seqid == @nucleotide_id & type =='CDS' & start>=@start & end<=@end")
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
                return None, None, unique_id, "Invalid usage of parameters"  # Invalid usage of parameters, return None and index

        # Process the attributes in GFF and merge with FAA
        subgff = unwrap_attributes(subgff)
        if "protein_id" in subgff.columns:
            key_join = "protein_id"
        else:
            key_join = "ID"
            subgff["protein_id"] = subgff["ID"]
        
        subgff = subgff.join(faa_df[["id", "sequence"]], left_on=key_join, right_on='id', how="left")
            
        if fna_file:
            fna_df = read_fasta(fna_file)
            nucleotide_id = str(nucleotide_id)
            faa_df["id"] = faa_df["id"].astype(str)
            #check if nucleotide id in fna_df["id"]
            if nucleotide_id in fna_df["id"].to_list():
                sequence = fna_df[fna_df["id"] == nucleotide_id]["sequence"].to_list()[0]
                end_win = end + window
                if end_win > len(sequence):
                    end_win = len(sequence)
                if sorfs:
                    #annotate genes in the neighborhood with pyrodigal:
                    orf_finder = pyrodigal.GeneFinder(meta=True)
                    new_genes = []

                    for i,pred in enumerate(orf_finder.find_genes(sequence.encode())):
                        overlap_flag = False
                        for row in subgff.iter_rows(named=True):
                            overlap_percentage = calculate_overlap(row['start'], row['end'], pred.begin, pred.end)
                            if overlap_percentage > 5:
                                overlap_flag = True
                                break

                        if not overlap_flag:
                            new_genes.append({
                                'seqid': nucleotide_id,
                                'source': 'pyrodigal',
                                'type': 'CDS',
                                'start': pred.begin + start_win,
                                'end': pred.end + start_win,
                                'score': pred.score,
                                'strand': "-" if pred.strand == "-1" else "+",
                                'phase': '.',
                                    key_join: f"{key_join}=sORF_{unique_id}_{i}",
                                'sequence': pred.translate()  # Assuming you want to store the translated protein sequence
                            })

                    # Convert new genes to DataFrame and concatenate with existing subgff
                    if new_genes:
                        new_genes_df = pl.DataFrame(new_genes)
                        subgff = pl.concat([subgff, new_genes_df], how="vertical")  
            else:
                print(nucleotide_id,fna_df["id"].to_list())            
        else:
            #wnd win should be the end of the last ORF in the window
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
    
    if "product" not in subgff.columns:
        subgff["product"] = None

    # Normalize identifier columns so callers always see a single canonical 'id'
    # Prefer existing lowercase 'id', then 'protein_id', then GFF 'ID', then 'gene_id'.
    try:
        if isinstance(subgff, pl.DataFrame):
            if 'id' not in subgff.columns:
                for cand in ('protein_id', 'ID', 'gene_id'):
                    if cand in subgff.columns:
                        subgff['id'] = subgff[cand]
                        break

            # Drop redundant uppercase or alternative id columns to avoid duplicate fields
            #if column "ID" exists, drop it
            if "ID" in subgff.columns:
                subgff.drop(["ID"])

            # Ensure canonical id is string to avoid merge/type surprises later
            if 'id' in subgff.columns:
                try:
                    subgff['id'] = subgff['id'].astype(str)
                except Exception:
                    pass
    except Exception:
        # Non-fatal: if normalization fails, keep original subgff
        pass

        
    return subgff, neighborhood, unique_id

def merge_cluster_result(result_df,cluster_df):
    counts = (
        cluster_df
        .group_by('clu_rep_seq')
        .agg(pl.len().alias('clu_size'))
        .sort('clu_size', descending=True)
    )
    new_df = counts.with_row_count('fam_cluster').with_columns(pl.col('fam_cluster').cast(pl.Utf8))
    new_df = new_df.filter(pl.col('clu_size') >= 2)
    merged_df = cluster_df.join(new_df[['clu_rep_seq','fam_cluster']], on='clu_rep_seq', how='left')
    results = result_df.join(merged_df[['clu_rep_seq','fam_cluster','member']], left_on='id', right_on='member', how='left').drop('member')
    return results

def flat(lis):
    flatList = []
    # Iterate with outer list
    for element in lis:
        if type(element) is list:
            # Check if type is list than iterate through the sublist
            for item in element:
                flatList.append(item)
        else:
            flatList.append(element)
    return flatList

