import pandas as pd
from hoodini.utils.core import console

def run_extra_annotations(
    *,
    records: pd.DataFrame,
    all_gff: pd.DataFrame,
    all_neigh: pd.DataFrame,
    den_data: pd.DataFrame,
    output: str,
    domains: str = None,
    domains_metadata: str = None,
    blast: str = None,
    padloc: bool = False,
    deffinder: bool = False,
    cctyper: bool = False,
    genomad: bool = False,
    ncrna: bool = False,
    antidefense: bool = False,
    phrogs: bool = False,
    num_threads: int = 4,
    valid_unique_ids: list = None,
) -> dict:
    """
    Run all extra annotation modules on the protein/gene data.
    
    Parameters:
        - all_gff: Annotated GFF as a DataFrame.
        - all_neigh: Neighborhoods DataFrame.
        - den_data: Dendrogram data for mapping tree labels.
        - output: Output directory path.
        - num_threads: Number of threads to use.
        - domains, domains_metadata: HMMs and metadata for domain annotation.
        - blast, padloc, deffinder, cctyper, genomad, ncrna, antidefense, phrogs: booleans or paths to activate annotations.
        - valid_unique_ids: list of IDs selected as valid neighborhoods.
        
    Returns:
        - all_gff: Annotated GFF DataFrame with added columns.
        - nc_data: DataFrame of non-coding hits (e.g. CRISPR, ncRNA, BLAST).
        - domains_data: Domain annotation metadata.
        - genomad_df: Virus/plasmid annotations from GenoMAD.
    """
    domains_data = pd.DataFrame()
    genomad_df = pd.DataFrame()
    nc_data = pd.DataFrame()
    columns = list(all_gff.columns)

    # Domain annotation
    if domains:
        from hoodini.extra_tools.domain import run_domain
        domains_data = run_domain(all_gff, output, domains, domains_metadata, num_threads)

    # BLAST annotation
    if blast:
        from hoodini.extra_tools.blast import run_blast
        blast_df = run_blast(all_neigh, output, blast, num_threads, valid_unique_ids)

    # PADLOC annotation
    if padloc:
        from hoodini.extra_tools.padloc import run_padloc
        padloc_df = run_padloc(all_gff, output, num_threads)

    # DefenseFinder annotation
    if deffinder:
        from hoodini.extra_tools.defensefinder import run_defensefinder
        deffinder_prots = run_defensefinder(all_gff, output)

    # CCTyper annotation
    if cctyper:
        from hoodini.extra_tools.cctyper import run_cctyper
        cctyper_prots, cctyper_arrays = run_cctyper(all_gff, all_neigh, den_data, output, num_threads, valid_unique_ids, nc_data)

    # ncRNA/Infernal annotation
    if ncrna:
        from hoodini.extra_tools.ncrna import run_ncrna
        nc_data = run_ncrna(all_neigh, den_data, output, num_threads, valid_unique_ids, nc_data)

    # GenoMAD annotation
    if genomad:
        from hoodini.extra_tools.genomad import run_genomad
        genomad_df = run_genomad(all_neigh, den_data, output, num_threads, valid_unique_ids)

    # Anti-defense annotation
    if antidefense:
        from hoodini.extra_tools.antidefense import run_antidefense
        all_gff = run_antidefense(all_gff, output, num_threads)

    # PHROGs annotation
    if phrogs:
        from hoodini.extra_tools.phrogs import run_phrogs
        all_gff = run_phrogs(all_gff, output, num_threads)

    return {
        "records": records,
        "all_gff": all_gff,
        "all_neigh": all_neigh,
        "domains_data": domains_data,
        "genomad_df": genomad_df,
        "nc_data": nc_data
    }


