# hoodini/config/schema.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Config:
    input_path: Optional[Path]
    inputsheet: Optional[Path]
    output: str
    max_concurrent_downloads: int
    apikey: str
    num_threads: int
    assembly_folder: Optional[str]
    cand_mode: str
    mod: str
    wn: int
    height_factor: int
    padloc: bool
    deffinder: bool
    tree_mode: str
    tree_file: str
    ncrna: bool
    cctyper: bool
    ngenes: int
    clust_method: str
    assembly_db: str
    img_db: Optional[str]
    img_nuc: Optional[str]
    keep: bool
    domains: Optional[str]
    domains_metadata: Optional[str]
    min_prevalence: float
    genomad: bool
    antidefense: bool
    phrogs: bool
    prot_links: bool
    img: str
    img_metadata: str
    blast: Optional[str]
    force: bool
    sorfs: bool
    minwin: int
    minwin_type: str
    nt_links: bool
    ani_mode: str
