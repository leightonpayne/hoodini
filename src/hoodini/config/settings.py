"""Runtime configuration helpers.

Centralizes how defaults, user TOML files, and CLI overrides are merged into a
single typed object. Keep this layer free of CLI concerns so it can be reused
by tests or other entry points.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, MutableMapping, Optional


def _lower_keys(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new dict with string keys lowercased (TOML keys often vary)."""
    lowered: Dict[str, Any] = {}
    for key, value in data.items():
        lowered[key.lower()] = value
    return lowered


def _flatten_config(grouped: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a nested TOML-style mapping into a single-level dict."""
    flat: Dict[str, Any] = {}
    for section, values in grouped.items():
        if isinstance(values, MutableMapping):
            flat.update(_lower_keys(values))
        else:
            flat[section] = values
    return flat


@dataclass(slots=True)
class RuntimeConfig:
    input_path: Optional[str] = None
    inputsheet: Optional[str] = None
    output: Optional[str] = None

    max_concurrent_downloads: Optional[int] = None
    apikey: Optional[str] = None
    num_threads: Optional[int] = None
    assembly_folder: Optional[str] = None
    assembly_db: Optional[str] = None

    prot_links: bool = False
    nt_links: bool = False

    ani_mode: Optional[str] = None
    nt_aln_mode: Optional[str] = None
    blast: Optional[str] = None
    cand_mode: Optional[str] = None
    clust_method: Optional[str] = None
    mod: Optional[str] = None
    wn: Optional[int] = None
    height_factor: Optional[int] = None
    ngenes: Optional[int] = None
    minwin: Optional[int] = None
    minwin_type: Optional[str] = None

    tree_mode: Optional[str] = None
    tree_file: Optional[str] = None
    aai_mode: Optional[str] = None
    aai_subset_mode: Optional[str] = None
    nj_algorithm: Optional[str] = None

    padloc: bool = False
    deffinder: bool = False
    ncrna: bool = False
    cctyper: bool = False
    genomad: bool = False
    sorfs: bool = False
    domains: Optional[list[str]] = field(default=None)
    emapper: bool = False
    min_prevalence: Optional[float] = None

    keep: bool = False
    force: bool = False

    def replace(self, **kwargs: Any) -> "RuntimeConfig":
        """Return a copy with provided fields updated."""
        return replace(self, **kwargs)


def build_runtime_config(
    *,
    defaults: Mapping[str, Any],
    file_overrides: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Merge defaults + file + CLI into a RuntimeConfig.

    Later sources override earlier ones. Unknown keys are ignored to keep the
    dataclass strict.
    """

    merged: Dict[str, Any] = {}
    for source in (defaults, file_overrides or {}, cli_overrides or {}):
        merged.update(_flatten_config(source))

    # Normalize keys that appear with dashes in TOML
    if "aai-subset-mode" in merged and "aai_subset_mode" not in merged:
        merged["aai_subset_mode"] = merged["aai-subset-mode"]
    if "aai_subset_mode" in merged and "aai-subset-mode" not in merged:
        merged["aai-subset-mode"] = merged["aai_subset_mode"]

    # Only pass known fields to the dataclass
    allowed_fields: set[str] = {field.name for field in RuntimeConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered: Dict[str, Any] = {k: v for k, v in merged.items() if k in allowed_fields}

    config = RuntimeConfig(**filtered)
    # Ensure runtime-only keys are set
    if config.input_path is None:
        config.input_path = None
    if config.inputsheet is None:
        config.inputsheet = None
    return config


__all__ = ["RuntimeConfig", "build_runtime_config"]
