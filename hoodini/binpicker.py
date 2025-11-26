"""Small helper to pick and run a bundled native binary for the current host.

Usage:
  - Place binaries under `hoodini/bin/{os}-{arch}/toolname` (e.g. linux-x86_64/mytool)
  - Call `get_binary_path('mytool')` to get a Path to the appropriate file.
  - Use `run_binary('mytool', args=[...])` to execute and capture output.

This intentionally keeps dependencies minimal (stdlib only).
"""
from __future__ import annotations
from pathlib import Path
import platform
import stat
import subprocess
from typing import Iterable, List, Optional, Tuple


def _normalize_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m


def _normalize_os(sysname: str) -> str:
    s = sysname.lower()
    if s.startswith("linux"):
        return "linux"
    if s.startswith("darwin") or s.startswith("mac"):
        return "darwin"
    if s.startswith("windows"):
        return "windows"
    return s


def detect_platform() -> Tuple[str, str]:
    """Return (os, arch) normalized strings, e.g. ('linux','x86_64') or ('darwin','arm64')."""
    sysname = platform.system()
    machine = platform.machine()
    return _normalize_os(sysname), _normalize_arch(machine)


def get_binary_path(name: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Return a Path to a bundled binary matching the host, or None if not found.

    Search layout:
      <base_dir or package_dir>/bin/{os}-{arch}/{name}
    Example: hoodini/bin/linux-x86_64/mytool
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    osname, arch = detect_platform()
    candidates = [f"{osname}-{arch}", f"{osname}-{arch}-gnu", f"{osname}-{arch}-musl"]
    for cand in candidates:
        p = (base_dir / "bin" / cand / name)
        if p.exists():
            return p
    # fallback: look for any directory starting with osname-
    bin_dir = base_dir / "bin"
    if bin_dir.exists():
        for d in bin_dir.iterdir():
            if d.is_dir() and d.name.startswith(osname + "-"):
                p = d / name
                if p.exists():
                    return p
    return None


def ensure_executable(path: Path) -> None:
    """Ensure the file is executable by the user (chmod +x)."""
    st = path.stat()
    # user exec bit
    if not (st.st_mode & stat.S_IXUSR):
        path.chmod(st.st_mode | stat.S_IXUSR)


def list_available_binaries(base_dir: Optional[Path] = None) -> List[str]:
    """Return a list of available binary paths under the default bin/ layout (relative to package)."""
    if base_dir is None:
        base_dir = Path(__file__).parent
    out = []
    bdir = base_dir / "bin"
    if not bdir.exists():
        return out
    for d in sorted(bdir.iterdir()):
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    out.append(str(f.relative_to(base_dir)))
    return out


def run_binary(name: str,
               args: Optional[Iterable[str]] = None,
               base_dir: Optional[Path] = None,
               check: bool = True,
               capture_output: bool = True,
               text: bool = True,
               env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Locate and run the bundled binary, returning subprocess.CompletedProcess.

    Raises FileNotFoundError if a suitable binary isn't present.
    """
    p = get_binary_path(name, base_dir=base_dir)
    if p is None:
        raise FileNotFoundError(f"No bundled binary found for '{name}' on this platform ({detect_platform()})")
    ensure_executable(p)
    cmd: List[str] = [str(p)]
    if args:
        cmd.extend(list(args))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text, env=env)
