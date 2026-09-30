from __future__ import annotations


import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VendorRoots:
    vggt_omega: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_vendor_roots() -> VendorRoots:
    repo_root = get_repo_root()
    return VendorRoots(
        vggt_omega=repo_root / "other" / "vggt-omega",
    )


def ensure_vendor_paths() -> VendorRoots:
    roots = get_vendor_roots()
    for path in (roots.vggt_omega,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return roots
