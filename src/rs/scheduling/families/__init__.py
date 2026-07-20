"""Formal RouterSense core definitions."""
from .core import (
    FAMILY_KERNEL_SPECS,
    PRIMARY_FAMILY_IDS,
    STRICT_FAMILY_IDS,
    FamilyKernelSpec,
    LiteratureLineage,
    family_inventory,
    get_family_kernel_spec,
    normalize_family_id,
)

__all__ = [
    "FAMILY_KERNEL_SPECS",
    "PRIMARY_FAMILY_IDS",
    "STRICT_FAMILY_IDS",
    "FamilyKernelSpec",
    "LiteratureLineage",
    "family_inventory",
    "get_family_kernel_spec",
    "normalize_family_id",
]
