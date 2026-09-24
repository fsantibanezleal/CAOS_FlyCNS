"""Release adapters: each turns one connectome release's official tables into a compiled directory."""

from .base import ReleaseError, SourceTable, check_table
from .malecns_v1 import compile_malecns_v1

__all__ = ["ReleaseError", "SourceTable", "check_table", "compile_malecns_v1"]
