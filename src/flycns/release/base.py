"""What every release adapter shares: locked source tables and streamed reading.

A release table is used only when its SHA-256 equals the value locked in the adapter. A table that differs, even by
one byte, is refused with both hashes named, because a compiled graph is only as trustworthy as the tables it came
from, and a silently changed table would change every number downstream.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..compiled import sha256_file


class ReleaseError(ValueError):
    """A release table that cannot be used: missing, or not the locked file."""


@dataclass(frozen=True)
class SourceTable:
    """One locked release table."""

    key: str
    file_name: str
    url: str
    bytes: int
    sha256: str

    def describe(self) -> dict:
        return {"key": self.key, "file": self.file_name, "url": self.url, "bytes": self.bytes, "sha256": self.sha256}


def check_table(directory: Path, table: SourceTable, verify_hash: bool = True) -> Path:
    """Return the path of ``table`` inside ``directory`` after checking its size and SHA-256."""
    path = Path(directory) / table.file_name
    if not path.is_file():
        raise ReleaseError(f"{table.file_name} is not in {directory}; download it from {table.url}")
    size = path.stat().st_size
    if size != table.bytes:
        raise ReleaseError(f"{table.file_name}: {size} bytes, expected {table.bytes}")
    if verify_hash:
        found = sha256_file(path)
        if found != table.sha256:
            raise ReleaseError(f"{table.file_name}: SHA-256 {found} differs from the locked {table.sha256}")
    return path


def iter_batches(path: Path, columns: Sequence[str]) -> Iterator[dict]:
    """Stream a feather (Arrow IPC) file batch by batch, yielding the requested columns as NumPy arrays.

    Dictionary-encoded columns come back decoded. The file is memory-mapped, so memory stays bounded by one batch.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path)) as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index).select(list(columns))
            out = {}
            for name, column in zip(batch.schema.names, batch.columns, strict=True):
                if pa.types.is_dictionary(column.type):
                    column = column.dictionary_decode()
                out[name] = column.to_numpy(zero_copy_only=False)
            yield out
