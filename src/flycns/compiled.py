"""The compiled directory: the one thing both simulators read.

A compiled connectome is a directory holding ``manifest.json`` and one little-endian binary file per array. The
manifest lists every array with its dtype, shape and SHA-256, plus the release it came from, the hashes of the source
tables, the string tables the index arrays point into, and the counts the compiler measured. The reader refuses a
directory whose arrays do not match their recorded hashes: a simulation must never run on a graph that is not the one
that was compiled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "flycns.compiled/1"

#: dtypes the format allows, by name; each is written little-endian.
DTYPES = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "float32": np.float32,
}


class CompiledError(ValueError):
    """A compiled directory that cannot be trusted: a missing file, a wrong shape, or a hash that differs."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _dtype_name(array: np.ndarray) -> str:
    for name, dtype in DTYPES.items():
        if array.dtype == np.dtype(dtype):
            return name
    raise CompiledError(f"dtype {array.dtype} is not allowed in a compiled directory")


def write_compiled(directory: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any],
                   schema: str = SCHEMA) -> dict[str, Any]:
    """Write ``arrays`` and a manifest into ``directory``; return the manifest.

    ``meta`` carries the release, sources, counts and string tables; it is stored as given under the manifest's
    ``release``, ``sources``, ``counts`` and ``strings`` keys. Array files are written in name order so two
    compilations of the same inputs produce the same bytes.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        dtype = _dtype_name(array)
        data = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
        file_name = f"{name}.bin"
        (directory / file_name).write_bytes(data)
        entries.append(
            {"name": name, "dtype": dtype, "shape": list(array.shape), "file": file_name, "bytes": len(data),
             "sha256": sha256_bytes(data)}
        )
    manifest = {
        "schema": schema,
        "release": meta.get("release", {}),
        "sources": meta.get("sources", []),
        "counts": meta.get("counts", {}),
        "strings": meta.get("strings", {}),
        "arrays": entries,
    }
    # LF on every system, so the same inputs give the same bytes on Windows and Linux
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                             newline="\n")
    return manifest


@dataclass
class Compiled:
    """A verified compiled connectome: its manifest, arrays and string tables."""

    directory: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(repr=False)

    @property
    def strings(self) -> dict[str, list[str]]:
        return self.manifest["strings"]

    @property
    def counts(self) -> dict[str, Any]:
        return self.manifest["counts"]

    @property
    def n_neurons(self) -> int:
        return int(self.arrays["neuron_body_id"].shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.arrays["csr_indices"].shape[0])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


def read_compiled(directory: Path, verify: bool = True, schema: str = SCHEMA) -> Compiled:
    """Read a compiled directory, checking every array against the manifest.

    With ``verify`` (the default) each file's SHA-256 is recomputed; any difference, a missing file or a size that
    does not match the declared shape raises :class:`CompiledError`.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise CompiledError(f"{directory} holds no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema:
        raise CompiledError(f"schema {manifest.get('schema')!r} is not {schema!r}")
    arrays: dict[str, np.ndarray] = {}
    for entry in manifest["arrays"]:
        path = directory / entry["file"]
        if not path.is_file():
            raise CompiledError(f"array {entry['name']} is missing its file {entry['file']}")
        data = path.read_bytes()
        if verify and sha256_bytes(data) != entry["sha256"]:
            raise CompiledError(
                f"array {entry['name']}: SHA-256 {sha256_bytes(data)} differs from the manifest's {entry['sha256']}"
            )
        dtype = np.dtype(DTYPES[entry["dtype"]]).newbyteorder("<")
        shape = tuple(entry["shape"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(data) != expected:
            raise CompiledError(f"array {entry['name']}: {len(data)} bytes where shape {shape} needs {expected}")
        arrays[entry["name"]] = np.frombuffer(data, dtype=dtype).reshape(shape).astype(dtype.newbyteorder("="))
    return Compiled(directory=directory, manifest=manifest, arrays=arrays)
