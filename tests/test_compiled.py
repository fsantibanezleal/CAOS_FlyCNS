"""The compiled directory refuses tampering and is reproducible byte for byte."""

from __future__ import annotations

import json

import pytest

from flycns.compiled import CompiledError, read_compiled
from flycns.release import compile_malecns_v1


def test_tampered_array_is_refused(synthetic_release, tmp_path):
    directory, tables = synthetic_release
    out = tmp_path / "out"
    compile_malecns_v1(directory, out, tables=tables)
    read_compiled(out)  # intact: reads
    path = out / "csr_indices.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(CompiledError) as caught:
        read_compiled(out)
    assert "csr_indices" in str(caught.value)


def test_compilation_is_deterministic(synthetic_release, tmp_path):
    directory, tables = synthetic_release
    first, second = tmp_path / "a", tmp_path / "b"
    compile_malecns_v1(directory, first, tables=tables)
    compile_malecns_v1(directory, second, tables=tables)
    manifest_a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_a == manifest_b
    for entry in manifest_a["arrays"]:
        assert (first / entry["file"]).read_bytes() == (second / entry["file"]).read_bytes()
