# flycns

[![ci](https://github.com/fsantibanezleal/CAOS_FlyCNS/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/fsantibanezleal/CAOS_FlyCNS/actions/workflows/ci.yaml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Compile a fly connectome release into a graph a simulator can run, model the two compound eyes over the release's
own optic-lobe columns, and simulate the whole central nervous system with the published neuron models, in Python
and in the browser, with the two implementations held to each other by parity tests.

- **Release:** MaleCNS v1.0, the complete male *Drosophila* CNS (166,700 neurons, brain and ventral nerve cord;
  Berg et al., *Cell* 189:5504-5526, 2026, doi:10.1016/j.cell.2026.08.015; data CC BY 4.0). Both optic lobes are
  mapped column by column in the release (879 columns left, 892 right), which is what lets two eyes feed it.
- **Neuron models:** the published leaky integrate-and-fire model of the fly brain (Shiu et al., *Nature*
  634:210-219, 2024, doi:10.1038/s41586-024-07763-9), and a graded optic lobe in the form of the connectome-constrained
  visual model (Lappalainen et al., *Nature* 2024, doi:10.1038/s41586-024-07939-3).
- **Two implementations, one set of equations:** a Python reference (NumPy, and PyTorch on a GPU) and a browser
  engine (`@fasl-work/flycns`, WebGPU with a worker fallback).

## What it is not

A general neural simulator, a learned model of the fly, or a claim about behaviour. Synapse counts and signs are
data and never change; the package integrates published equations over them and says where those equations are
known to fall short.

## Status

Version 0.01.000: the MaleCNS v1.0 compiler and the compiled format. The eye model is the next unit; this README lists
capabilities only as they land.

## Compile MaleCNS v1.0

Download the four tables listed in [`docs/releases/01_malecns-v1.md`](docs/releases/01_malecns-v1.md) into one
folder, then:

```python
from pathlib import Path
from flycns.release import compile_malecns_v1
from flycns.compiled import read_compiled

compile_malecns_v1(Path("malecns-tables"), Path("compiled/malecns-v1.0"), progress=print)
graph = read_compiled(Path("compiled/malecns-v1.0"))      # every array verified against its SHA-256
print(graph.n_neurons, graph.n_edges, graph.counts["columns"])
```

A table whose SHA-256 differs from the locked value is refused. Compilation streams the 13 GB synapse table in record
batches and takes a few minutes on a desktop.

## Install (development)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,release]"     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest -rs
npm ci && npm test
```

## Documentation

The wiki starts at [`docs/README.md`](docs/README.md).

## License

Code: MIT. Connectome data are not redistributed here; they are fetched from the release with their hashes checked,
and remain under their own licence (MaleCNS: CC BY 4.0).
