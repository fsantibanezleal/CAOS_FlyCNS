# Develop: both toolchains, the tests, the guards

## Python

Python 3.12 is the reference interpreter (3.11 to 3.13 are supported by the package itself).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,release,parity]"    # add ,gpu for the PyTorch path
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest -rs
```

Tests that need the release files, a CUDA device or brian2 carry the markers `data`, `gpu` and `parity`; they skip
when their requirement is missing and `-rs` lists every skip, so a skipped gate is never read as a pass. Continuous
integration installs `dev`, `release` and `parity`, so the Brian2 transcription of the published model (about 3 s)
runs on every push to `develop` and `main`.

The whole-graph tests read a compiled MaleCNS (`FLYCNS_MALECNS_COMPILED`, default
`E:/_Datos/destello/compiled/malecns-v1.0`) and take minutes, not seconds: the GPU engine against the reference runs
fifteen half-second trials and two short ones on the whole CNS (about 3 minutes), and the exact degree-preserving null
rewires 25.6 million connections (about 20 s). Run them before every release that touches the dynamics or the nulls:

```bash
.venv/Scripts/python -m pytest -rs -m "data or gpu"
```

## The flyvis extraction (once per flyvis version)

The trained flyvis numbers already ship inside the package; the extraction is needed only to refresh them or to
produce the parity target of the graded engines (`FLYCNS_FLYVIS_EXTRACT`, default
`E:/_Datos/destello/models/flyvis-1.2.0`). flyvis needs Python 3.9 to 3.12 and dependencies flycns does not carry,
so it runs in its own environment:

```bash
py -3.11 -m venv flyvis_venv
flyvis_venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
flyvis_venv/Scripts/python -m pip install flyvis==1.2.0
flyvis_venv/Scripts/python -m pip install -e path/to/CAOS_FlyCNS           # the script writes flycns's format
flyvis download-pretrained                                                  # results_pretrained_models.zip
FLYVIS_ROOT_DIR=path/to/flyvis_data flyvis_venv/Scripts/python scripts/extract_flyvis_ensemble.py OUT_DIR
```

The script patches one Windows defect of flyvis's storage layer (datamate 1.0.0 deletes an h5 file it still holds
open) before importing flyvis; run twice, it writes the same arrays byte for byte.

## TypeScript

Node 20 or newer.

```bash
npm ci
npm run typecheck
npm test
```

## Guards

Standard library only, run before anything is installed (continuous integration runs them first):

```bash
python scripts/check_content_standards.py   # no em-dash, no emoji in tracked text
python scripts/check_ci_budget.py           # workflows stay cheap and trunk-only
python scripts/check_sdd.py                 # every requirement in the design names a gate that exists
```

## Versions

`VERSION` holds the display form (`0.00.000`); `pyproject.toml`, `package.json` and `flycns.__version__` hold the
semantic form (`0.0.0`). `tests/test_version.py` and `ts/test/version.test.ts` fail when they disagree.
