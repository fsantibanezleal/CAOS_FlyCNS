# Develop: both toolchains, the tests, the guards

## Python

Python 3.12 is the reference interpreter (3.11 to 3.13 are supported by the package itself).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,release]"    # add ,gpu for the PyTorch path, ,parity for brian2
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest -rs
```

Tests that need the release files, a CUDA device or brian2 carry the markers `data`, `gpu` and `parity`; they skip
when their requirement is missing and `-rs` lists every skip, so a skipped gate is never read as a pass.

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
