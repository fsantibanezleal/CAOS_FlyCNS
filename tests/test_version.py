"""The three places a version is written must agree, or a release publishes one number and documents another."""

import re
import tomllib
from pathlib import Path

import flycns

ROOT = Path(__file__).resolve().parents[1]


def semver(display: str) -> str:
    return ".".join(str(int(part)) for part in display.split("."))


def test_version_sources_agree():
    display = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"\d\.\d{2}\.\d{3}", display), display
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    npm = __import__("json").loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"]
    assert manifest == semver(display)
    assert flycns.__version__ == semver(display)
    assert npm == semver(display)
