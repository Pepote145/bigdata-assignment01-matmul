"""Shared helpers: paths, config and the command line for each implementation."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_config():
    return json.loads((ROOT / "config.json").read_text())


def command(lang, cfg):
    """Return the argv prefix that starts the implementation for `lang`."""
    if lang == "c":
        return [str(BUILD / "matmul_c")]
    if lang == "java":
        return ["java", *cfg["java_opts"], "-cp", str(BUILD / "java"), "MatMul"]
    if lang == "python":
        return [sys.executable, str(ROOT / "src" / "python" / "matmul.py")]
    raise ValueError(lang)
