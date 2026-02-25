from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "VERSION"
__version__ = VERSION_PATH.read_text(encoding="utf-8").strip()