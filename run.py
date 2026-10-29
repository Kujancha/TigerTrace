"""Run from a source checkout without installing the package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from tiger_reid.cli import main

if __name__ == "__main__":
    main(project_root=ROOT)
