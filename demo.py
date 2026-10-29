"""Launch the optional local Gradio demo; training code is unchanged."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from tiger_reid.demo import main

    main(ROOT)
