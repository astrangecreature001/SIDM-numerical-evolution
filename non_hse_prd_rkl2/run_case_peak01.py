"""Run a non-HSE case with the 0.1-t0 post-peak confirmation interval."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_case.py")), run_name="__main__")
