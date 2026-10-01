"""Standalone experiment entry point; CUDA visibility is set before torch imports."""
from experiment.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
