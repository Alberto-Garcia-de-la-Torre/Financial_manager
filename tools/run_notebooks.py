"""Execute notebooks headless and save their outputs in place.

    python tools/run_notebooks.py [notebooks/day24_momentum.ipynb ...]

With no arguments, runs every notebook under notebooks/. A cell that raises
(a failed `assert` included) stops that notebook and makes the exit status
non-zero, so the notebook's own checks fail loudly instead of being read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def run(path: Path) -> bool:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    try:
        client.execute()
    except CellExecutionError as error:
        print(f"FAIL {path}\n{error}", file=sys.stderr)
        return False
    finally:
        nbformat.write(notebook, path)
    print(f"ok   {path}")
    return True


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    results = [run(path) for path in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
