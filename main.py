"""Run the Heart Disease Prediction notebook end-to-end.

Usage:
    uv run python main.py

This executes notebooks/Heart_Disease_Prediction.ipynb in place, regenerating all
figures in report/figures, the metric files in report/, and the saved models. It then
exports the executed notebook to report/Heart_Disease_Prediction.html.
"""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    notebook = root / "notebooks" / "Heart_Disease_Prediction.ipynb"
    report_dir = root / "report"

    if not notebook.exists():
        raise SystemExit(f"Notebook not found: {notebook}")

    print(f"Executing {notebook.name} ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=1200",
            str(notebook),
        ],
        cwd=root,
        check=True,
    )

    print("Exporting rendered HTML report ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "html",
            "--output-dir",
            str(report_dir),
            "--output",
            "Heart_Disease_Prediction",
            str(notebook),
        ],
        cwd=root,
        check=True,
    )
    print("Done. Outputs written to report/ and models/.")


if __name__ == "__main__":
    main()
