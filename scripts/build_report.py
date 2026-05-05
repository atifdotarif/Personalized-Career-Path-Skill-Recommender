"""
Convert docs/REPORT.md to docs/REPORT.html and docs/REPORT.pdf.

Run from the project root:

    .venv/Scripts/python.exe scripts/build_report.py

Requires `pandoc` on PATH (for the HTML step) and the `xhtml2pdf` Python
package (for the PDF step). Both are listed in requirements.txt.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def main() -> None:
    md = DOCS / "REPORT.md"
    css = DOCS / "report_style.css"
    html = DOCS / "REPORT.html"
    pdf = DOCS / "REPORT.pdf"

    if not md.exists():
        sys.exit(f"ERROR: {md} not found.")
    if not shutil.which("pandoc"):
        sys.exit(
            "ERROR: pandoc not found on PATH. Install from "
            "https://pandoc.org/installing.html"
        )

    # MD -> HTML (CSS embedded so the file is self-contained).
    subprocess.check_call(
        [
            "pandoc",
            str(md.name),
            "--css", str(css.name),
            "--embed-resources",
            "--standalone",
            "-o", str(html.name),
        ],
        cwd=DOCS,
    )
    print(f"Wrote {html} ({html.stat().st_size / 1024:.1f} KB)")

    # HTML -> PDF (pure-Python via xhtml2pdf).
    try:
        from xhtml2pdf import pisa  # type: ignore
    except ImportError:
        sys.exit("ERROR: xhtml2pdf not installed. Run `pip install xhtml2pdf`.")

    with open(html, "r", encoding="utf-8") as f:
        html_text = f.read()
    with open(pdf, "wb") as out:
        result = pisa.CreatePDF(html_text, dest=out)
    if result.err:
        sys.exit(f"ERROR: xhtml2pdf reported {result.err} errors.")
    print(f"Wrote {pdf} ({pdf.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
