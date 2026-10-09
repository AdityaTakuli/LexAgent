# LexAgent v3.0 | paper/render_pages.py
"""Render the pages of the compiled paper to PNG, for a visual check of layout and figures.

    .venv\\Scripts\\python.exe paper/render_pages.py [dpi]
"""

import os
import sys

import pymupdf

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PAPER_DIR, "generated", "pages")


def main(dpi: int = 110) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    document = pymupdf.open(os.path.join(PAPER_DIR, "lexagent_ieee.pdf"))
    for number, page in enumerate(document, start=1):
        path = os.path.join(OUT_DIR, f"page_{number:02d}.png")
        page.get_pixmap(dpi=dpi).save(path)
        print(path)
    print(f"{len(document)} pages")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 110)
