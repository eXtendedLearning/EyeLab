#!/usr/bin/env python3
"""Build the EyeLab wiki into one self-contained page.

    python docs/wiki/build_wiki.py

Reads src/head.html, src/wiki.css, src/shell.html, src/pages/*.html (in name order)
and src/wiki.js, and writes index.html next to this script. Images stay in img/ and
are referenced relatively. Standard library only.

Editing rules: every user-facing text exists twice, in .l-en and .l-it elements;
numbers in table.data cells are written with a decimal point (converted for Italian
at runtime); each page is an <article class="page" id="p-KEY" data-group data-en data-it>.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def build() -> Path:
    head = (SRC / "head.html").read_text(encoding="utf-8")
    css = (SRC / "wiki.css").read_text(encoding="utf-8")
    shell = (SRC / "shell.html").read_text(encoding="utf-8")
    js = (SRC / "wiki.js").read_text(encoding="utf-8")
    pages = "\n".join(p.read_text(encoding="utf-8") for p in sorted((SRC / "pages").glob("*.html")))
    if "<!--PAGES-->" not in shell:
        raise SystemExit("shell.html lacks the <!--PAGES--> marker")
    body = shell.replace("<!--PAGES-->", pages)
    out = HERE / "index.html"
    out.write_text(f"{head}\n<style>\n{css}</style>\n{body}\n<script>\n{js}</script>\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
