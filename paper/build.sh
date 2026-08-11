#!/usr/bin/env bash
# Rebuild the paper: markdown -> LaTeX -> PDF.
set -e
python3 mark_figures.py
pandoc manuscript_marked.md -o body_raw.tex --wrap=preserve
python3 fix_tex.py
xelatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null || true   # first pass has no .aux yet
xelatex -interaction=nonstopmode main.tex > /dev/null
xelatex -interaction=nonstopmode main.tex > /dev/null
python3 flatten.py
rm -f *.blg body_raw.tex manuscript_marked.md *.aux *.log *.out
echo "  bibliography: $(grep -c bibitem main.bbl) entries"
echo "main.pdf rebuilt ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
