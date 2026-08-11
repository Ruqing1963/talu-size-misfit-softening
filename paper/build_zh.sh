#!/usr/bin/env bash
# Rebuild the Chinese paper: markdown -> LaTeX -> PDF (xelatex + xeCJK).
set -e
python3 mark_figures.py --zh
pandoc manuscript_zh_marked.md -o body_raw.tex --wrap=preserve
python3 fix_tex.py --zh
xelatex -interaction=nonstopmode main_zh.tex > /dev/null
bibtex main_zh > /dev/null || true   # first pass has no .aux yet
xelatex -interaction=nonstopmode main_zh.tex > /dev/null
xelatex -interaction=nonstopmode main_zh.tex > /dev/null
python3 flatten.py --zh
rm -f *.blg body_raw.tex manuscript_zh_marked.md *.aux *.log *.out
echo "  bibliography: $(grep -c bibitem main_zh.bbl) entries"
echo "main_zh.pdf rebuilt ($(pdfinfo main_zh.pdf | awk '/Pages/{print $2}') pages)"
