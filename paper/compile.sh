#!/bin/bash

# Ensure we are in the paper directory
cd "$(dirname "$0")"

echo "Compiling main.tex..."
# If pdflatex is not in PATH, try standard macTeX paths
if command -v pdflatex >/dev/null 2>&1; then
    PDFLATEX=pdflatex
elif [ -f "/Library/TeX/texbin/pdflatex" ]; then
    PDFLATEX="/Library/TeX/texbin/pdflatex"
else
    echo "pdflatex not found. Make sure LaTeX is installed."
    exit 1
fi

$PDFLATEX main.tex
$PDFLATEX main.tex # run twice to ensure references are updated

echo "Done."
