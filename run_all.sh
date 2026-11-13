#!/usr/bin/env bash
# One-command reproducible pipeline for the EZH2-BRCA project.
# Usage:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/7] Installing Python dependencies"
pip install -q -r requirements.txt

echo "==> [2/7] Fetching expression, subtypes, survival (Xena + cBioPortal)"
python scripts/01_fetch_ualcan.py

echo "==> [3/7] Fetching mutations + CNA (cBioPortal)"
python scripts/02_fetch_cbioportal.py

echo "==> [4/7] Fetching PPI network + enrichment (STRING v12)"
python scripts/03_fetch_string.py

echo "==> [5/7] Generating figures + statistics"
python scripts/04_generate_figures.py

echo "==> [6/7] Literature comparison (PubMed)"
python scripts/05_literature_compare.py

echo "==> [7/7] Building LaTeX report"
python scripts/06_build_report.py

echo ""
echo "Done. Deliverables:"
ls -la latex/main.pdf
ls figures/*.png
