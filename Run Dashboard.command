#!/bin/bash

# ── Macro ETF Dashboard Launcher ──────────────────────────────────────────────
# Double-click this file to generate the dashboard.
# Place this file in the same folder as generate_dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

# Move to the folder where this script lives
cd "$(dirname "$0")"

echo "============================================"
echo "  Macro ETF Dashboard Generator"
echo "============================================"
echo ""

# Check Python 3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found."
    echo "Install it from https://www.python.org/downloads/"
    read -p "Press Enter to close..."
    exit 1
fi

echo "Python: $(python3 --version)"
echo ""

# Install / upgrade dependencies silently
echo "Checking dependencies..."
python3 -m pip install --quiet "numpy<2" pandas yfinance 2>/dev/null
echo "Dependencies OK."
echo ""

# Run
echo "Starting download and analysis (~3 minutes)..."
echo ""
python3 generate_dashboard.py

echo ""
echo "--------------------------------------------"

# Open the HTML in the default browser if it was created
if [ -f "macro_etf_dashboard.html" ]; then
    echo "Opening dashboard in browser..."
    open "macro_etf_dashboard.html"
else
    echo "ERROR: Dashboard file was not created."
fi

echo ""
read -p "Press Enter to close this window..."
