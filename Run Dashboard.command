#!/bin/bash
set -u  # error on undefined vars (do NOT use -e: we want to keep the window open on error)

# ── Macro ETF Dashboard Launcher ──────────────────────────────────────────────
# Double-click this file to generate the dashboard.
# Place this file in the same folder as generate_dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

# Move to the folder where this script lives
cd "$(dirname "$0")" || { echo "Could not cd to script dir"; read -r -p "Press Enter..."; exit 1; }

# Load shell environment (needed for ANTHROPIC_API_KEY and other vars)
[ -f ~/.zshrc ] && source ~/.zshrc
[ -f ~/.bash_profile ] && source ~/.bash_profile

echo "============================================"
echo "  Macro ETF Dashboard Generator"
echo "============================================"
echo ""

# Check Python 3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found."
    echo "Install it from https://www.python.org/downloads/"
    read -r -p "Press Enter to close..."
    exit 1
fi

echo "Python: $(python3 --version)"
echo ""

# Install / upgrade dependencies
echo "Checking dependencies..."
if [ -f requirements.txt ]; then
    if ! python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt; then
        echo "ERROR: Could not install dependencies from requirements.txt."
        read -r -p "Press Enter to close..."
        exit 1
    fi
else
    if ! python3 -m pip install --quiet --disable-pip-version-check "numpy<2" pandas yfinance; then
        echo "ERROR: Could not install dependencies."
        read -r -p "Press Enter to close..."
        exit 1
    fi
fi
echo "Dependencies OK."
echo ""

# Warn if no API key configured
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ ! -f .env ]; then
    echo "NOTE: ANTHROPIC_API_KEY not set and no .env file found."
    echo "      AI analysis will be skipped if you answer 's'."
    echo ""
fi

# Run
echo "Starting download and analysis (~1-3 minutes)..."
echo ""
python3 generate_dashboard.py
RC=$?

echo ""
echo "--------------------------------------------"

if [ $RC -ne 0 ]; then
    echo "ERROR: generate_dashboard.py exited with status $RC."
    read -r -p "Press Enter to close..."
    exit $RC
fi

# Open the HTML in the default browser if it was created
if [ -f "macro_etf_dashboard.html" ]; then
    echo "Opening dashboard in browser..."
    open "macro_etf_dashboard.html"
else
    echo "ERROR: Dashboard file was not created."
fi

echo ""
read -r -p "Press Enter to close this window..."
