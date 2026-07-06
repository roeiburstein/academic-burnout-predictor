#!/bin/bash
# run_pipeline.sh - Automated Data Science Pipeline

# Stop on errors
set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

echo "===================================================================="
echo "🚀 Starting Data Science & Machine Learning Pipeline"
echo "===================================================================="

echo "Step 1: Running Exploratory Data Analysis & Cleaning..."
python3 "${SRC_DIR}/eda.py"

echo ""
echo "Step 2: Training, Tuning & Validating Models..."
python3 "${SRC_DIR}/train.py"

echo ""
echo "===================================================================="
echo "✅ Pipeline Executed Successfully!"
echo "Generated assets:"
echo " - data/cleaned_burnout_data.csv"
echo " - web/stats_data.json"
echo " - web/model_assets.json"
echo " - web/plots/ (visualizations & diagnostic curves)"
echo "===================================================================="
