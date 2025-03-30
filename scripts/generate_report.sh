#!/bin/bash
# Script to generate comprehensive reports and visualizations

echo "========== Generating Reports =========="
echo "$(date): Report generation started"

# Directory paths
SRC_DIR="src"
OUTPUT_DIR="outputs"
REPORT_DIR="${OUTPUT_DIR}/results_summary"

# Create reports directory if it doesn't exist
mkdir -p ${REPORT_DIR}

# Generate performance reports
echo "---------- Generating performance reports ----------"
python -m ${SRC_DIR}.visualization.plots --mode=performance

# Generate comparative analysis
echo "---------- Generating comparative analysis ----------"
python -m ${SRC_DIR}.visualization.plots --mode=comparison

# Generate model statistics
echo "---------- Generating model statistics ----------"
python -m ${SRC_DIR}.evaluation.metrics --mode=summary --output=${REPORT_DIR}/summary_report.txt

echo "========== Report Generation Completed =========="
echo "$(date): All reports have been generated"
echo "Reports saved in ${REPORT_DIR} directory"