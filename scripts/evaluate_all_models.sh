#!/bin/bash
# Script to evaluate all models

echo "========== Starting Model Evaluation =========="
echo "$(date): Evaluation started"

# Directory paths
SRC_DIR="src"
OUTPUT_DIR="outputs"

# Evaluate PPM models
echo "---------- Evaluating PPM models ----------"
python -m ${SRC_DIR}.evaluation.ppm_evaluation

# Evaluate ML models
echo "---------- Evaluating traditional ML models ----------"
python -m ${SRC_DIR}.evaluation.ml_evaluation

# Generate comprehensive reports
echo "---------- Generating reports ----------"
python -m ${SRC_DIR}.evaluation.metrics

echo "========== Evaluation Completed =========="
echo "$(date): All evaluations have been completed"
echo "Results saved in ${OUTPUT_DIR}/evaluation directory"