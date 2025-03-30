#!/bin/bash
# scripts/run_pipeline.sh
# Master script to run the complete Nigerian Pidgin English Code-Switching detection pipeline

set -e  # Exit on any error

# Set base directory
BASE_DIR=$(pwd)
CONFIG_FILE="$BASE_DIR/config/config.yaml"
DATA_DIR="$BASE_DIR/data/corpus"
OUTPUT_DIR="$BASE_DIR/outputs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create necessary directories
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/predictions"
mkdir -p "$OUTPUT_DIR/evaluation"
mkdir -p "$OUTPUT_DIR/logs"

# Log file
LOG_FILE="$OUTPUT_DIR/logs/pipeline_$TIMESTAMP.log"

# Function to log messages
log() {
  local message="$1"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $message" | tee -a "$LOG_FILE"
}

# Start pipeline
log "Starting Nigerian Pidgin English Code-Switching detection pipeline"
log "Timestamp: $TIMESTAMP"
log "Configuration: $CONFIG_FILE"

# Step 1: Train models
log "Step 1: Training models"

# Train PPM models using Tawa toolkit
log "Training PPM models..."
if bash scripts/train_ppm_models.sh >> "$LOG_FILE" 2>&1; then
  log "PPM models trained successfully"
else
  log "ERROR: PPM model training failed"
  exit 1
fi

# Train ML models using Python
log "Training ML models..."
if python -m src.main --mode train --model all --config "$CONFIG_FILE" >> "$LOG_FILE" 2>&1; then
  log "ML models trained successfully"
else
  log "ERROR: ML model training failed"
  exit 1
fi

# Step 2: Make predictions on test data
log "Step 2: Making predictions on test data"
TEST_FILE="$DATA_DIR/ml_ground_truth.txt"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
  log "ERROR: Test file not found: $TEST_FILE"
  exit 1
fi

# Make predictions with all models
log "Making predictions..."
if python -m src.main --mode predict --model all --input "$TEST_FILE" --output "$OUTPUT_DIR/predictions/test_predictions_$TIMESTAMP" --config "$CONFIG_FILE" >> "$LOG_FILE" 2>&1; then
  log "Predictions completed successfully"
else
  log "ERROR: Prediction failed"
  exit 1
fi

# Step 3: Evaluate predictions
log "Step 3: Evaluating predictions"
if python -m src.main --mode evaluate --input "$OUTPUT_DIR/predictions" --ground-truth "$TEST_FILE" --output "$OUTPUT_DIR/evaluation/evaluation_$TIMESTAMP" --config "$CONFIG_FILE" >> "$LOG_FILE" 2>&1; then
  log "Evaluation completed successfully"
else
  log "ERROR: Evaluation failed"
  exit 1
fi

# Step 4: Generate reports 
log "Step 4: Generating final reports"
if bash scripts/generate_report.sh >> "$LOG_FILE" 2>&1; then
  log "Report generation completed successfully"
else
  log "ERROR: Report generation failed"
  exit 1
fi

# Pipeline completed
log "Pipeline completed successfully!"
log "Results saved to: $OUTPUT_DIR"
log "Log file: $LOG_FILE"

echo ""
echo "====================================="
echo "Nigerian Pidgin English Code-Switching Detection Pipeline Complete"
echo "Results are available in: $OUTPUT_DIR"
echo "====================================="
