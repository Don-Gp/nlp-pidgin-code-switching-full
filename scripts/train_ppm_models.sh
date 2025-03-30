#!/bin/bash
# scripts/train_ppm_models.sh
# Shell script to train PPM models for Nigerian Pidgin/English code-switching detection

# Set base directories
BASE_DIR=$(pwd)
DATA_DIR="$BASE_DIR/data/corpus"
MODEL_DIR="$BASE_DIR/models/ppm"

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

# Input files (these should exist)
PIDGIN_FILE="$DATA_DIR/pidgin.txt"
ENGLISH_FILE="$DATA_DIR/english.txt"

# Check if input files exist
if [ ! -f "$PIDGIN_FILE" ]; then
  echo "ERROR: Pidgin training file not found: $PIDGIN_FILE"
  exit 1
fi

if [ ! -f "$ENGLISH_FILE" ]; then
  echo "ERROR: English training file not found: $ENGLISH_FILE"
  exit 1
fi

# Define model orders to train
ORDERS=(2 3 4 5 6 7 8)

# Train models for each order
for ORDER in "${ORDERS[@]}"; do
  echo "=== Training Pidgin model with order $ORDER ==="
  train -i $PIDGIN_FILE -o $MODEL_DIR/pidgin${ORDER}.model -S -p 1000000 -a 256 -O $ORDER -e 'D' -T "Order $ORDER Pidgin"

  echo "=== Training English model with order $ORDER ==="
  train -i $ENGLISH_FILE -o $MODEL_DIR/english${ORDER}.model -S -p 1000000 -a 256 -O $ORDER -e 'D' -T "Order $ORDER English"

  # Create the .dat file for this order
  DAT_FILE="$MODEL_DIR/models_o${ORDER}.dat"
  echo "Creating .dat file for order $ORDER: $DAT_FILE"

  # Create the .dat file with proper tab spacing
  echo -e "English\t$MODEL_DIR/english${ORDER}.model" > $DAT_FILE
  echo -e "Pidgin\t$MODEL_DIR/pidgin${ORDER}.model" >> $DAT_FILE

  echo "Completed training for order $ORDER"
  echo "========================================"
done

echo "All models trained successfully!"
echo "Models are stored in: $MODEL_DIR"
echo "You can now use the markup tool with the .dat files:"
echo "Example: markup -m $MODEL_DIR/models_o5.dat -i input.txt -o output.txt"