#!/bin/bash
# Script to train all models (PPM, ML, and BiLSTM)

echo "========== Starting Model Training =========="
echo "$(date): Training started"

# Directory paths
SRC_DIR="src"
MODELS_DIR="models"
DATA_DIR="data"

# Train PPM models
echo "---------- Training PPM models ----------"
python -m ${SRC_DIR}.models.ppm.ppm_trainer

# Train traditional ML models
echo "---------- Training traditional ML models ----------"
python -m ${SRC_DIR}.models.traditional.ml_models

# Train BiLSTM model
echo "---------- Training BiLSTM model ----------"
python -m ${SRC_DIR}.models.neural.bilstm_model

echo "========== Training Completed =========="
echo "$(date): All models have been trained"
echo "Models saved in ${MODELS_DIR} directory"