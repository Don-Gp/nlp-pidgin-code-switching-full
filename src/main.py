#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Nigerian Pidgin English Code-Switching detection project.
This module orchestrates the various components of the project.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules
from src.data.preprocessing import TextPreprocessor
from src.models.ppm.ppm_predictor import PPMPredictor
from src.models.ppm.ppm_trainer import train_ppm_models
from src.models.traditional.ml_models import train_ml_models  # Updated import
from src.models.neural.bilstm_model import train_bilstm
from src.evaluation.metrics import evaluate_models
from src.utils.logging_utils import setup_logger
from src.utils.file_io import ensure_dir, read_text_file, write_text_file

# Initialize logger
logger = setup_logger('main')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Nigerian Pidgin English Code-Switching Detection")

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'predict', 'evaluate', 'pipeline'],
                        help="Operation mode: train, predict, evaluate, or full pipeline")

    # Model type selection
    parser.add_argument('--model', type=str, default='all',
                        choices=['ppm', 'ngram', 'bilstm', 'all'],
                        help="Model type to use")

    # Configuration
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help="Path to configuration file")

    # Data inputs
    parser.add_argument('--input', type=str, help="Input file or directory")
    parser.add_argument('--output', type=str, help="Output file or directory")

    # Training parameters
    parser.add_argument('--orders', type=str, default='2,3,4,5,6,7,8',
                        help="PPM model orders (comma-separated)")
    parser.add_argument('--ngram', type=str, default='1,2,3,4,5',
                        help="Character n-gram sizes (comma-separated)")

    # Evaluation parameters
    parser.add_argument('--ground-truth', type=str,
                        help="Ground truth file for evaluation")

    # Execution options
    parser.add_argument('--fast', action='store_true', help="Run in fast mode with reduced complexity")
    parser.add_argument('--super-fast', action='store_true', help="Run in super fast mode (minimal models)")

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default settings.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}


def train_models(args, config):
    """Train models based on configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting model training with timestamp {timestamp}")

    # Update config with command line arguments
    if args.fast:
        config['ml']['fast_mode'] = True
    if args.super_fast:
        config['ml']['super_fast'] = True

    # Train models based on selection
    if args.model in ['ppm', 'all']:
        logger.info("Training PPM models...")
        ppm_config = config.get('ppm', {})
        orders = [int(o) for o in args.orders.split(',')]
        ppm_config['training'] = ppm_config.get('training', {})
        ppm_config['training']['orders'] = orders
        train_ppm_models(config_path=args.config, timestamp=timestamp)

    if args.model in ['ngram', 'all']:
        logger.info("Training traditional ML models...")
        train_ml_models(config_path=args.config, timestamp=timestamp)

    if args.model in ['bilstm', 'all']:
        logger.info("Training BiLSTM model...")
        train_bilstm(config_path=args.config, timestamp=timestamp)

    logger.info("Model training complete!")
    return timestamp


def predict_with_models(args, config):
    """Make predictions using trained models."""
    if not args.input:
        logger.error("Input file must be specified for prediction")
        return False

    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} not found")
        return False

    if not args.output:
        # Create default output path based on input
        input_base = os.path.basename(args.input)
        args.output = f"outputs/predictions/{input_base}_predictions.txt"

    # Ensure output directory exists
    ensure_dir(os.path.dirname(args.output))

    # Read input text
    input_text = read_text_file(args.input)

    # Predict with PPM models
    if args.model in ['ppm', 'all']:
        logger.info("Predicting with PPM models...")
        ppm_predictor = PPMPredictor(config_path=args.config)
        order = 5  # Default order
        if args.orders:
            # Use first order in list
            order = int(args.orders.split(',')[0])

        ppm_output = ppm_predictor.markup_text(input_text, order=order)
        if ppm_output:
            # Save PPM output
            ppm_output_path = f"{args.output}.ppm.txt"
            write_text_file(ppm_output_path, ppm_output)
            logger.info(f"PPM predictions saved to {ppm_output_path}")
        else:
            logger.error("Failed to make PPM predictions")

    # TODO: Add prediction with ML models
    # For now, we'll use PPM models as the main prediction method

    logger.info(f"Predictions completed")
    return True


def evaluate_model_outputs(args, config):
    """Evaluate model outputs against ground truth."""
    if not args.ground_truth:
        logger.error("Ground truth file must be specified for evaluation")
        return False

    if not args.input:
        logger.error("Input directory with model outputs must be specified")
        return False

    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file {args.ground_truth} not found")
        return False

    if not os.path.exists(args.input):
        logger.error(f"Input directory/file {args.input} not found")
        return False

    output_dir = args.output or "outputs/evaluation"
    ensure_dir(output_dir)

    # Call evaluation module
    results = evaluate_models(
        ground_truth_file=args.ground_truth,
        model_outputs_path=args.input,
        output_dir=output_dir
    )

    logger.info(f"Evaluation results saved to {output_dir}")
    return results


def run_full_pipeline(args, config):
    """Run the full pipeline: train, predict, and evaluate."""
    # 1. Train models
    timestamp = train_models(args, config)

    # 2. Make predictions
    if args.input:
        predict_with_models(args, config)

    # 3. Evaluate if ground truth is available
    if args.ground_truth and args.input:
        evaluate_model_outputs(args, config)

    logger.info("Full pipeline completed successfully")
    return True


def main():
    """Main function to orchestrate the project."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting Nigerian Pidgin English Code-Switching Detection in {args.mode} mode")

    # Run in selected mode
    if args.mode == 'train':
        train_models(args, config)
    elif args.mode == 'predict':
        predict_with_models(args, config)
    elif args.mode == 'evaluate':
        evaluate_model_outputs(args, config)
    elif args.mode == 'pipeline':
        run_full_pipeline(args, config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Record end time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Completed in {duration}")


if __name__ == "__main__":
    main()
