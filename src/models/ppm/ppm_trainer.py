#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPM (Prediction by Partial Matching) model training using the TAWA toolkit
for Nigerian Pidgin/English code-switching detection.

This implementation follows the exact training parameters from the original script.
"""

import os
import subprocess
import time
import yaml
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logging_utils import setup_logger
from src.utils.file_io import ensure_dir

logger = setup_logger('ppm_trainer')

def load_config(config_path):
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default settings.")
        return {
            "data": {
                "english_corpus": "data/corpus/english.txt",
                "pidgin_corpus": "data/corpus/pidgin.txt"
            },
            "ppm": {
                "tawa_models_dir": "models/ppm/model_files",
                "train_cmd": "/c/Users/ogbonda/Documents/Tawa-1.0.2/Tawa-1.0/apps/train/train",
                "languages": [
                    {"name": "English", "model_prefix": "english"},
                    {"name": "Pidgin", "model_prefix": "pidgin"}
                ],
                "training": {
                    "orders": [2, 3, 4, 5, 6, 7, 8],
                    "params": {
                        "memory": 1000000,
                        "alpha": 256,
                        "exclusion": "D"
                    }
                }
            }
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_ppm_model(input_file, output_model, order, params, tawa_train_cmd):
    try:
        if not os.path.exists(tawa_train_cmd):
            logger.error(f"Tawa train command not found: {tawa_train_cmd}")
            return False
        cmd_parts = [
            tawa_train_cmd,
            "-i", input_file,
            "-o", output_model,
            "-S",
            "-p", str(params.get('memory', 1000000)),
            "-a", str(params.get('alpha', 256)),
            "-O", str(order),
            "-e", params.get('exclusion', 'D'),
            "-T", f"Order {order} {os.path.basename(input_file).split('.')[0].capitalize()}"
        ]
        command = " ".join(cmd_parts)
        logger.info(f"Running command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"Successfully trained order {order} model: {output_model}")
            return True
        else:
            logger.error(f"Training process failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        return False

def create_dat_file(output_dir, order, english_model, pidgin_model):
    dat_file = os.path.join(output_dir, f"models_o{order}.dat").replace("\\", "/")
    try:
        with open(dat_file, 'w') as f:
            f.write(f"English\t{english_model}\n")
            f.write(f"Pidgin\t{pidgin_model}\n")
        logger.info(f"Created .dat file for order {order}: {dat_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating .dat file: {e}")
        return False

def train_ppm_models(config_path, timestamp=None):
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    config = load_config(config_path)
    english_corpus = config['data']['english_corpus']
    pidgin_corpus = config['data']['pidgin_corpus']
    if not os.path.exists(english_corpus):
        logger.error(f"English corpus file not found: {english_corpus}")
        return False
    if not os.path.exists(pidgin_corpus):
        logger.error(f"Pidgin corpus file not found: {pidgin_corpus}")
        return False
    output_dir = config['ppm']['tawa_models_dir']
    orders = config['ppm']['training']['orders']
    params = config['ppm']['training']['params']
    tawa_train_cmd = config['ppm'].get('train_cmd', 'train')
    ensure_dir(output_dir)
    for order in orders:
        logger.info(f"=== Training models with order {order} ===")
        pidgin_model_path = os.path.join(output_dir, f"pidgin{order}.model").replace("\\", "/")
        english_model_path = os.path.join(output_dir, f"english{order}.model").replace("\\", "/")
        pidgin_success = train_ppm_model(pidgin_corpus, pidgin_model_path, order, params, tawa_train_cmd)
        english_success = train_ppm_model(english_corpus, english_model_path, order, params, tawa_train_cmd)
        if pidgin_success and english_success:
            create_dat_file(output_dir, order, english_model_path, pidgin_model_path)
        else:
            logger.warning(f"Skipping .dat file creation for order {order} due to training failures")
    logger.info(f"All PPM model training complete. Models saved to {output_dir}")
    return True

if __name__ == "__main__":
    train_ppm_models("config/config.yaml")
