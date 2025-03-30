#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPM Model Predictor module for code-switching detection.
This module handles the prediction and markup functionality
using the pre-trained PPM models.
"""

import os
import sys
import subprocess
import logging
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.logging_utils import setup_logger
from src.utils.file_io import read_text_file, write_text_file

logger = setup_logger('ppm_predictor')

class PPMPredictor:
    """
    Class for loading PPM models and making predictions on text.
    Uses the Tawa toolkit for prediction and markup.
    """

    def __init__(self, config_path='config/ppm_config.yaml'):
        """
        Initialize the PPM predictor with configuration.
        """
        self.config = self._load_config(config_path)
        self.models_dir = self.config['ppm']['tawa_models_dir']
        self.languages = self.config['ppm']['languages']
        self.orders = self.config['ppm']['training']['orders']
        self.markup_cmd = self.config['ppm'].get('markup_cmd', 'markup')
        self._validate_models()

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _validate_models(self):
        missing_models = []
        for language in self.languages:
            for order in self.orders:
                model_path = os.path.join(self.models_dir, f"{language['model_prefix']}{order}.model").replace("\\", "/")
                if not os.path.exists(model_path):
                    missing_models.append(model_path)
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.warning("Some models may need to be trained first.")

    def markup_text(self, text, order=5, output_file=None):
        temp_input = "temp_input.txt"
        temp_output = "temp_output.txt"
        try:
            write_text_file(temp_input, text)
            models_str = " ".join([
                f"-m{language['name']}:" + os.path.join(self.models_dir, f"{language['model_prefix']}{order}.model").replace("\\", "/")
                for language in self.languages
            ])
            command = f"{self.markup_cmd} -i {temp_input} -o {temp_output} {models_str} -d0"
            logger.info(f"Running command: {command}")
            process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.returncode != 0:
                logger.error(f"Error in markup process: {process.stderr}")
                return None
            marked_up_text = read_text_file(temp_output)
            if output_file:
                write_text_file(output_file, marked_up_text)
                logger.info(f"Marked up text written to {output_file}")
            return marked_up_text
        except Exception as e:
            logger.error(f"Error in markup process: {e}")
            return None
        finally:
            for temp_file in [temp_input, temp_output]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def predict_language(self, text, order=5):
        probabilities = {}
        for language in self.languages:
            model_path = os.path.join(self.models_dir, f"{language['model_prefix']}{order}.model").replace("\\", "/")
            command = f"test -m {model_path} -t"
            process = subprocess.run(command, shell=True, input=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.returncode != 0:
                logger.error(f"Error in test process for {language['name']}: {process.stderr}")
                continue
            try:
                bits_per_char = float(process.stdout.split()[0])
                probabilities[language['name']] = -bits_per_char
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing test output for {language['name']}: {e}")
        if probabilities:
            return max(probabilities, key=probabilities.get)
        else:
            logger.warning("No valid probability estimates obtained")
            return None

if __name__ == "__main__":
    predictor = PPMPredictor()
    text = "How mama be today? You no sabi book but you sabi plenty thing wey pass book, my dear girl what a waste of effort."
    marked_up = predictor.markup_text(text, order=5, output_file="example_markup.txt")
    print("Marked up text:")
    print(marked_up)
