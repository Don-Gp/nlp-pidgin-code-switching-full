"""
tests/test_ppm_models.py

Integration tests for Tawa (PPM) models using real Tawa commands.
This test will:
  1. Train new PPM models (orders 2..5 by default) using train_ppm_models function
  2. Use PPMPredictor to markup a small sample text
  3. Confirm the markup output is created
NOTE: This requires Tawa to be installed and the `train` and `markup` commands on your PATH.
"""

import os
import pytest

from src.models.ppm.ppm_trainer import train_ppm_models
from src.models.ppm.ppm_predictor import PPMPredictor
from src.utils.file_io import write_file, read_file, ensure_dir

@pytest.fixture
def sample_ppm_test_text():
    """
    A small piece of text containing both English and Pidgin
    that we will markup. Feel free to edit.
    """
    return """Make we see how this text be. 
    I think it will test code-switching well. 
    Sometimes e go be pure English. Other times, we dey pidgin."""


def test_ppm_training_and_markup(temp_dir, sample_ppm_test_text):
    """
    This test uses the real Tawa CLI to:
      1) Train PPM models with a separate test config
      2) Create a small input text
      3) Call PPMPredictor.markup_text() using one of the trained orders
    All new models will be written to models_test/ppm (as per test_config.yaml).
    """

    # 1. Train PPM models using a dedicated test config that writes to models_test/ppm
    config_path = "config/test_config.yaml"  # Make sure this file exists (see below)
    train_success = train_ppm_models(config_path=config_path)
    assert train_success, "PPM training failed"

    # 2. Write out a sample input text to the temp_dir
    sample_file = os.path.join(temp_dir, "sample_ppm_input.txt")
    write_file(sample_file, sample_ppm_test_text)

    # 3. Use PPMPredictor to do markup on this text
    predictor = PPMPredictor(config_path=config_path)
    # We'll pick order=5 here. You can pick any order from your test_config.yaml (2..5).
    markup_output = predictor.markup_text(sample_ppm_test_text, order=5)

    # 4. Basic asserts
    assert markup_output, "Markup output is empty"
    # Check if <english> or <pidgin> appear (Tawa might guess the entire text as English or Pidgin though)
    # This is just a mild assertion to see if it tries to parse code-switching.
    # Real logic can vary, especially with short text. 
    assert ("<english>" in markup_output.lower() or "<pidgin>" in markup_output.lower()), \
        f"Output does not contain English/Pidgin tags.\nOutput:\n{markup_output}"

    # If you want to write the markup to a file:
    markup_file = os.path.join(temp_dir, "ppm_marked_output.txt")
    write_file(markup_file, markup_output)

    print(f"\nPPM markup written to: {markup_file}")
    print(f"\nModels were trained in: models_test/ppm (per test_config.yaml)")
