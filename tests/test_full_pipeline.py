# tests/test_full_pipeline.py
"""
Integration test that runs the entire pipeline (train + evaluate + report)
with a dedicated test config, so we confirm everything works end-to-end.
"""

import os
import subprocess
import pytest

@pytest.mark.full_pipeline
def test_run_full_pipeline():
    """
    Calls the 'src.main' in pipeline mode with --config config/test_config.yaml
    1) Trains all models (PPM, ML, BiLSTM)
    2) Possibly does prediction or evaluation if your pipeline code calls it
    3) Produces outputs in outputs_test/ and logs in logs_test/
    4) Confirm we didn't crash
    """
    # Path to your test config
    config_path = "config/test_config.yaml"

    # We assume your pipeline logic in main.py can handle `--mode pipeline --model all`
    # You can adapt these arguments to match your code if needed
    cmd = [
        "python", "-m", "src.main",
        "--mode", "pipeline",
        "--model", "all",
        "--config", config_path
    ]

    print(f"Running full pipeline with: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check return code
    assert result.returncode == 0, f"Pipeline failed.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    # (Optional) add checks to see if certain files exist now:
    # e.g. did we create a PPM model?
    assert os.path.exists("models_test/ppm/pidgin2.model"), "Did not find the pidgin2.model in test folder!"
    # Did we create an ML model?
    # assert os.path.exists("models_test/traditional/char_2_vectorizer.pkl"), "ML vectorizer not found."

    print("Full pipeline test completed successfully.")
