# Code-Switching Detection Between Nigerian Pidgin English and Standard English

## Overview

This project implements an automatic system for detecting and segmenting code-switching between Nigerian Pidgin English (NPE) and Standard English. It employs multiple modeling paradigms, including traditional machine learning classifiers, Bidirectional LSTM (BiLSTM), and compression-based models using Prediction by Partial Matching (PPM) via the Tawa toolkit.

The system can accurately identify language boundaries in mixed-language text, allowing for fine-grained language analysis and potential applications in multilingual NLP systems.

## Key Features

- **Comprehensive Corpus**: Multi-domain corpus compiled from literary texts, forum discussions, and online media
- **Multiple Model Paradigms**: Implementation of traditional ML, neural, and compression-based approaches
- **Character-Level Evaluation**: Fine-grained evaluation using character-level comparison between model outputs and gold-standard annotations
- **Visualization Tools**: Analysis and visualization of linguistic patterns and model performance

## Project Structure
nlp-pidgin-code-switching/
├── config/           # Configuration files for models and evaluation
├── data/             # Dataset organization
├── models/           # Saved model artifacts
├── notebooks/        # Jupyter notebooks for analysis
├── outputs/          # Model outputs and visualizations
├── scripts/          # Utility scripts for training and evaluation
├── src/              # Source code
├── tests/            # Unit tests
└── docs/             # Documentation
Copy
## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nlp-pidgin-code-switching.git
cd nlp-pidgin-code-switching

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
Usage
Training Models
bashCopy# Train traditional machine learning models
python -m src.main train --model-type traditional

# Train BiLSTM model
python -m src.main train --model-type bilstm

# Train PPM compression models
python -m src.main train --model-type ppm
Evaluation
bashCopy# Evaluate all models
python -m src.main evaluate --all

# Compare model performance
python -m src.main compare-models
Visualizations
bashCopy# Generate visualizations
python -m src.main visualize
Results
The project demonstrates that compression-based models (PPM) offer competitive performance compared to supervised approaches for code-switching detection, with N-gram Random Forest (3-gram and 5-gram) achieving the highest accuracy (0.89). PPM-based models performed exceptionally well, particularly for Pidgin text identification.
ModelAccuracyF1-ScorePrecisionRecallRF (3-gram)0.890.890.880.90PPM (Order 2)0.890.890.890.89LR (6-gram)0.870.870.860.88BiLSTM0.780.770.750.80
Future Work

Expanding the corpus to include more dialectal and regional varieties
Implementing transformer-based approaches for comparison
Integrating models into practical applications like speech recognition or educational tools

License
This project is licensed under the MIT License - see the LICENSE file for details.
Citation
Copy@article{ogbonda2025detecting,
  title={Automatically Detecting Code-Switching Between Nigerian Pidgin English and English},
  author={Ogbonda, Godspower},
  journal={},
  year={2025},
}
Contact

Godspower Ogbonda - ogbondagodspower30@gmail.com

Copy
Now, let's create a proper setup.py file that will make your project installable:

### setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nlp-pidgin-code-switching",
    version="0.1.0",
    author="Godspower Ogbonda",
    author_email="ogbondagodspower30@gmail.com",
    description="Code-switching detection between Nigerian Pidgin English and Standard English",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp-pidgin-code-switching",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            "nlp-pidgin=src.main:main",
        ],
    },
)