# NLP-Pidgin-Code-Switching: System Architecture

This document outlines the architecture of the Nigerian Pidgin English Code-Switching detection system, providing an overview of the components, their interactions, and the data flow throughout the system.

## System Overview

The architecture follows a modular design with clear separation of concerns, making it easier to maintain, extend, and understand. The system comprises several core components:

1. **Data Management**: Handles corpus creation, preprocessing, and management
2. **Model Training**: Implements various modeling approaches for language identification
3. **Evaluation**: Provides comprehensive performance assessment
4. **Visualization**: Generates insights through visualizations and reports

## Component Architecture

### 1. Data Management

**Core functionality**: Building and managing corpora for Nigerian Pidgin English and Standard English, with proper preprocessing and transformation.

**Key components**:
- `src/data/corpus_builder.py`: Creates, manages, and manipulates corpus data
- `src/data/preprocessing.py`: Cleans, normalizes, and prepares text data

**Data flow**:
1. Raw text collected from various sources (literature, web, etc.)
2. Preprocessing applied (cleaning, normalization)
3. Structured corpus created with language annotations
4. Data split into training and testing sets

### 2. Model Training

**Core functionality**: Training language models using multiple paradigms to identify and mark code-switching boundaries.

**Key components**:
- `src/models/ppm/`: Prediction by Partial Matching (PPM) compression-based models
  - `ppm_trainer.py`: Trains PPM models using the Tawa toolkit
  - `ppm_predictor.py`: Makes predictions using trained PPM models
  
- `src/models/traditional/`: Traditional machine learning models
  - `ml_models.py`: Implements various character and word n-gram models
  
- `src/models/neural/`: Deep learning approaches
  - `bilstm_model.py`: Bidirectional LSTM model for sequential labeling

**Data flow**:
1. Preprocessed data fed into model training pipelines
2. Models trained with appropriate parameters
3. Trained models saved to disk for later use
4. Models applied to test data for evaluation

### 3. Evaluation

**Core functionality**: Assessing model performance using various metrics and comparative analysis.

**Key components**:
- `src/evaluation/ml_evaluation.py`: Evaluates machine learning models
- `src/evaluation/ppm_evaluation.py`: Evaluates PPM models
- `src/evaluation/metrics.py`: Calculates and analyzes performance metrics

**Data flow**:
1. Models generate predictions on test data
2. Predictions compared with ground truth
3. Metrics calculated (accuracy, precision, recall, F1-score)
4. Results aggregated and summarized
5. Visualizations generated for analysis

### 4. Visualization

**Core functionality**: Generating visual representations of results and insights.

**Key components**:
- `src/visualization/plots.py`: Creates charts, graphs, and visual reports

**Data flow**:
1. Evaluation results fed into visualization tools
2. Visual representations generated
3. Comprehensive reports created

## Cross-Cutting Concerns

- **Configuration Management**: YAML configuration files in the `config/` directory
- **Logging**: Centralized logging with `logging_utils.py`
- **File I/O**: Common file operations in `file_io.py`
- **Text Processing**: Shared text manipulation utilities in `text_processing.py`

## System Interaction

The system components interact through well-defined interfaces:
Data Sources] → [Data Management] → [Model Training] → [Evaluation] → [Visualization] → [Reports/Insights]
Copy
Each component can be used independently or as part of the complete pipeline, allowing for flexibility in system usage.

## Deployment View

The system is designed as a Python package that can be:
1. Run as a standalone application
2. Integrated into larger NLP pipelines
3. Extended with new models or evaluation methods

All components are accessible through the main entry point (`src/main.py`), which provides a unified interface for the entire system.

## Future Extensions

The architecture allows for several extension points:
1. Adding new model types in their respective directories
2. Incorporating additional evaluation metrics in the evaluation module
3. Extending preprocessing capabilities for more complex text normalization
4. Integrating with web services for online text processing

This modular design ensures that the system can evolve to meet future requirements without major architectural changes.
10. Creating docs/usage.md
markdownCopy# NLP-Pidgin-Code-Switching: Usage Guide

This document provides comprehensive instructions for using the Nigerian Pidgin English Code-Switching detection system.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training Models](#training-models)
4. [Making Predictions](#making-predictions)
5. [Evaluation](#evaluation)
6. [Visualization](#visualization)
7. [Command-Line Interface](#command-line-interface)
8. [Common Use Cases](#common-use-cases)
9. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- For PPM models: Tawa toolkit installed and available in PATH

### Installing the Package

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nlp-pidgin-code-switching.git
   cd nlp-pidgin-code-switching

Install dependencies:
bashCopypip install -r requirements.txt

Install the package in development mode:
bashCopypip install -e .


Data Preparation
Using Existing Corpora
The package comes with sample data in the data/ directory:

data/corpus/english.txt: English corpus
data/corpus/pidgin.txt: Nigerian Pidgin corpus
data/corpus/mixed.txt: Mixed text with code-switching

Preparing Your Own Data

Place your raw text files in the data/raw/ directory.
Use the preprocessing module to clean and prepare your data:
pythonCopyfrom src.data.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
preprocessor.process_file(
    "data/raw/my_text.txt", 
    "data/interim/my_text_clean.txt",
    lowercase=True,
    remove_urls=True
)

For creating a tagged corpus with language labels:
pythonCopy# Automatic tagging (basic heuristic approach)
tagged_text = preprocessor.tag_potential_pidgin("Your text with code-switching")

# Or manually annotate text with <english> and <pidgin> tags

Extract separate language corpora from tagged text:
pythonCopypreprocessor.extract_tagged_sections(
    "data/interim/tagged_text.txt",
    pidgin_output="data/corpus/extracted_pidgin.txt",
    english_output="data/corpus/extracted_english.txt"
)


Training Models
Training All Models
The simplest way to train all models is using the provided script:
bashCopybash scripts/train_all_models.sh
Training Specific Models
PPM Models
pythonCopyfrom src.models.ppm.ppm_trainer import PPMTrainer

trainer = PPMTrainer()
trainer.train_models(orders=[2, 3, 4, 5])
Traditional ML Models
pythonCopyfrom src.models.traditional.ml_models import train_traditional_models

train_traditional_models(
    ngram_range=(1, 8),
    models=['svm', 'naive_bayes', 'logistic_regression', 'random_forest']
)
BiLSTM Model
pythonCopyfrom src.models.neural.bilstm_model import train_bilstm_model

train_bilstm_model(
    input_file="data/corpus/mixed.txt",
    epochs=10,
    batch_size=32
)
Making Predictions
Using PPM Models
pythonCopyfrom src.models.ppm.ppm_predictor import PPMPredictor

predictor = PPMPredictor()
marked_up_text = predictor.markup_text(
    "How mama be today? You no sabi book but you sabi plenty thing wey pass book, my dear girl what a waste of effort.",
    order=5,
    output_file="outputs/predictions/example_prediction.txt"
)
Using ML Models
pythonCopyfrom src.models.traditional.ml_models import predict_language

predictions = predict_language(
    text="How mama be today?",
    model_type="random_forest",
    ngram_size=3
)
Evaluation
Evaluating All Models
To evaluate all models at once:
bashCopybash scripts/evaluate_all_models.sh
Evaluating Specific Models
pythonCopyfrom src.evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()
results = metrics.calculate_metrics_from_files(
    true_file="data/corpus/ground_truth/ml_ground_truth.txt",
    pred_file="outputs/predictions/ml/markup/test_ngram_3_logistic_regression.txt",
    char_level=True
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-score: {results['f1_score']:.4f}")
Generating Reports
pythonCopymetrics.generate_summary_report(
    results,
    output_file="outputs/results_summary/model_evaluation_report.txt"
)
Visualization
Creating Performance Plots
pythonCopyfrom src.visualization.plots import create_performance_plots

create_performance_plots(
    results_file="outputs/results_summary/model_evaluation_summary.csv",
    output_dir="outputs/visualizations/ml_model_training",
    metrics=["accuracy", "f1_score"]
)
Generating Confusion Matrices
pythonCopyfrom src.visualization.plots import plot_confusion_matrix

plot_confusion_matrix(
    true_labels=true_labels,
    pred_labels=pred_labels,
    labels=["English", "Pidgin"],
    output_file="outputs/visualizations/confusion_matrix.png"
)
Command-Line Interface
The package provides a command-line interface for common operations:
Training
bashCopypython -m src.main train --model ppm --orders 2 3 4 5
python -m src.main train --model ml --ngram-range 1 8
python -m src.main train --model bilstm --epochs 10
Evaluation
bashCopypython -m src.main evaluate --model ppm --orders 2 3 4 5
python -m src.main evaluate --model ml --ngram-range 1 8
python -m src.main evaluate --model bilstm
Prediction
bashCopypython -m src.main predict --model ppm --input-file input.txt --output-file output.txt
python -m src.main predict --model ml --input-file input.txt --output-file output.txt
Common Use Cases
Complete Pipeline
For running the complete pipeline from preprocessing to visualization:
bashCopy# Preprocess data
python -m src.data.preprocessing --input data/raw/text.txt --output data/interim/clean.txt

# Train models
bash scripts/train_all_models.sh

# Evaluate models
bash scripts/evaluate_all_models.sh

# Generate reports
bash scripts/generate_report.sh
Working with Web Data
To process and analyze web content with code-switching:
pythonCopyimport requests
from src.data.preprocessing import TextPreprocessor
from src.models.ppm.ppm_predictor import PPMPredictor

# Fetch content
response = requests.get("https://example.com/pidgin-content")
text = response.text

# Preprocess
preprocessor = TextPreprocessor()
clean_text = preprocessor.clean_text(text, remove_urls=True)

# Detect code-switching
predictor = PPMPredictor()
marked_up = predictor.markup_text(clean_text, order=5)

# Save results
with open("outputs/web_analysis.txt", "w") as f:
    f.write(marked_up)
Troubleshooting
Common Issues

Missing models error: Ensure you've trained the models before evaluation or prediction.
bashCopybash scripts/train_all_models.sh

File not found errors: Check that all required data files are in the correct locations.
Tawa toolkit errors: Verify that the Tawa toolkit is properly installed and accessible in your PATH.
Performance issues: For large files, consider processing in chunks or using the streaming APIs.

Getting Help
If you encounter any issues not covered in this guide, please:

Check the logs in the logs/ directory
Review the error messages and stack traces
Consult the documentation for more details
Submit an issue on the GitHub repository


### 11. Enhancing README.md

```markdown
# Nigerian Pidgin English Code-Switching Detection

A comprehensive toolkit for detecting and analyzing code-switching between Nigerian Pidgin English (NPE) and Standard English using machine learning and compression-based approaches.

## Project Overview

This project addresses the challenge of automatically identifying code-switching between Nigerian Pidgin English and Standard English, a common linguistic practice in Nigerian multilingual contexts. Given the lack of standardization and regional variability of NPE, traditional NLP approaches face significant challenges. 

Our solution employs multiple modeling paradigms:
- Traditional machine learning with character and word n-grams
- Compression-based models using Prediction by Partial Matching (PPM)
- Deep learning with Bidirectional LSTMs

## Key Features

- **Corpus Management**: Tools for building, cleaning, and preparing corpora for NPE and English
- **Multi-paradigm Modeling**: Integration of traditional ML, compression-based, and deep learning approaches
- **Fine-grained Evaluation**: Character-level evaluation with comprehensive metrics
- **Visualization**: Tools for analyzing and visualizing language patterns and model performance
- **Extensible Architecture**: Modular design allowing for easy integration of new models and techniques

## Project Structure
nlp-pidgin-code-switching/
├── config/                 # Configuration files
├── data/                   # Corpus and data files
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for analysis
├── outputs/                # Model outputs and results
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # Model implementations
│   ├── evaluation/         # Evaluation modules
│   ├── visualization/      # Visualization tools
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
└── README.md               # This file
Copy
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nlp-pidgin-code-switching.git
   cd nlp-pidgin-code-switching

Install the required dependencies:
bashCopypip install -r requirements.txt

For PPM models, install the Tawa toolkit:
bashCopy# Follow installation instructions from the Tawa documentation


Quick Start
Training Models
To train all models with default settings:
bashCopybash scripts/train_all_models.sh
Evaluating Models
To evaluate all trained models:
bashCopybash scripts/evaluate_all_models.sh
Generating Reports
To generate comprehensive evaluation reports:
bashCopybash scripts/generate_report.sh
Example Usage
Detecting Code-Switching in Text
pythonCopyfrom src.models.ppm.ppm_predictor import PPMPredictor

# Initialize predictor
predictor = PPMPredictor()

# Example text with code-switching
text = "How mama be today? You no sabi book but you sabi plenty thing wey pass book, my dear girl what a waste of effort."

# Markup code-switching points
marked_up = predictor.markup_text(text, order=5)
print(marked_up)
Processing a Corpus
pythonCopyfrom src.data.preprocessing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Process a corpus file
preprocessor.process_file(
    "data/raw/mixed_text.txt",
    "data/interim/processed_text.txt",
    lowercase=True,
    remove_urls=True
)
Model Performance
Our evaluation shows that the different modeling approaches have complementary strengths:
ModelAccuracyF1-ScorePrecisionRecallRandom Forest (3-gram)0.890.890.880.90PPM (Order 2)0.890.890.890.89Logistic Regression (6-gram)0.870.870.860.88BiLSTM0.780.770.750.80
Documentation
For more detailed information, please refer to:

Architecture Overview
Usage Guide

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Tawa Toolkit for the PPM implementation
Contributors to the Nigerian Pidgin English corpus