# NLP-Pidgin-Code-Switching: Usage Guide

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
