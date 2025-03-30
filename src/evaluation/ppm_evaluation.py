"""
Evaluation script for PPM (Prediction by Partial Matching) models
for Nigerian Pidgin/English code-switching detection.
"""

import os
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import yaml
import logging
import re

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load evaluation configuration"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default settings.")
        return {
            "data": {
                "ground_truth": "data/corpus/ground_truth/tawa_ground_truth.txt",
                "test_data": "data/test/ppm_test.txt"
            },
            "ppm": {
                "orders": [2, 3, 4, 5, 6, 7, 8],
                "models_dir": "models/ppm/tawa_models"
            },
            "evaluation": {
                "output_dir": "outputs/predictions/ppm/markup"
            }
        }

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_data(config):
    """Load test data and ground truth"""
    try:
        test_path = config['data']['test_data']
        ground_truth_path = config['data']['ground_truth']

        with open(test_path, 'r', encoding='utf-8') as f:
            test_text = f.read()

        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read()

        # Extract text and labels from ground truth
        # This is simplified and would be replaced by your actual extraction logic
        from src.utils.text_processing import extract_tagged_text
        text, true_labels = extract_tagged_text(ground_truth)

        return test_text, text, true_labels
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise


def run_ppm_markup(test_file, output_file, english_model, pidgin_model, order):
    """Run the TAWA markup process using PPM models"""
    # This would use your specific TAWA command-line interface
    # This is a placeholder - replace with actual TAWA command syntax
    try:
        # Example command structure - adjust based on your actual TAWA toolkit
        command = [
            "tawa", "markup",
            "-i", test_file,
            "-o", output_file,
            "-me", english_model,
            "-mp", pidgin_model
        ]

        # Run the process
        logger.info(f"Running markup with order {order} models")
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Check if successful
        if result.returncode == 0:
            logger.info(f"Successfully ran markup with order {order}")
            return True
        else:
            logger.error(f"Markup process failed: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running markup command: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in markup process: {str(e)}")
        return False


def evaluate_markup_output(markup_file, true_labels, order, output_dir, timestamp):
    """Evaluate the markup output against ground truth"""
    try:
        # Read markup output
        with open(markup_file, 'r', encoding='utf-8') as f:
            markup_text = f.read()

        # Extract text and predicted labels
        from src.utils.text_processing import extract_tagged_text
        text, pred_labels = extract_tagged_text(markup_text)

        # Ensure labels are of equal length for comparison
        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, pos_label='P', average='binary')
        recall = recall_score(true_labels, pred_labels, pos_label='P', average='binary')
        f1 = f1_score(true_labels, pred_labels, pos_label='P', average='binary')

        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=['E', 'P'])

        # Save results
        results_dir = f"{output_dir}/markup_output{order}"
        os.makedirs(results_dir, exist_ok=True)

        # Save metrics
        metrics = {
            'model': f"PPM_Order_{order}",
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Save as CSV
        pd.DataFrame([metrics]).to_csv(f"{results_dir}/metrics_{timestamp}.csv", index=False)

        # Create and save confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['English', 'Pidgin'],
                    yticklabels=['English', 'Pidgin'])
        plt.title(f"Confusion Matrix - PPM Order {order}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/confusion_matrix_{timestamp}.png")
        plt.close()

        # Save comparison to ground truth
        with open(f"{results_dir}/markup_output{order}_vs_ground_truth.txt", 'w', encoding='utf-8') as f:
            f.write(f"Ground Truth: {true_labels[:100]}...\n")
            f.write(f"Predictions:  {pred_labels[:100]}...\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")

        logger.info(f"Evaluation results for PPM Order {order}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating markup output for order {order}: {str(e)}")
        return None


def evaluate_ppm(config_path, timestamp=None):
    """Evaluate all PPM models"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    config = load_config(config_path)

    # Load test data
    test_text, ground_truth_text, true_labels = load_test_data(config)

    # Temporary test file
    test_file = f"temp_test_{timestamp}.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_text)

    # Get model parameters
    orders = config['ppm']['orders']
    models_dir = config['ppm']['models_dir']
    output_dir = config['evaluation']['output_dir']

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each order
    results = []

    for order in orders:
        # Define model paths
        english_model = f"{models_dir}/english{order}.model"
        pidgin_model = f"{models_dir}/pidgin{order}.model"

        # Check if models exist
        if not (os.path.exists(english_model) and os.path.exists(pidgin_model)):
            logger.warning(f"Models for order {order} not found, skipping")
            continue

        # Define output markup file
        markup_file = f"{output_dir}/markup_output{order}.txt"

        # Run markup process
        success = run_ppm_markup(test_file, markup_file, english_model, pidgin_model, order)

        if success and os.path.exists(markup_file):
            # Evaluate the markup output
            metrics = evaluate_markup_output(markup_file, true_labels, order, output_dir, timestamp)

            if metrics:
                results.append(metrics)

    # Clean up temporary file
    if os.path.exists(test_file):
        os.remove(test_file)

    # Compile overall results
    if results:
        results_df = pd.DataFrame(results)

        # Save summary
        summary_dir = "results_summary/ppm_models"
        os.makedirs(summary_dir, exist_ok=True)

        results_df.to_csv(f"{summary_dir}/evaluation_results_{timestamp}.csv", index=False)

        # Create summary report
        with open(f"{summary_dir}/evaluation_summary_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(f"PPM Model Evaluation Summary - {timestamp}\n")
            f.write("=" * 50 + "\n\n")

            # Overall best model
            best_model_idx = results_df['accuracy'].idxmax()
            best_model = results_df.iloc[best_model_idx]

            f.write(f"Best model by accuracy: {best_model['model']}\n")
            f.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
            f.write(f"F1-Score: {best_model['f1_score']:.4f}\n\n")

            # Individual model summaries
            f.write("Individual Model Performance:\n")
            f.write("-" * 30 + "\n")

            for i, row in results_df.iterrows():
                f.write(f"Model: {row['model']}\n")
                f.write(f"  Accuracy:  {row['accuracy']:.4f}\n")
                f.write(f"  Precision: {row['precision']:.4f}\n")
                f.write(f"  Recall:    {row['recall']:.4f}\n")
                f.write(f"  F1-Score:  {row['f1_score']:.4f}\n\n")

        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='accuracy', data=results_df)
        plt.title('PPM Model Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/accuracy_comparison_{timestamp}.png")
        plt.close()

        logger.info(f"PPM evaluation complete. Summary saved to {summary_dir}")

    else:
        logger.warning("No PPM model evaluation results were generated")

    return results


if __name__ == "__main__":
    # Configure logging for stand-alone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # When run directly, use default config path
    evaluate_ppm("config/evaluation_config.yaml")