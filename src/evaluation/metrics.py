#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for Nigerian Pidgin/English code-switching detection.
This module provides comprehensive evaluation of model predictions.
"""

import os
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logging_utils import setup_logger
from src.utils.file_io import ensure_dir, read_text_file, write_text_file

# Set up logger
logger = setup_logger('evaluation')


def normalize_tags(text):
    """Fix tag inconsistencies and standardize format"""
    # Fix incorrect closing tags in model output
    text = text.replace('<\\English>', '</english>')
    text = text.replace('<\\Pidgin>', '</pidgin>')
    text = text.replace('<\\english>', '</english>')
    text = text.replace('<\\pidgin>', '</pidgin>')

    # Normalize case for all tags
    text = re.sub(r'<(\/?)([Pp]idgin)>', r'<\1pidgin>', text)
    text = re.sub(r'<(\/?)([Ee]nglish)>', r'<\1english>', text)

    # Remove whitespace between tags
    text = re.sub(r'</english>\s+<english>', '</english><english>', text)
    text = re.sub(r'</pidgin>\s+<pidgin>', '</pidgin><pidgin>', text)
    text = re.sub(r'</english>\s+<pidgin>', '</english><pidgin>', text)
    text = re.sub(r'</pidgin>\s+<english>', '</pidgin><english>', text)

    return text


def extract_text_without_tags(text):
    """Extract text without any tags for comparing content directly"""
    # Remove all tags
    clean_text = re.sub(r'</?(?:pidgin|english)>', '', text, flags=re.IGNORECASE)
    return clean_text


def create_language_label_map(text):
    """
    Create a map indicating the language label for each character position.
    Improved version with better tag boundary and newline handling.
    """
    # Initialize a map to hold language label for each character
    char_labels = {}

    # Extract text without tags to get the total content length
    text_without_tags = extract_text_without_tags(text)

    # Process multi-line content properly by creating a map from positions in the
    # original text to positions in the text without tags
    orig_to_clean_map = {}
    clean_pos = 0
    in_tag = False

    for i, char in enumerate(text):
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
            continue

        if not in_tag:
            orig_to_clean_map[i] = clean_pos
            clean_pos += 1

    # Process English content
    for match in re.finditer(r'<english>(.*?)</english>', text, re.DOTALL | re.IGNORECASE):
        content = match.group(1)
        start = match.start(1)
        end = match.end(1)

        # Label each character in this block as English
        for pos in range(start, end):
            if pos in orig_to_clean_map:
                clean_pos = orig_to_clean_map[pos]
                char_labels[clean_pos] = 'E'

    # Process Pidgin content
    for match in re.finditer(r'<pidgin>(.*?)</pidgin>', text, re.DOTALL | re.IGNORECASE):
        content = match.group(1)
        start = match.start(1)
        end = match.end(1)

        # Label each character in this block as Pidgin
        for pos in range(start, end):
            if pos in orig_to_clean_map:
                clean_pos = orig_to_clean_map[pos]
                char_labels[clean_pos] = 'P'

    return char_labels


def create_language_label_map_no_whitespace(text):
    """
    Create a language label map for text without whitespace.
    Returns a dict mapping positions in no-whitespace text to language labels.
    """
    # First get original labels
    original_labels = create_language_label_map(text)

    # Extract text without tags and remove whitespace
    text_without_tags = extract_text_without_tags(text)

    # Create map from positions in no-whitespace text to positions in original text
    no_whitespace_text = ""
    position_map = []  # Maps positions in no_whitespace_text to positions in original text

    for i, char in enumerate(text_without_tags):
        if not char.isspace():
            no_whitespace_text += char
            position_map.append(i)

    # Create new label map for no-whitespace text
    no_ws_labels = {}

    for i, original_pos in enumerate(position_map):
        if original_pos in original_labels:
            no_ws_labels[i] = original_labels[original_pos]

    return no_ws_labels, no_whitespace_text


def evaluate_file(ground_truth_file, model_output_file, output_dir, visualize=True):
    """
    Evaluate a single model output file against the ground truth.

    Args:
        ground_truth_file (str): Path to the ground truth file
        model_output_file (str): Path to the model output file
        output_dir (str): Directory to save evaluation results
        visualize (bool): Whether to generate visualizations

    Returns:
        dict: Evaluation metrics
    """
    try:
        # Read files
        logger.info(f"Reading ground truth file: {ground_truth_file}")
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_text = f.read()

        logger.info(f"Reading model output file: {model_output_file}")
        with open(model_output_file, 'r', encoding='utf-8') as f:
            model_text = f.read()

        # Get model name from file path
        model_name = os.path.basename(model_output_file).split('.')[0]

        # Preprocess both files
        gt_text = normalize_tags(gt_text)
        model_text = normalize_tags(model_text)

        # Extract text for content comparison (with whitespace)
        gt_clean = extract_text_without_tags(gt_text)
        model_clean = extract_text_without_tags(model_text)

        # Print character counts with whitespace
        logger.info(f"Character counts after preprocessing:")
        logger.info(f"Ground Truth: {len(gt_clean)} characters")
        logger.info(f"Model Output: {len(model_clean)} characters")

        # Get language labels (with whitespace)
        logger.info("\nCreating language label maps...")
        gt_orig_labels = create_language_label_map(gt_text)
        model_orig_labels = create_language_label_map(model_text)

        # Get text without whitespace and their language labels
        gt_labels, gt_no_ws = create_language_label_map_no_whitespace(gt_text)
        model_labels, model_no_ws = create_language_label_map_no_whitespace(model_text)

        # Print character counts without whitespace
        logger.info(f"\nCharacter counts (without whitespace):")
        logger.info(f"Ground Truth: {len(gt_no_ws)} characters")
        logger.info(f"Model Output: {len(model_no_ws)} characters")

        # Perform alignment when comparing texts
        alignment = {}
        for i in range(min(len(gt_no_ws), len(model_no_ws))):
            alignment[i] = i

        # Initialize metrics calculations
        tp_e = 0  # True positive English
        fp_e = 0  # False positive English
        fn_e = 0  # False negative English
        tp_p = 0  # True positive Pidgin
        fp_p = 0  # False positive Pidgin
        fn_p = 0  # False negative Pidgin
        total_evaluated = 0

        # Compare language labels
        for gt_pos, model_pos in alignment.items():
            if gt_pos in gt_labels and model_pos in model_labels:
                total_evaluated += 1
                gt_label = gt_labels[gt_pos]
                model_label = model_labels[model_pos]

                if gt_label == 'E':
                    if model_label == 'E':
                        tp_e += 1
                    else:
                        fn_e += 1
                        fp_p += 1
                else:  # gt_label == 'P'
                    if model_label == 'P':
                        tp_p += 1
                    else:
                        fn_p += 1
                        fp_e += 1

        # Calculate metrics
        total_correct = tp_e + tp_p
        accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0

        precision_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0
        recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0
        f1_e = 2 * precision_e * recall_e / (precision_e + recall_e) if (precision_e + recall_e) > 0 else 0

        precision_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0
        recall_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0
        f1_p = 2 * precision_p * recall_p / (precision_p + recall_p) if (precision_p + recall_p) > 0 else 0

        precision = (precision_e + precision_p) / 2
        recall = (recall_e + recall_p) / 2
        f1_score = (f1_e + f1_p) / 2

        # Calculate language distribution
        gt_dist_e = sum(1 for label in gt_labels.values() if label == 'E') / len(gt_labels) if gt_labels else 0
        gt_dist_p = sum(1 for label in gt_labels.values() if label == 'P') / len(gt_labels) if gt_labels else 0
        model_dist_e = sum(1 for label in model_labels.values() if label == 'E') / len(model_labels) if model_labels else 0
        model_dist_p = sum(1 for label in model_labels.values() if label == 'P') / len(model_labels) if model_labels else 0

        # Create visualization if requested
        visualization_results = None
        if visualize:
            logger.info("\nGenerating visualization...")
            visualization_dir = os.path.join(output_dir, "visualizations")
            ensure_dir(visualization_dir)

            visualization_results = create_visualization(
                {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'precision_E': precision_e,
                    'recall_E': recall_e,
                    'f1_score_E': f1_e,
                    'precision_P': precision_p,
                    'recall_P': recall_p,
                    'f1_score_P': f1_p,
                    'TP_E': tp_e,
                    'FP_E': fp_e,
                    'FN_E': fn_e,
                    'TP_P': tp_p,
                    'FP_P': fp_p,
                    'FN_P': fn_p,
                    'gt_english_pct': gt_dist_e,
                    'gt_pidgin_pct': gt_dist_p,
                    'model_english_pct': model_dist_e,
                    'model_pidgin_pct': model_dist_p
                },
                model_name,
                visualization_dir
            )

        # Compile results
        result = {
            'model_name': model_name,
            'ground_truth': os.path.basename(ground_truth_file),
            'model_output': os.path.basename(model_output_file),
            'gt_chars': len(gt_clean),
            'model_chars': len(model_clean),
            'gt_chars_no_ws': len(gt_no_ws),
            'model_chars_no_ws': len(model_no_ws),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'precision_E': precision_e,
            'recall_E': recall_e,
            'f1_score_E': f1_e,
            'precision_P': precision_p,
            'recall_P': recall_p,
            'f1_score_P': f1_p,
            'TP_E': tp_e,
            'FP_E': fp_e,
            'FN_E': fn_e,
            'TP_P': tp_p,
            'FP_P': fp_p,
            'FN_P': fn_p,
            'total_evaluated': total_evaluated,
            'correct_predictions': total_correct,
            'gt_english_pct': gt_dist_e,
            'gt_pidgin_pct': gt_dist_p,
            'model_english_pct': model_dist_e,
            'model_pidgin_pct': model_dist_p,
            'gt_labeled_chars': len(gt_labels),
            'model_labeled_chars': len(model_labels)
        }

        # Print results
        logger.info("\nRESULTS SUMMARY FOR: " + model_name)
        logger.info("=" * 80)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Matching positions: {total_correct} out of {total_evaluated}")

        logger.info("\nEnglish Metrics:")
        logger.info(f"  Precision: {precision_e:.4f}")
        logger.info(f"  Recall: {recall_e:.4f}")
        logger.info(f"  F1-Score: {f1_e:.4f}")

        logger.info("\nPidgin Metrics:")
        logger.info(f"  Precision: {precision_p:.4f}")
        logger.info(f"  Recall: {recall_p:.4f}")
        logger.info(f"  F1-Score: {f1_p:.4f}")

        logger.info(f"\nLanguage distribution:")
        logger.info(f"  Model: {model_dist_e:.4f} English, {model_dist_p:.4f} Pidgin")
        logger.info(f"  Ground truth: {gt_dist_e:.4f} English, {gt_dist_p:.4f} Pidgin")

        # Save results to CSV and JSON
        results_dir = os.path.join(output_dir, "results")
        ensure_dir(results_dir)

        # Save as CSV
        results_csv = os.path.join(results_dir, f"{model_name}_evaluation.csv")
        pd.DataFrame([result]).to_csv(results_csv, index=False)

        logger.info(f"\nResults saved to {results_csv}")
        return result

    except Exception as e:
        logger.error(f"Error evaluating {model_output_file}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_visualization(result, model_name, output_dir):
    """Create visualization for the evaluation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(output_dir)

    # Create bar chart for metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [result['accuracy'], result['precision'], result['recall'], result['f1_score']]

    # Bar chart for overall metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1.0)
    plt.title(f'Evaluation Metrics for {model_name}')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

    metrics_chart = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.png")
    plt.savefig(metrics_chart)
    plt.close()

    # Create comparison chart for English vs. Pidgin
    labels = ['Precision', 'Recall', 'F1-Score']
    english_values = [result['precision_E'], result['recall_E'], result['f1_score_E']]
    pidgin_values = [result['precision_P'], result['recall_P'], result['f1_score_P']]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, english_values, width, label='English F1', color='blue')
    plt.bar(x + width / 2, pidgin_values, width, label='Pidgin F1', color='green')

    plt.ylabel('Score')
    plt.title(f'English vs. Pidgin Metrics for {model_name}')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(english_values):
        plt.text(i - width / 2, v + 0.02, f'{v:.4f}', ha='center')

    for i, v in enumerate(pidgin_values):
        plt.text(i + width / 2, v + 0.02, f'{v:.4f}', ha='center')

    comparison_chart = os.path.join(output_dir, f"{model_name}_language_comparison_{timestamp}.png")
    plt.savefig(comparison_chart)
    plt.close()

    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    confmat = np.array([
        [result['TP_E'], result['FP_E']],
        [result['FN_E'], result['TP_P']]
    ])
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.colorbar()

    classes = ['English', 'Pidgin']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')

    thresh = confmat.max() / 2.
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            plt.text(j, i, format(confmat[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confmat[i, j] > thresh else "black")

    confusion_matrix = os.path.join(output_dir, f"{model_name}_confusion_matrix_{timestamp}.png")
    plt.savefig(confusion_matrix)
    plt.close()

    # Create pie chart for language distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    gt_dist = [result['gt_english_pct'], result['gt_pidgin_pct']]
    plt.pie(gt_dist, labels=['English', 'Pidgin'], autopct='%1.1f%%', colors=['blue', 'green'])
    plt.title('Ground Truth Language Distribution')

    plt.subplot(1, 2, 2)
    model_dist = [result['model_english_pct'], result['model_pidgin_pct']]
    plt.pie(model_dist, labels=['English', 'Pidgin'], autopct='%1.1f%%', colors=['blue', 'green'])
    plt.title(f'{model_name} Language Distribution')

    dist_chart = os.path.join(output_dir, f"{model_name}_language_dist_{timestamp}.png")
    plt.savefig(dist_chart)
    plt.close()

    logger.info(f"Created visualization for {model_name}:")
    logger.info(f"- Metrics Chart: {metrics_chart}")
    logger.info(f"- Language Comparison: {comparison_chart}")
    logger.info(f"- Confusion Matrix: {confusion_matrix}")
    logger.info(f"- Language Distribution: {dist_chart}")

    return {
        'metrics_chart': metrics_chart,
        'comparison_chart': comparison_chart,
        'confusion_matrix': confusion_matrix,
        'dist_chart': dist_chart
    }


def evaluate_models(ground_truth_file, model_outputs_path, output_dir):
    """
    Evaluate multiple model outputs against ground truth.

    Args:
        ground_truth_file (str): Path to the ground truth file
        model_outputs_path (str): Path to directory with model outputs
        output_dir (str): Directory to save evaluation results

    Returns:
        list: Evaluation results for all models
    """
    # Create output directory
    ensure_dir(output_dir)

    # Find model output files
    model_files = []
    if os.path.isdir(model_outputs_path):
        model_files = [os.path.join(model_outputs_path, f) for f in os.listdir(model_outputs_path)
                       if f.endswith('.txt') or f.endswith('.ppm.txt')]
    elif os.path.isfile(model_outputs_path):
        model_files = [model_outputs_path]

    if not model_files:
        logger.error(f"No model output files found in {model_outputs_path}")
        return []

    # Evaluate each model output
    results = []
    for model_file in model_files:
        logger.info(f"\nEvaluating model output: {os.path.basename(model_file)}")
        result = evaluate_file(ground_truth_file, model_file, output_dir)
        if result:
            results.append(result)

    # Create summary report
    if results:
        create_summary_report(results, output_dir)

    return results


def create_summary_report(results, output_dir):
    """Create a summary report comparing all models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sort results by accuracy
    sorted_results = sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True)

    # Save to CSV
    summary_csv = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.csv")
    pd.DataFrame(sorted_results).to_csv(summary_csv, index=False)

    # Create a text report
    summary_txt = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.txt")

    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("ENGLISH-PIDGIN CODE-SWITCHING MODEL COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of models compared: {len(results)}\n\n")

        # Write model rankings
        f.write("MODEL RANKING BY ACCURACY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<5}{'Model':<30}{'Accuracy':<10}{'F1-Score':<10}{'Precision':<10}{'Recall':<10}\n")
        f.write("-" * 80 + "\n")

        for rank, result in enumerate(sorted_results, 1):
            model_name = result.get('model_name', 'Unknown')
            f.write(f"{rank:<5}{model_name:<30}{result.get('accuracy', 0):<10.4f}{result.get('f1_score', 0):<10.4f}")
            f.write(f"{result.get('precision', 0):<10.4f}{result.get('recall', 0):<10.4f}\n")

        # Write detailed section for each model
        f.write("\n\nDETAILED MODEL ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for result in sorted_results:
            model_name = result.get('model_name', 'Unknown')
            f.write(f"MODEL: {model_name}\n")
            f.write("-" * 80 + "\n")

            # Overall metrics
            f.write(f"Accuracy: {result.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {result.get('precision', 0):.4f}\n")
            f.write(f"Recall: {result.get('recall', 0):.4f}\n")
            f.write(f"F1-Score: {result.get('f1_score', 0):.4f}\n\n")

            # Language-specific metrics
            f.write("English Metrics:\n")
            f.write(f"  Precision: {result.get('precision_E', 0):.4f}\n")
            f.write(f"  Recall: {result.get('recall_E', 0):.4f}\n")
            f.write(f"  F1-Score: {result.get('f1_score_E', 0):.4f}\n\n")

            f.write("Pidgin Metrics:\n")
            f.write(f"  Precision: {result.get('precision_P', 0):.4f}\n")
            f.write(f"  Recall: {result.get('recall_P', 0):.4f}\n")
            f.write(f"  F1-Score: {result.get('f1_score_P', 0):.4f}\n\n")

            # Language distribution
            f.write("Language Distribution:\n")
            f.write(
                f"  Model: {result.get('model_english_pct', 0):.4f} English, {result.get('model_pidgin_pct', 0):.4f} Pidgin\n")
            f.write(
                f"  Ground truth: {result.get('gt_english_pct', 0):.4f} English, {result.get('gt_pidgin_pct', 0):.4f} Pidgin\n\n")

            f.write("\n")

    # Create comparative visualizations
    create_comparison_visualizations(sorted_results, os.path.join(output_dir, "visualizations"))

    logger.info(f"Summary report saved to {summary_txt} and {summary_csv}")
    return summary_txt, summary_csv


def create_comparison_visualizations(results, output_dir):
    """Create comparative visualizations across models."""
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(results) < 2:
        logger.warning("Need at least 2 models to create comparison visualizations")
        return

    # Extract data
    model_names = [r.get('model_name', f"Model {i}") for i, r in enumerate(results)]
    accuracies = [r.get('accuracy', 0) for r in results]
    f1_scores = [r.get('f1_score', 0) for r in results]
    english_f1 = [r.get('f1_score_E', 0) for r in results]
    pidgin_f1 = [r.get('f1_score_P', 0) for r in results]

    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies, color='blue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

    plt.savefig(os.path.join(output_dir, f"model_accuracy_comparison_{timestamp}.png"))
    plt.close()

    # F1-score comparison
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, f1_scores, color='green')
    plt.title('Model F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

    plt.savefig(os.path.join(output_dir, f"model_f1_comparison_{timestamp}.png"))
    plt.close()

    # Language-specific F1 comparison
    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, english_f1, width, label='English F1', color='blue')
    plt.bar(x + width / 2, pidgin_f1, width, label='Pidgin F1', color='green')

    plt.title('Language-Specific F1-Score Comparison')
    plt.xlabel('Model')
    plt.ylabel('F1-Score')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(english_f1):
        plt.text(i - width / 2, v + 0.02, f'{v:.4f}', ha='center')

    for i, v in enumerate(pidgin_f1):
        plt.text(i + width / 2, v + 0.02, f'{v:.4f}', ha='center')

    plt.savefig(os.path.join(output_dir, f"language_f1_comparison_{timestamp}.png"))
    plt.close()

    logger.info(f"Created comparison visualizations in {output_dir}")


if __name__ == "__main__":
    # When run directly, evaluate a test file
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth file")
    parser.add_argument("--model-output", required=True, help="Path to model output file or directory")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Directory to save evaluation results")

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_models(args.ground_truth, args.model_output, args.output_dir)

    if results:
        logger.info(f"Evaluation complete. Found {len(results)} valid model outputs.")
    else:
        logger.error("Evaluation failed or no valid model outputs found.")
