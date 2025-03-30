#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for Nigerian Pidgin/English code-switching detection project.
Creates visualizations of evaluation results and corpus statistics.
"""

import os
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
from src.utils.file_io import ensure_dir

# Set up logger
logger = setup_logger('visualization')

# Set visualization style
plt.style.use('ggplot')


def plot_metrics(result, model_name, output_dir):
    """
    Create bar chart of performance metrics for a model.

    Args:
        result (dict): Model evaluation results
        model_name (str): Name of the model
        output_dir (str): Output directory

    Returns:
        str: Path to saved plot
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create bar chart for metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        result.get('accuracy', 0),
        result.get('precision', 0),
        result.get('recall', 0),
        result.get('f1_score', 0)
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1.0)
    plt.title(f'Evaluation Metrics for {model_name}')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center')

    # Save plot
    metrics_chart = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.png")
    plt.savefig(metrics_chart)
    plt.close()

    logger.info(f"Created metrics chart: {metrics_chart}")
    return metrics_chart


def plot_language_comparison(result, model_name, output_dir):
    """
    Create language-specific metrics comparison chart.

    Args:
        result (dict): Model evaluation results
        model_name (str): Name of the model
        output_dir (str): Output directory

    Returns:
        str: Path to saved plot
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract language-specific metrics
    labels = ['Precision', 'Recall', 'F1-Score']
    english_values = [
        result.get('precision_E', 0),
        result.get('recall_E', 0),
        result.get('f1_score_E', 0)
    ]
    pidgin_values = [
        result.get('precision_P', 0),
        result.get('recall_P', 0),
        result.get('f1_score_P', 0)
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, english_values, width, label='English', color='blue')
    plt.bar(x + width/2, pidgin_values, width, label='Pidgin', color='green')

    plt.ylabel('Score')
    plt.title(f'English vs. Pidgin Metrics for {model_name}')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for i, v in enumerate(english_values):
        plt.text(i - width/2, v + 0.02, f'{v:.4f}', ha='center')

    for i, v in enumerate(pidgin_values):
        plt.text(i + width/2, v + 0.02, f'{v:.4f}', ha='center')

    # Save plot
    comp_chart = os.path.join(output_dir, f"{model_name}_language_comparison_{timestamp}.png")
    plt.savefig(comp_chart)
    plt.close()

    logger.info(f"Created language comparison chart: {comp_chart}")
    return comp_chart


def plot_confusion_matrix(result, model_name, output_dir):
    """
    Create confusion matrix visualization.

    Args:
        result (dict): Model evaluation results
        model_name (str): Name of the model
        output_dir (str): Output directory

    Returns:
        str: Path to saved plot
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    confmat = np.array([
        [result.get('TP_E', 0), result.get('FP_E', 0)],
        [result.get('FN_E', 0), result.get('TP_P', 0)]
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

    # Add text annotations
    thresh = confmat.max() / 2.
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            plt.text(j, i, format(confmat[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confmat[i, j] > thresh else "black")

    # Save plot
    cm_chart = os.path.join(output_dir, f"{model_name}_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_chart)
    plt.close()

    logger.info(f"Created confusion matrix: {cm_chart}")
    return cm_chart


def plot_language_distribution(result, model_name, output_dir):
    """
    Create pie charts showing language distribution.

    Args:
        result (dict): Model evaluation results
        model_name (str): Name of the model
        output_dir (str): Output directory

    Returns:
        str: Path to saved plot
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create pie charts
    plt.figure(figsize=(12, 5))

    # Ground Truth distribution
    plt.subplot(1, 2, 1)
    gt_dist = [result.get('gt_english_pct', 0.5), result.get('gt_pidgin_pct', 0.5)]
    plt.pie(gt_dist, labels=['English', 'Pidgin'], autopct='%1.1f%%', colors=['blue', 'green'])
    plt.title('Ground Truth Language Distribution')

    # Model distribution
    plt.subplot(1, 2, 2)
    model_dist = [result.get('model_english_pct', 0.5), result.get('model_pidgin_pct', 0.5)]
    plt.pie(model_dist, labels=['English', 'Pidgin'], autopct='%1.1f%%', colors=['blue', 'green'])
    plt.title(f'{model_name} Language Distribution')

    # Save plot
    dist_chart = os.path.join(output_dir, f"{model_name}_language_dist_{timestamp}.png")
    plt.savefig(dist_chart)
    plt.close()

    logger.info(f"Created language distribution chart: {dist_chart}")
    return dist_chart


def create_model_visualizations(result, model_name, output_dir):
    """
    Create all visualizations for a single model evaluation.

    Args:
        result (dict): Model evaluation results
        model_name (str): Name of the model
        output_dir (str): Output directory

    Returns:
        dict: Paths to all created visualizations
    """
    model_output_dir = os.path.join(output_dir, model_name)
    ensure_dir(model_output_dir)

    logger.info(f"Creating visualizations for model: {model_name}")

    # Create all plots
    metrics_chart = plot_metrics(result, model_name, model_output_dir)
    comp_chart = plot_language_comparison(result, model_name, model_output_dir)
    cm_chart = plot_confusion_matrix(result, model_name, model_output_dir)
    dist_chart = plot_language_distribution(result, model_name, model_output_dir)

    # Return all chart paths
    return {
        'metrics_chart': metrics_chart,
        'language_comparison_chart': comp_chart,
        'confusion_matrix': cm_chart,
        'language_distribution_chart': dist_chart
    }


def plot_model_comparison(results, output_dir):
    """
    Create comparative visualizations across multiple models.

    Args:
        results (list): List of model evaluation results
        output_dir (str): Output directory

    Returns:
        dict: Paths to created visualizations
    """
    if len(results) < 2:
        logger.warning("Need at least 2 models for comparison")
        return {}

    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sort results by accuracy
    sorted_results = sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True)

    # Extract data for plotting
    model_names = [r.get('model_name', f'Model {i}') for i, r in enumerate(sorted_results)]
    accuracies = [r.get('accuracy', 0) for r in sorted_results]
    f1_scores = [r.get('f1_score', 0) for r in sorted_results]
    english_f1 = [r.get('f1_score_E', 0) for r in sorted_results]
    pidgin_f1 = [r.get('f1_score_P', 0) for r in sorted_results]

    charts = {}

    # 1. Accuracy comparison chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color='blue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center')

    acc_chart = os.path.join(output_dir, f"model_accuracy_comparison_{timestamp}.png")
    plt.savefig(acc_chart)
    plt.close()
    charts['accuracy_comparison'] = acc_chart

    # 2. F1-score comparison chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, f1_scores, color='green')
    plt.title('Model F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center')

    f1_chart = os.path.join(output_dir, f"model_f1_comparison_{timestamp}.png")
    plt.savefig(f1_chart)
    plt.close()
    charts['f1_comparison'] = f1_chart

    # 3. Language-specific F1 comparison
    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, english_f1, width, label='English F1', color='blue')
    plt.bar(x + width/2, pidgin_f1, width, label='Pidgin F1', color='green')

    plt.title('Language-Specific F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for i, v in enumerate(english_f1):
        plt.text(i - width/2, v + 0.02, f'{v:.4f}', ha='center')

    for i, v in enumerate(pidgin_f1):
        plt.text(i + width/2, v + 0.02, f'{v:.4f}', ha='center')

    lang_f1_chart = os.path.join(output_dir, f"language_f1_comparison_{timestamp}.png")
    plt.savefig(lang_f1_chart)
    plt.close()
    charts['language_f1_comparison'] = lang_f1_chart

    logger.info(f"Created model comparison visualizations in {output_dir}")
    return charts


if __name__ == "__main__":
    # When run directly, test with sample data
    sample_result = {
        'model_name': 'test_model',
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.82,
        'f1_score': 0.825,
        'precision_E': 0.88,
        'recall_E': 0.85,
        'f1_score_E': 0.865,
        'precision_P': 0.78,
        'recall_P': 0.79,
        'f1_score_P': 0.785,
        'TP_E': 850,
        'FP_E': 120,
        'FN_E': 150,
        'TP_P': 750,
        'gt_english_pct': 0.55,
        'gt_pidgin_pct': 0.45,
        'model_english_pct': 0.52,
        'model_pidgin_pct': 0.48
    }

    # Test single model visualization
    create_model_visualizations(sample_result, 'test_model', 'test_viz')

    # Test model comparison
    test_results = [
        sample_result,
        {**sample_result, 'model_name': 'model2', 'accuracy': 0.78, 'f1_score': 0.76},
        {**sample_result, 'model_name': 'model3', 'accuracy': 0.92, 'f1_score': 0.91}
    ]

    plot_model_comparison(test_results, 'test_viz/comparison')