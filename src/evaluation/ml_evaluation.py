"""
Evaluation script for traditional machine learning and BiLSTM models
for Nigerian Pidgin/English code-switching detection.
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load evaluation configuration"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default settings.")
        return {
            "data": {
                "ground_truth": "data/corpus/ground_truth/ml_ground_truth.txt",
                "test_data": "data/test/ml_test.txt"
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "output_dir": "outputs/evaluation"
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


def evaluate_char_ngram_model(vectorizer, model, test_text, true_labels, model_name, n, output_dir, timestamp):
    """Evaluate character n-gram models"""
    logger.info(f"Evaluating {model_name} with {n}-gram features")

    # Prepare test data
    X_test = [char for char in test_text]

    # Transform using vectorizer
    X_test_features = vectorizer.transform(X_test)

    # Get predictions
    y_pred = model.predict(X_test_features)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, y_pred)
    precision = precision_score(true_labels, y_pred, pos_label='P', average='binary')
    recall = recall_score(true_labels, y_pred, pos_label='P', average='binary')
    f1 = f1_score(true_labels, y_pred, pos_label='P', average='binary')

    # Create confusion matrix
    cm = confusion_matrix(true_labels, y_pred, labels=['E', 'P'])

    # Save results
    results_dir = f"{output_dir}/character_ngram/{n}/{model_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'model': f"{model_name}_{n}gram",
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
    plt.title(f"Confusion Matrix - {model_name} ({n}-gram)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/confusion_matrix_{timestamp}.png")
    plt.close()

    # Generate marked-up output
    from src.utils.text_processing import tag_languages
    marked_up_text = tag_languages(test_text, y_pred)

    with open(f"{results_dir}/markup_output_{timestamp}.txt", 'w', encoding='utf-8') as f:
        f.write(marked_up_text)

    logger.info(f"Evaluation results for {model_name} ({n}-gram): Accuracy={accuracy:.4f}, F1={f1:.4f}")

    return metrics


def evaluate_all_ngram_models(test_text, true_labels, config, timestamp):
    """Evaluate all character n-gram models"""
    output_dir = config['evaluation']['output_dir']
    results = []

    n_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Could be loaded from config
    model_types = ['svm', 'random_forest', 'logistic_regression', 'naive_bayes']

    for n in n_values:
        model_dir = f"models/traditional/char_{n}_gram_sequence"

        # Skip if models don't exist for this n-gram
        if not os.path.exists(model_dir):
            logger.warning(f"No models found for {n}-gram sequence")
            continue

        # Load vectorizer
        try:
            with open(f"{model_dir}/char_{n}_vectorizer.pkl", 'rb') as f:
                vectorizer = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load vectorizer for {n}-gram: {str(e)}")
            continue

        # Evaluate each model type
        for model_type in model_types:
            try:
                # Load model
                with open(f"{model_dir}/{model_type}_{n}_model.pkl", 'rb') as f:
                    model = pickle.load(f)

                # Evaluate model
                metrics = evaluate_char_ngram_model(
                    vectorizer, model, test_text, true_labels,
                    model_type, n, output_dir, timestamp
                )

                results.append(metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_type} with {n}-gram: {str(e)}")

    # Compile and save overall results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{output_dir}/all_model_results_{timestamp}.csv", index=False)

        # Create comparison chart
        plt.figure(figsize=(12, 8))
        sns.barplot(x='model', y='f1_score', data=results_df)
        plt.title('Model F1-Score Comparison')
        plt.xlabel('Model')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison_{timestamp}.png")
        plt.close()

    return results


def evaluate_bilstm(config_path, timestamp=None):
    """Evaluate BiLSTM model"""
    logger.info("BiLSTM evaluation not implemented yet")
    # This would be implemented based on your BiLSTM model structure


def evaluate_ml_models(config_path, timestamp=None):
    """Evaluate all ML models"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    config = load_config(config_path)

    # Load test data
    test_text, ground_truth_text, true_labels = load_test_data(config)

    # Evaluate n-gram models
    results = evaluate_all_ngram_models(test_text, true_labels, config, timestamp)

    logger.info(f"All ML models evaluated. Timestamp: {timestamp}")

    return results


if __name__ == "__main__":
    # Configure logging for stand-alone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # When run directly, use default config path
    evaluate_ml_models("config/evaluation_config.yaml")