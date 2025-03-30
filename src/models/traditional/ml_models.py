#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of traditional machine learning models for
Nigerian Pidgin/English code-switching detection.

This implementation exactly follows the original training code
including window features, character n-grams, and specific parameters.
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logging_utils import setup_logger
from src.utils.file_io import ensure_dir, read_text_file, write_text_file

# Set up logger
logger = setup_logger('ml_models')

# Define language labels (exactly as in original code)
ENGLISH_LABEL = 'E'
PIDGIN_LABEL = 'P'


def load_config(config_path):
    """Load model configuration"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default settings.")
        return {
            "data": {
                "mixed_corpus": "data/corpus/mixed.txt",
                "english_corpus": "data/corpus/english.txt",
                "pidgin_corpus": "data/corpus/pidgin.txt",
                "test_data": "data/test/test_text.txt"
            },
            "ml": {
                "validation_split": 0.1,
                "sample_rate": 3,
                "window_size": 5,
                "fast_mode": False,
                "super_fast": False,
                "batch_size": 32,
                "epochs": 10,
                "patience": 3,
                "embedding_dim": 64,
                "lstm_units": 128
            }
        }

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_tagged_content(file_path, max_texts=None):
    """Extract tagged content from mixed dataset files."""
    logger.info(f"Extracting tagged content from {file_path}...")
    start_time = time.time()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Print data size
            logger.info(f"Read {len(content) / 1024:.1f} KB of data")

            # Process the matched content to preserve the original sequence
            segments = []

            # Simple regex to find all tags and their positions
            tag_pattern = re.compile(r'<(english|pidgin)>(.*?)</\1>', re.DOTALL)

            # Track progress
            matches = list(tag_pattern.finditer(content))
            total_matches = len(matches)
            logger.info(f"Found {total_matches} tagged segments")

            # Process matches
            for i, match in enumerate(matches):
                if i % 1000 == 0:
                    logger.info(f"Processing segment {i}/{total_matches}")

                tag_type = match.group(1)
                tag_content = match.group(2)
                segments.append((tag_content, 'english' if tag_type == 'english' else 'pidgin'))

                # Limit to max_texts if specified
                if max_texts is not None and len(segments) >= max_texts:
                    break

            elapsed = time.time() - start_time
            logger.info(f"Extracted {len(segments)} tagged segments in {elapsed:.1f}s")
            return segments
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def create_character_level_dataset(tagged_segments):
    """Create dataset with character-level labels with no length limits."""
    logger.info("Creating character-level dataset...")
    start_time = time.time()

    all_texts = []
    all_char_labels = []

    # Create texts by combining adjacent segments
    current_text = ""
    current_labels = []

    # Track progress
    total_segments = len(tagged_segments)
    total_chars = 0

    for i, (text, language) in enumerate(tagged_segments):
        if i % 1000 == 0:
            logger.info(f"Processing segment {i}/{total_segments}")

        # Add text
        current_text += text
        total_chars += len(text)

        # Add character-level labels
        for char in text:
            if language == 'english':
                current_labels.append(ENGLISH_LABEL)
            else:  # pidgin
                current_labels.append(PIDGIN_LABEL)

        # After processing a significant chunk, save and start new text
        # This is to avoid enormous texts while still preserving context
        if len(current_text) > 5000:  # Using a smaller threshold for faster processing
            all_texts.append(current_text)
            all_char_labels.append(current_labels)
            current_text = ""
            current_labels = []

    # Add the last text if there's any remaining
    if current_text:
        all_texts.append(current_text)
        all_char_labels.append(current_labels)

    elapsed = time.time() - start_time
    logger.info(f"Created {len(all_texts)} texts with {total_chars} total characters in {elapsed:.1f}s")
    return all_texts, all_char_labels


def prepare_window_features(texts, char_labels, window_size=5, sample_rate=3):
    """Create windowed features for character-level sequence labeling with increased sampling rate."""
    logger.info(f"Creating windowed features with window size {window_size} (sampling 1/{sample_rate} characters)...")
    start_time = time.time()

    all_X = []
    all_y = []
    all_text_ids = []
    all_char_positions = []

    # Get total character count for progress estimation
    total_chars = sum(len(text) for text in texts)
    progress_chars = 0  # Count processed characters
    progress_interval = total_chars // 20  # Update progress every 5%

    for text_id, (text, labels) in enumerate(zip(texts, char_labels)):
        # Print progress based on character count
        progress_chars += len(text)
        if progress_chars >= progress_interval:
            pct_complete = progress_chars / total_chars * 100
            elapsed = time.time() - start_time
            est_total = elapsed * (total_chars / progress_chars)
            logger.info(f"Window feature extraction: {pct_complete:.1f}% complete ({elapsed:.1f}s/{est_total:.1f}s)")
            progress_interval = progress_chars + (total_chars // 20)  # Next 5% mark

        for i in range(0, len(text), sample_rate):  # Process every sample_rate characters
            # Create window around current character
            start = max(0, i - window_size)
            end = min(len(text), i + window_size + 1)

            window = text[start:end]

            # Pad window if needed
            if len(window) < 2 * window_size + 1:
                if i < window_size:
                    window = ' ' * (window_size - i) + window
                else:
                    window = window + ' ' * (window_size - (len(text) - i - 1))

            all_X.append(window)
            all_y.append(1 if labels[i] == ENGLISH_LABEL else 0)  # 1 for English, 0 for Pidgin
            all_text_ids.append(text_id)
            all_char_positions.append(i)

    elapsed = time.time() - start_time
    logger.info(f"Created {len(all_X)} window samples in {elapsed:.1f}s")
    return all_X, all_y, all_text_ids, all_char_positions


def train_char_ngram_model(n, X_train, y_train, X_val, y_val, class_weight, results_dir, params):
    """Train a single character n-gram model set with optimizations for speed."""
    start_time = time.time()
    logger.info(f"Training character {n}-gram models...")

    # Create vectorizer with correct parameters
    min_df = 3 if params.get('super_fast', False) else 2  # Require features to appear more often

    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(n, n),
        min_df=min_df,
        max_df=0.95,
        max_features=10000 if params.get('super_fast', False) else None  # Limit features for super fast mode
    )

    # Fit vectorizer on training data
    logger.info(f"  Vectorizing data for n={n}...")
    vec_start = time.time()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    logger.info(f"  Vectorization complete for n={n} in {time.time() - vec_start:.1f}s")

    # Save vectorizer
    ngram_dir = os.path.join(results_dir, f"char_{n}_gram_sequence")
    ensure_dir(ngram_dir)

    vectorizer_path = os.path.join(ngram_dir, f"char_{n}_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save window size
    with open(os.path.join(ngram_dir, "window_size.txt"), "w") as f:
        f.write("5")  # Standard window size of 5

    # Results storage
    ngram_results = []

    # Select classifiers based on speed mode
    classifiers = {}
    if params.get('super_fast', False):
        # Only train the fastest classifier in super fast mode
        classifiers = {
            'logistic_regression': LogisticRegression(
                max_iter=200,  # Reduced iterations
                C=1.0,
                class_weight=class_weight,
                solver='saga',  # Faster solver
                n_jobs=1
            )
        }
    else:
        n_estimators = 20 if params.get('fast_mode', False) else 50 if not params.get('super_fast', False) else 10

        classifiers = {
            'linear_svc': LinearSVC(  # Using LinearSVC instead of SVC for speed
                C=1.0,
                class_weight=class_weight,
                max_iter=200 if params.get('fast_mode', False) else 1000,
                dual=False  # Faster for datasets with many samples
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                class_weight=class_weight,
                n_jobs=1,  # Single thread to avoid Windows issues
                max_depth=10 if params.get('fast_mode', False) else None  # Limit depth for faster training
            ),
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(
                max_iter=200 if params.get('fast_mode', False) else 1000,
                C=1.0,
                class_weight=class_weight,
                solver='saga' if params.get('fast_mode', False) else 'lbfgs',  # Faster solver in fast mode
                n_jobs=1
            )
        }

    # Train and evaluate each classifier
    for clf_name, classifier in classifiers.items():
        clf_start = time.time()
        logger.info(f"  Training {clf_name} with char_{n}...")

        # Train model
        classifier.fit(X_train_vec, y_train)

        # Save model
        model_path = os.path.join(ngram_dir, f"{clf_name}_char_{n}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)

        # Evaluate on validation set
        y_pred = classifier.predict(X_val_vec)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y_val, y_pred, pos_label=1, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        }

        clf_elapsed = time.time() - clf_start
        logger.info(
            f"  {clf_name} n={n} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f} in {clf_elapsed:.1f}s"
        )

        # Store results
        result = {
            'n_gram': n,
            'classifier': clf_name,
            **metrics
        }
        ngram_results.append(result)

    elapsed = time.time() - start_time
    logger.info(f"Completed character {n}-gram models in {elapsed:.1f}s")
    return ngram_results


def train_char_ngram_models(X_train, y_train, X_val, y_val, text_ids_train, text_ids_val, results_dir, params):
    """Train character n-gram models with optimizations for speed."""
    logger.info("=== Training Character N-gram Models ===")
    start_time = time.time()

    # Calculate class weights properly
    class_labels = np.unique(y_train)
    computed_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
    class_weight = {class_labels[i]: computed_weights[i] for i in range(len(class_labels))}

    logger.info(f"Class weights: {class_weight}")

    # Limit n-gram range in fast modes
    n_gram_range = range(1, 4) if params.get('super_fast', False) else range(1, 6) if params.get('fast_mode', False) else range(1, 9)
    logger.info(f"Training n-gram models for n = {list(n_gram_range)}")

    all_results = []
    for n in n_gram_range:
        # Train model for n-gram size n
        results = train_char_ngram_model(
            n, X_train, y_train, X_val, y_val, class_weight, results_dir, params
        )
        all_results.extend(results)

    # Save results
    ngram_results_df = pd.DataFrame(all_results)
    ngram_results_df.to_csv(os.path.join(results_dir, "char_ngram_sequence_validation.csv"), index=False)

    elapsed = time.time() - start_time
    logger.info(f"All n-gram models completed in {elapsed:.1f}s")
    return ngram_results_df


def train_word_level_models(X_train, y_train, X_val, y_val, text_ids_train, text_ids_val, results_dir, params):
    """Train word-level models with optimizations for speed."""
    logger.info("=== Training Word-Level Models ===")
    start_time = time.time()

    # Calculate class weights properly
    class_labels = np.unique(y_train)
    computed_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
    class_weight = {class_labels[i]: computed_weights[i] for i in range(len(class_labels))}

    logger.info(f"Class weights: {class_weight}")

    # Create directory for word-level models
    word_level_dir = os.path.join(results_dir, "word_level_sequence")
    ensure_dir(word_level_dir)

    # Create vectorizer with optimized parameters
    vectorizer = CountVectorizer(
        analyzer='word',
        token_pattern=r'\b\w+\b',  # Match whole words
        ngram_range=(1, 1) if params.get('super_fast', False) else (1, 2),  # Use unigrams only in super fast mode
        min_df=3 if params.get('super_fast', False) else 2,
        max_df=0.95,
        max_features=5000 if params.get('super_fast', False) else None  # Limit features for speed
    )

    # Fit vectorizer on training data
    logger.info("Vectorizing word-level data...")
    vec_start = time.time()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    logger.info(f"Word vectorization complete in {time.time() - vec_start:.1f}s")

    # Save vectorizer
    vectorizer_path = os.path.join(word_level_dir, "word_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save window size
    with open(os.path.join(word_level_dir, "window_size.txt"), "w") as f:
        f.write("10")  # Standard window size of 10 for word level

    # Select classifiers based on speed mode
    classifiers = {}
    if params.get('super_fast', False):
        # Only train the fastest classifier in super fast mode
        classifiers = {
            'logistic_regression': LogisticRegression(
                max_iter=200,  # Reduced iterations
                C=1.0,
                class_weight=class_weight,
                solver='saga',  # Faster solver
                n_jobs=1
            )
        }
    else:
        n_estimators = 20 if params.get('fast_mode', False) else 50 if not params.get('super_fast', False) else 10

        classifiers = {
            'linear_svc': LinearSVC(  # Using LinearSVC instead of SVC for speed
                C=1.0,
                class_weight=class_weight,
                max_iter=200 if params.get('fast_mode', False) else 1000,
                dual=False  # Faster for datasets with many samples
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                class_weight=class_weight,
                n_jobs=1,  # Single thread to avoid Windows issues
                max_depth=10 if params.get('fast_mode', False) else None  # Limit depth for faster training
            ),
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(
                max_iter=200 if params.get('fast_mode', False) else 1000,
                C=1.0,
                class_weight=class_weight,
                solver='saga' if params.get('fast_mode', False) else 'lbfgs',  # Faster solver in fast mode
                n_jobs=1
            )
        }

    # Results storage
    word_level_results = []

    # Train and evaluate each classifier
    for clf_name, classifier in classifiers.items():
        clf_start = time.time()
        logger.info(f"Training word-level {clf_name}...")

        # Train model
        classifier.fit(X_train_vec, y_train)

        # Save model
        model_path = os.path.join(word_level_dir, f"{clf_name}_word_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)

        # Evaluate on validation set
        y_pred = classifier.predict(X_val_vec)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y_val, y_pred, pos_label=1, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        }

        clf_elapsed = time.time() - clf_start
        logger.info(
            f"Word-level {clf_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f} in {clf_elapsed:.1f}s"
        )

        # Store results
        result = {
            'classifier': clf_name,
            **metrics
        }
        word_level_results.append(result)

    # Save results
    word_level_results_df = pd.DataFrame(word_level_results)
    word_level_results_df.to_csv(os.path.join(word_level_dir, "word_level_sequence_validation.csv"), index=False)

    elapsed = time.time() - start_time
    logger.info(f"Word-level models completed in {elapsed:.1f}s")
    return word_level_results_df


def train_ml_models(config_path, timestamp=None):
    """Train all traditional ML models for Nigerian Pidgin English code-switching detection."""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load configuration
    config = load_config(config_path)

    # Create model directory
    results_dir = os.path.join(config.get('paths', {}).get('models_dir', 'models'), 'traditional')
    ensure_dir(results_dir)

    # Extract parameters
    mixed_file = config['data']['mixed_corpus']
    sample_rate = config['ml'].get('sample_rate', 3)
    validation_split = config['ml'].get('validation_split', 0.1)

    # Extract tagged content from the mixed dataset
    tagged_segments = extract_tagged_content(mixed_file)
    logger.info(f"Working with {len(tagged_segments)} tagged segments")

    # Create character-level dataset
    texts, char_labels = create_character_level_dataset(tagged_segments)
    logger.info(f"Created dataset with {len(texts)} texts")

    # Prepare window features for traditional ML models
    window_size = config['ml'].get('window_size', 5)
    sample_rate = config['ml'].get('sample_rate', 3) * 2 if config['ml'].get('super_fast', False) else config['ml'].get('sample_rate', 3)
    logger.info(f"Using sample rate of 1/{sample_rate} for window features")

    # Character n-gram models (window size 5)
    X_windows_char, y_labels_char, text_ids_char, char_positions_char = prepare_window_features(
        texts, char_labels, window_size=window_size, sample_rate=sample_rate
    )

    # Word-level models (window size 10)
    X_windows_word, y_labels_word, text_ids_word, char_positions_word = prepare_window_features(
        texts, char_labels, window_size=10, sample_rate=sample_rate
    )

    # Split data for window-based models
    logger.info("Splitting data for training...")

    # Character n-gram data
    unique_text_ids_char = np.unique(text_ids_char)
    val_text_ids_char = np.random.choice(unique_text_ids_char,
                                        size=int(len(unique_text_ids_char) * validation_split),
                                        replace=False)
    val_indices_char = [i for i, tid in enumerate(text_ids_char) if tid in val_text_ids_char]
    train_indices_char = [i for i, tid in enumerate(text_ids_char) if tid not in val_text_ids_char]

    X_train_char = [X_windows_char[i] for i in train_indices_char]
    y_train_char = [y_labels_char[i] for i in train_indices_char]
    X_val_char = [X_windows_char[i] for i in val_indices_char]
    y_val_char = [y_labels_char[i] for i in val_indices_char]
    logger.info(f"Character n-gram data: {len(X_train_char)} training, {len(X_val_char)} validation windows")

    # Word-level data
    unique_text_ids_word = np.unique(text_ids_word)
    val_text_ids_word = np.random.choice(unique_text_ids_word,
                                        size=int(len(unique_text_ids_word) * validation_split),
                                        replace=False)
    val_indices_word = [i for i, tid in enumerate(text_ids_word) if tid in val_text_ids_word]
    train_indices_word = [i for i, tid in enumerate(text_ids_word) if tid not in val_text_ids_word]

    X_train_word = [X_windows_word[i] for i in train_indices_word]
    y_train_word = [y_labels_word[i] for i in train_indices_word]
    X_val_word = [X_windows_word[i] for i in val_indices_word]
    y_val_word = [y_labels_word[i] for i in val_indices_word]
    logger.info(f"Word-level data: {len(X_train_word)} training, {len(X_val_word)} validation windows")

    # Train models
    logger.info("\nStarting model training...")

    # Train character n-gram models
    char_ngram_results = train_char_ngram_models(
        X_train_char, y_train_char, X_val_char, y_val_char,
        [text_ids_char[i] for i in train_indices_char],
        [text_ids_char[i] for i in val_indices_char],
        results_dir,
        config['ml']
    )

    # Train word-level models
    word_level_results = train_word_level_models(
        X_train_word, y_train_word, X_val_word, y_val_word,
        [text_ids_word[i] for i in train_indices_word],
        [text_ids_word[i] for i in val_indices_word],
        results_dir,
        config['ml']
    )

    # Print summary of model performances
    logger.info("\n=== Model Performance Summary ===")

    logger.info("Character N-gram Models:")
    if isinstance(char_ngram_results, pd.DataFrame):
        for n in sorted(char_ngram_results['n_gram'].unique()):
            ngram_results = char_ngram_results[char_ngram_results['n_gram'] == n]
            if len(ngram_results) > 0:
                best_model = ngram_results.loc[ngram_results['f1_score'].idxmax()]
                logger.info(f"  n={n}: Best F1={best_model['f1_score']:.4f} ({best_model['classifier']})")

    logger.info("\nWord-level Models:")
    if isinstance(word_level_results, pd.DataFrame):
        best_word_model = word_level_results.loc[word_level_results['f1_score'].idxmax()]
        logger.info(f"  Best F1={best_word_model['f1_score']:.4f} ({best_word_model['classifier']})")

    logger.info(f"\nAll traditional ML models trained and saved to: {os.path.abspath(results_dir)}")
    return char_ngram_results, word_level_results


if __name__ == "__main__":
    # When run directly, use default config path
    train_ml_models("config/config.yaml")
