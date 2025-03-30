#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BiLSTM model implementation for Nigerian Pidgin/English code-switching detection.
"""

import os
import re
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, TimeDistributed
import yaml
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logging_utils import setup_logger
from src.utils.file_io import ensure_dir, read_text_file  # Updated import

# Set up logger
logger = setup_logger('bilstm_model')

# Define language labels (exactly as in original code)
ENGLISH_LABEL = 'E'
PIDGIN_LABEL = 'P'

# Disable TensorFlow warnings if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_config(config_path):
    """Load model configuration"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default settings.")
        return {
            "data": {
                "mixed_corpus": "data/corpus/mixed.txt",
                "english_corpus": "data/corpus/english.txt",
                "pidgin_corpus": "data/corpus/pidgin.txt"
            },
            "ml": {
                "batch_size": 32,
                "epochs": 10,
                "embedding_dim": 64,
                "lstm_units": 128,
                "validation_split": 0.1,
                "patience": 3,
                "fast_mode": False,
                "super_fast": False
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
        if len(current_text) > 5000:  # Smaller threshold for faster processing
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


def efficient_batch_generator(texts, char_labels, batch_size, char_to_index):
    """Generate batches for training by grouping similar lengths."""
    text_lengths = [len(text) for text in texts]
    indices = np.argsort(text_lengths)

    batches = []

    # Track progress
    logger.info(f"Creating {len(indices) // batch_size + 1} batches for BiLSTM training...")
    batch_count = 0

    # Group by similar lengths
    for i in range(0, len(indices), batch_size):
        if batch_count % 10 == 0:
            logger.info(f"  Creating batch {batch_count + 1}/{len(indices) // batch_size + 1}")

        batch_indices = indices[i:i + batch_size]

        # Get texts and labels for this batch
        batch_texts = [texts[j] for j in batch_indices]
        batch_labels = [char_labels[j] for j in batch_indices]

        # Find max length in this batch
        max_len = max(len(text) for text in batch_texts)

        # Create arrays
        X_batch = np.zeros((len(batch_indices), max_len), dtype=np.int32)
        y_batch = np.zeros((len(batch_indices), max_len, 2), dtype=np.int32)

        # Fill arrays
        for j, (text, labels) in enumerate(zip(batch_texts, batch_labels)):
            for k, char in enumerate(text):
                X_batch[j, k] = char_to_index.get(char, 0)

            for k, label in enumerate(labels):
                if label == ENGLISH_LABEL:
                    y_batch[j, k, 1] = 1  # English
                else:
                    y_batch[j, k, 0] = 1  # Pidgin

        batches.append((X_batch, y_batch))
        batch_count += 1

    # Shuffle batches
    np.random.shuffle(batches)
    logger.info(f"Created {len(batches)} training batches")
    return batches


def train_character_level_bilstm(texts, char_labels, results_dir, params):
    """Train a character-level BiLSTM model with optimizations for speed."""
    logger.info("=== Training Character-Level BiLSTM Model ===")
    start_time = time.time()

    # Create directory for BiLSTM model
    bilstm_dir = os.path.join(results_dir, "bilstm_char_level")
    ensure_dir(bilstm_dir)

    # Prepare character mappings
    logger.info("Building character mapping...")
    chars = set()
    for text in texts:
        chars.update(text)

    char_to_index = {c: i + 1 for i, c in enumerate(sorted(chars))}  # Reserve 0 for padding
    index_to_char = {i + 1: c for i, c in enumerate(sorted(chars))}
    index_to_char[0] = ''  # Padding character

    # Save character mappings
    char_mappings = {
        'char_to_index': char_to_index,
        'index_to_char': index_to_char
    }
    mappings_path = os.path.join(bilstm_dir, "char_mappings.pkl")
    with open(mappings_path, 'wb') as f:
        pickle.dump(char_mappings, f)

    # Split data into training and validation sets
    logger.info("Splitting data for BiLSTM training...")
    validation_split = params.get('validation_split', 0.1)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, char_labels, test_size=validation_split, random_state=42
    )
    logger.info(f"Training on {len(train_texts)} texts, validating on {len(val_texts)} texts")

    # Calculate class weights
    logger.info("Calculating class weights...")
    all_train_labels = []
    for labels in train_labels:
        all_train_labels.extend(labels)

    class_counts = {ENGLISH_LABEL: 0, PIDGIN_LABEL: 0}
    for label in all_train_labels:
        class_counts[label] += 1

    total_chars = len(all_train_labels)
    class_weight = {
        0: total_chars / (2 * class_counts[PIDGIN_LABEL]),  # Pidgin index
        1: total_chars / (2 * class_counts[ENGLISH_LABEL])    # English index
    }
    logger.info(f"Character class distribution: {class_counts}")
    logger.info(f"Class weights: {class_weight}")

    # Adjust architecture based on speed settings
    embedding_dim = params.get('embedding_dim', 64) // 2 if params.get('super_fast', False) else params.get('embedding_dim', 64)
    lstm_units = params.get('lstm_units', 128) // 2 if params.get('super_fast', False) else params.get('lstm_units', 128)

    # Build BiLSTM model for sequence labeling
    logger.info("Building BiLSTM model...")
    model = Sequential()
    model.add(Embedding(len(char_to_index) + 1, embedding_dim))

    if params.get('super_fast', False):
        model.add(Bidirectional(LSTM(lstm_units // 2, return_sequences=True)))
    else:
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(2, activation='softmax')))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        sample_weight_mode="temporal"
    )

    model.summary()

    logger.info("Generating training batches...")
    batch_size = params.get('batch_size', 32)
    batches = efficient_batch_generator(train_texts, train_labels, batch_size, char_to_index)

    logger.info("Preparing validation data...")
    max_val_len = max(len(t) for t in val_texts)
    X_val = np.zeros((len(val_texts), max_val_len), dtype=np.int32)
    y_val = np.zeros((len(val_texts), max_val_len, 2), dtype=np.int32)

    for i, (text, labels) in enumerate(zip(val_texts, val_labels)):
        for j, char in enumerate(text):
            X_val[i, j] = char_to_index.get(char, 0)
        for j, label in enumerate(labels):
            if label == ENGLISH_LABEL:
                y_val[i, j, 1] = 1
            else:
                y_val[i, j, 0] = 1

    val_sample_weights = np.ones(X_val.shape)
    val_sample_weights[X_val == 0] = 0

    epochs = 3 if params.get('super_fast', False) else 5 if params.get('fast_mode', False) else params.get('epochs', 10)
    logger.info(f"Training BiLSTM model for {epochs} epochs...")
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    best_val_accuracy = 0
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        np.random.shuffle(batches)

        epoch_loss = []
        epoch_acc = []
        batch_count = len(batches)
        batch_interval = max(1, batch_count // 10)

        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            sample_weights = np.ones(X_batch.shape)
            sample_weights[X_batch == 0] = 0

            batch_hist = model.train_on_batch(
                X_batch, y_batch,
                sample_weight=sample_weights,
                reset_metrics=False
            )
            epoch_loss.append(batch_hist[0])
            epoch_acc.append(batch_hist[1])

            if (batch_idx + 1) % batch_interval == 0 or batch_idx == 0 or batch_idx == len(batches) - 1:
                logger.info(f"  Batch {batch_idx + 1}/{len(batches)} - loss: {batch_hist[0]:.4f} - accuracy: {batch_hist[1]:.4f}")
                elapsed_so_far = time.time() - epoch_start
                batches_left = len(batches) - (batch_idx + 1)
                est_time_left = elapsed_so_far / (batch_idx + 1) * batches_left
                logger.info(f"  Estimated time remaining for epoch: {est_time_left:.1f}s")

        logger.info("Evaluating on validation data...")
        val_hist = model.evaluate(
            X_val, y_val,
            verbose=0,
            sample_weight=val_sample_weights
        )

        avg_loss = np.mean(epoch_loss)
        avg_acc = np.mean(epoch_acc)
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        history['val_loss'].append(val_hist[0])
        history['val_accuracy'].append(val_hist[1])

        epoch_elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1}/{epochs} complete - loss: {avg_loss:.4f} - accuracy: {avg_acc:.4f} - val_loss: {val_hist[0]:.4f} - val_accuracy: {val_hist[1]:.4f} - {epoch_elapsed:.1f}s")

        if val_hist[1] > best_val_accuracy:
            best_val_accuracy = val_hist[1]
            best_weights = model.get_weights()
            epochs_no_improve = 0
            logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epochs")

        patience = params.get('patience', 3)
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping: no improvement for {patience} epochs")
            break

        elapsed_so_far = time.time() - start_time
        epochs_completed = epoch + 1
        epochs_left = epochs - epochs_completed
        est_time_left = elapsed_so_far / epochs_completed * epochs_left
        logger.info(f"Estimated time for remaining {epochs_left} epochs: {est_time_left:.1f}s")

    if best_weights is not None:
        logger.info("Restoring best model weights...")
        model.set_weights(best_weights)

    logger.info("Saving final model...")
    model.save(os.path.join(bilstm_dir, "bilstm_char_level_model.h5"))

    logger.info("Evaluating final model on validation set...")
    val_pred = model.predict(X_val, batch_size=batch_size, verbose=1)
    val_pred_classes = np.argmax(val_pred, axis=-1)
    y_val_classes = np.argmax(y_val, axis=-1)

    mask = X_val > 0
    correct_chars = np.sum((val_pred_classes == y_val_classes) & mask)
    total_chars = np.sum(mask)
    char_accuracy = correct_chars / total_chars

    logger.info(f"Character-level BiLSTM - Final character accuracy: {char_accuracy:.4f}")

    logger.info("Generating training history plots...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(bilstm_dir, "training_history.png"))

    bilstm_result = {
        'model': 'character_level_bilstm',
        'accuracy': char_accuracy,
        'epochs_trained': len(history['accuracy']),
        'best_val_accuracy': best_val_accuracy,
        'best_epoch': 1  # Placeholder since epochs_no_improve is not an array
    }

    bilstm_df = pd.DataFrame([bilstm_result])
    bilstm_df.to_csv(os.path.join(bilstm_dir, "bilstm_char_level_validation.csv"), index=False)

    elapsed = time.time() - start_time
    logger.info(f"BiLSTM training completed in {elapsed:.1f}s")
    return bilstm_result


def train_bilstm(config_path, timestamp=None):
    """Train BiLSTM model for Nigerian Pidgin English code-switching detection."""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    config = load_config(config_path)

    results_dir = os.path.join(config.get('paths', {}).get('models_dir', 'models'), 'advanced')
    ensure_dir(results_dir)

    mixed_file = config['data']['mixed_corpus']

    tagged_segments = extract_tagged_content(mixed_file)
    logger.info(f"Working with {len(tagged_segments)} tagged segments")

    texts, char_labels = create_character_level_dataset(tagged_segments)
    logger.info(f"Created dataset with {len(texts)} texts")

    bilstm_result = train_character_level_bilstm(
        texts, char_labels, results_dir, config['ml']
    )

    logger.info("\n=== BiLSTM Model Performance Summary ===")
    logger.info(f"Accuracy: {bilstm_result['accuracy']:.4f}")
    logger.info(f"Best validation accuracy: {bilstm_result['best_val_accuracy']:.4f}")
    logger.info(f"Epochs trained: {bilstm_result['epochs_trained']}")

    logger.info(f"\nBiLSTM model trained and saved to: {os.path.abspath(results_dir)}")
    return bilstm_result


if __name__ == "__main__":
    train_bilstm("config/config.yaml")
