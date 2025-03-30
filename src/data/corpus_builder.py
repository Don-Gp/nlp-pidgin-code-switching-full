#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corpus builder module for Nigerian Pidgin English code-switching detection.
Creates and manipulates corpus files for training and testing.
"""

import random
import re
import numpy as np
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.logging_utils import setup_logger
from src.utils.file_io import ensure_dir, read_file, write_file

# Set up logger
logger = setup_logger('corpus_builder')


def extract_sentences(file_path):
    """
    Extract Pidgin and English sentences from a mixed dataset.

    Args:
        file_path (str): Path to the mixed dataset file

    Returns:
        tuple: Lists of Pidgin and English sentences
    """
    try:
        text = read_file(file_path)

        pidgin_sentences = re.findall(r'<pidgin>(.*?)</pidgin>', text, re.DOTALL)
        english_sentences = re.findall(r'<[Ee]nglish>(.*?)</[Ee]nglish>', text, re.DOTALL)

        logger.info(
            f"Extracted {len(pidgin_sentences)} Pidgin sentences and {len(english_sentences)} English sentences")
        return pidgin_sentences, english_sentences

    except Exception as e:
        logger.error(f"Error extracting sentences from {file_path}: {e}")
        return [], []


def save_to_file(sentences, file_path):
    """
    Save sentences to a file.

    Args:
        sentences (list): List of sentences
        file_path (str): Output file path

    Returns:
        bool: Success status
    """
    try:
        # Ensure output directory exists
        ensure_dir(os.path.dirname(file_path))

        # Write sentences to file
        content = "\n".join([sentence.strip() for sentence in sentences])
        write_file(file_path, content)

        logger.info(f"Saved {len(sentences)} sentences to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving sentences to {file_path}: {e}")
        return False


def stratified_sampling(sentences, sample_size):
    """
    Perform stratified sampling based on sentence length.

    Args:
        sentences (list): List of sentences
        sample_size (int): Desired sample size

    Returns:
        list: Sampled sentences
    """
    if not sentences:
        logger.warning("No sentences provided for sampling")
        return []

    # Divide sentences into short, medium, and long
    short = [s for s in sentences if len(s.split()) <= 5]
    medium = [s for s in sentences if 6 <= len(s.split()) <= 15]
    long = [s for s in sentences if len(s.split()) > 15]

    logger.info(f"Sentence distribution - Short: {len(short)}, Medium: {len(medium)}, Long: {len(long)}")

    # Ensure at least some samples are taken from each category
    short_sample = random.sample(short, min(len(short), sample_size // 3))
    medium_sample = random.sample(medium, min(len(medium), sample_size // 3))
    long_sample = random.sample(long, min(len(long), sample_size - len(short_sample) - len(medium_sample)))

    # Combine samples
    sampled = short_sample + medium_sample + long_sample
    logger.info(f"Sampled {len(sampled)} sentences using stratified sampling")

    return sampled


def create_mixed_code_switching_ground_truth(pidgin_sentences, english_sentences, ground_truth_size=600):
    """
    Create a ground truth dataset with multiple code-switching patterns.

    Args:
        pidgin_sentences (list): List of Pidgin sentences
        english_sentences (list): List of English sentences
        ground_truth_size (int): Desired ground truth size

    Returns:
        list: Ground truth dataset with code-switching
    """
    logger.info(f"Creating mixed code-switching ground truth (size: {ground_truth_size})")

    # Determine allocation of different types
    intra_count = ground_truth_size // 3  # 1/3 intra-sentential
    adjacency_count = ground_truth_size - intra_count  # 2/3 adjacency pairs

    logger.info(f"Planned distribution - Intra-sentential: {intra_count}, Adjacency pairs: {adjacency_count}")

    # Sort sentences by length
    pidgin_short = [s for s in pidgin_sentences if len(s.split()) <= 8]
    pidgin_long = [s for s in pidgin_sentences if len(s.split()) > 8]
    english_short = [s for s in english_sentences if len(s.split()) <= 8]
    english_long = [s for s in english_sentences if len(s.split()) > 8]

    # Create ground truth container
    ground_truth = []

    # 1. Create intra-sentential examples (combining shorter sentences)
    pidgin_fragments = random.sample(pidgin_short, min(len(pidgin_short), intra_count))
    english_fragments = random.sample(english_short, min(len(english_short), intra_count))

    # Make sure we have equal numbers
    min_fragments = min(len(pidgin_fragments), len(english_fragments))
    pidgin_fragments = pidgin_fragments[:min_fragments]
    english_fragments = english_fragments[:min_fragments]

    logger.info(f"Creating {min_fragments} intra-sentential examples")

    for i in range(min_fragments):
        if i % 2 == 0:
            # Pidgin first, then English
            ground_truth.append(
                f'<pidgin>{pidgin_fragments[i]}</pidgin> <english>{english_fragments[i]}</english>'
            )
        else:
            # English first, then Pidgin
            ground_truth.append(
                f'<english>{english_fragments[i]}</english> <pidgin>{pidgin_fragments[i]}</pidgin>'
            )

    # 2. Create adjacency pairs (using longer sentences when available)
    # Determine how many pairs we need (each pair has two sentences)
    pairs_needed = adjacency_count // 2
    logger.info(f"Creating {pairs_needed} adjacency pairs")

    # Preferentially use longer sentences for adjacency pairs
    pidgin_for_pairs = random.sample(
        pidgin_long if len(pidgin_long) >= pairs_needed else pidgin_sentences,
        min(len(pidgin_long) if len(pidgin_long) >= pairs_needed else len(pidgin_sentences), pairs_needed)
    )
    english_for_pairs = random.sample(
        english_long if len(english_long) >= pairs_needed else english_sentences,
        min(len(english_long) if len(english_long) >= pairs_needed else len(english_sentences), pairs_needed)
    )

    # Make sure we have equal numbers
    min_pairs = min(len(pidgin_for_pairs), len(english_for_pairs))
    pidgin_for_pairs = pidgin_for_pairs[:min_pairs]
    english_for_pairs = english_for_pairs[:min_pairs]

    # Create the pairs
    for i in range(min_pairs):
        # Randomly determine order
        if random.choice([True, False]):
            ground_truth.append(f'<pidgin>{pidgin_for_pairs[i]}</pidgin>')
            ground_truth.append(f'<english>{english_for_pairs[i]}</english>')
        else:
            ground_truth.append(f'<english>{english_for_pairs[i]}</english>')
            ground_truth.append(f'<pidgin>{pidgin_for_pairs[i]}</pidgin>')

    # Shuffle to mix the different patterns
    random.shuffle(ground_truth)
    logger.info(f"Created {len(ground_truth)} ground truth entries")

    return ground_truth


def split_dataset(mixed_file, pidgin_file, english_file, ground_truth_file, ground_truth_size=600):
    """
    Split the mixed dataset, creating separate files for training and ground truth.

    Args:
        mixed_file (str): Path to mixed dataset file
        pidgin_file (str): Path to output Pidgin corpus file
        english_file (str): Path to output English corpus file
        ground_truth_file (str): Path to output ground truth file
        ground_truth_size (int): Desired ground truth size

    Returns:
        bool: Success status
    """
    logger.info(f"Splitting dataset - Mixed: {mixed_file}")

    # Extract sentences from mixed file
    pidgin_sentences, english_sentences = extract_sentences(mixed_file)

    if not pidgin_sentences or not english_sentences:
        logger.error("Failed to extract sentences from mixed file")
        return False

    # Create mixed code-switching ground truth
    ground_truth_samples = create_mixed_code_switching_ground_truth(
        pidgin_sentences, english_sentences, ground_truth_size
    )

    # Extract the raw sentences from the ground truth for removal from training set
    ground_truth_pidgin_raw = []
    ground_truth_english_raw = []

    for sample in ground_truth_samples:
        # Extract all pidgin segments
        pidgin_matches = re.findall(r'<pidgin>(.*?)</pidgin>', sample)
        ground_truth_pidgin_raw.extend(pidgin_matches)

        # Extract all english segments
        english_matches = re.findall(r'<english>(.*?)</english>', sample)
        ground_truth_english_raw.extend(english_matches)

    # Remove ground truth sentences from training data
    pidgin_final = [s for s in pidgin_sentences if s not in ground_truth_pidgin_raw]
    english_final = [s for s in english_sentences if s not in ground_truth_english_raw]

    logger.info(f"After removing ground truth: {len(pidgin_final)} Pidgin, {len(english_final)} English sentences")

    # Save to files
    save_to_file(pidgin_final, pidgin_file)
    save_to_file(english_final, english_file)
    save_to_file(ground_truth_samples, ground_truth_file)

    logger.info("Dataset split completed successfully!")
    logger.info(f"- Pidgin Sentences: {len(pidgin_final)} saved in {pidgin_file}")
    logger.info(f"- English Sentences: {len(english_final)} saved in {english_file}")
    logger.info(f"- Ground Truth Samples: {len(ground_truth_samples)} saved in {ground_truth_file}")
    logger.info(f"  - Intra-sentential samples: ~{ground_truth_size // 3}")
    logger.info(f"  - Adjacency pair samples: ~{ground_truth_size - (ground_truth_size // 3)}")
    logger.info(f"  - Pidgin segments in Ground Truth: {len(ground_truth_pidgin_raw)}")
    logger.info(f"  - English segments in Ground Truth: {len(ground_truth_english_raw)}")

    return True


def create_corpus_statistics(english_file, pidgin_file, mixed_file, output_dir):
    """
    Create statistics about the corpus files.

    Args:
        english_file (str): Path to English corpus
        pidgin_file (str): Path to Pidgin corpus
        mixed_file (str): Path to mixed corpus
        output_dir (str): Directory to save statistics

    Returns:
        dict: Corpus statistics
    """
    logger.info("Generating corpus statistics")

    stats = {}

    # Process English corpus
    try:
        english_text = read_file(english_file)
        english_words = re.findall(r'\b\w+\b', english_text.lower())
        english_sentences = re.split(r'[.!?]', english_text)

        stats['english'] = {
            'total_sentences': len(english_sentences),
            'total_words': len(english_words),
            'total_chars': len(english_text),
            'unique_words': len(set(english_words)),
            'avg_word_length': sum(len(word) for word in english_words) / len(english_words) if english_words else 0
        }

        logger.info(
            f"English corpus: {stats['english']['total_words']} words, {stats['english']['unique_words']} unique")
    except Exception as e:
        logger.error(f"Error processing English corpus: {e}")

    # Process Pidgin corpus
    try:
        pidgin_text = read_file(pidgin_file)
        pidgin_words = re.findall(r'\b\w+\b', pidgin_text.lower())
        pidgin_sentences = re.split(r'[.!?]', pidgin_text)

        stats['pidgin'] = {
            'total_sentences': len(pidgin_sentences),
            'total_words': len(pidgin_words),
            'total_chars': len(pidgin_text),
            'unique_words': len(set(pidgin_words)),
            'avg_word_length': sum(len(word) for word in pidgin_words) / len(pidgin_words) if pidgin_words else 0
        }

        logger.info(f"Pidgin corpus: {stats['pidgin']['total_words']} words, {stats['pidgin']['unique_words']} unique")
    except Exception as e:
        logger.error(f"Error processing Pidgin corpus: {e}")

    # Process Mixed corpus
    try:
        mixed_text = read_file(mixed_file)

        # Count pidgin and english tokens in mixed corpus
        pidgin_tokens = len(re.findall(r'<pidgin>(.*?)</pidgin>', mixed_text, re.DOTALL))
        english_tokens = len(re.findall(r'<english>(.*?)</english>', mixed_text, re.DOTALL))
        total_tokens = pidgin_tokens + english_tokens

        stats['mixed'] = {
            'total_tokens': total_tokens,
            'pidgin_tokens': pidgin_tokens,
            'english_tokens': english_tokens,
            'pidgin_percentage': (pidgin_tokens / total_tokens) * 100 if total_tokens > 0 else 0,
            'english_percentage': (english_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        }

        logger.info(f"Mixed corpus: {total_tokens} total segments, {pidgin_tokens} Pidgin, {english_tokens} English")
    except Exception as e:
        logger.error(f"Error processing Mixed corpus: {e}")

    # Save statistics to file
    try:
        ensure_dir(output_dir)
        stats_file = os.path.join(output_dir, "corpus_statistics.json")

        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Corpus statistics saved to {stats_file}")
    except Exception as e:
        logger.error(f"Error saving corpus statistics: {e}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and manage corpus for Nigerian Pidgin English code-switching")

    # Command selection
    parser.add_argument("--command", choices=["split", "stats"], default="split",
                        help="Command to execute")

    # Input/output paths
    parser.add_argument("--mixed", default="data/corpus/mixed.txt",
                        help="Path to mixed dataset file")
    parser.add_argument("--pidgin", default="data/corpus/pidgin.txt",
                        help="Path to output Pidgin corpus file")
    parser.add_argument("--english", default="data/corpus/english.txt",
                        help="Path to output English corpus file")
    parser.add_argument("--ground-truth", default="data/corpus/ml_ground_truth.txt",
                        help="Path to output ground truth file")
    parser.add_argument("--stats-dir", default="outputs/corpus_stats",
                        help="Directory to save corpus statistics")

    # Parameters
    parser.add_argument("--size", type=int, default=600,
                        help="Size of ground truth dataset")

    args = parser.parse_args()

    if args.command == "split":
        split_dataset(
            mixed_file=args.mixed,
            pidgin_file=args.pidgin,
            english_file=args.english,
            ground_truth_file=args.ground_truth,
            ground_truth_size=args.size
        )
    elif args.command == "stats":
        create_corpus_statistics(
            english_file=args.english,
            pidgin_file=args.pidgin,
            mixed_file=args.mixed,
            output_dir=args.stats_dir
        )