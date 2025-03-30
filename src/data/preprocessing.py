#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing module for cleaning and preparing text data for
Nigerian Pidgin English code-switching detection.
"""

import os
import sys
import re
import unicodedata
import string
from collections import Counter
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.logging_utils import setup_logger
from src.utils.file_io import read_text_file, write_text_file


# Set up logger
logger = setup_logger('preprocessing')


class TextPreprocessor:
    """
    Class for preprocessing text data for Nigerian Pidgin English
    code-switching detection.
    """

    def __init__(self):
        """
        Initialize the text preprocessor.
        """
        # Common pidgin markers that might help with identification
        self.pidgin_markers = [
            'na', 'dey', 'don', 'dem', 'abi', 'wetin', 'oga', 'sabi',
            'abeg', 'wahala', 'wey', 'make', 'una', 'nna', 'chop',
            'pikin', 'dey', 'go', 'come', 'talk', 'gist', 'kolo',
            'kpuff', 'yarn', 'jare', 'kpai', 'kpatakpata', 'sha',
            'shebi', 'shey', 'sef', 'no be', 'how far', 'wetyn',
            'bros', 'guy', 'geh', 'doh', 'nawa', 'wahala'
        ]

    def normalize_unicode(self, text):
        """
        Normalize Unicode characters in text.

        Args:
            text (str): Input text

        Returns:
            str: Normalized text
        """
        return unicodedata.normalize('NFKC', text)

    def clean_text(self, text, lowercase=True, remove_urls=True,
                   remove_numbers=False, remove_punctuation=False,
                   fix_spacing=True):
        """
        Clean text by applying various preprocessing steps.

        Args:
            text (str): Input text
            lowercase (bool): Convert to lowercase
            remove_urls (bool): Remove URLs
            remove_numbers (bool): Remove numbers
            remove_punctuation (bool): Remove punctuation
            fix_spacing (bool): Fix spacing issues

        Returns:
            str: Cleaned text
        """
        # Normalize Unicode
        text = self.normalize_unicode(text)

        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', ' ', text)

        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Convert to lowercase
        if lowercase:
            text = text.lower()

        # Fix spacing issues
        if fix_spacing:
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text)
            # Remove spaces at the beginning and end
            text = text.strip()

        return text

    def tokenize_text(self, text, by_character=False):
        """
        Tokenize text into words or characters.

        Args:
            text (str): Input text
            by_character (bool): Tokenize by character if True, else by word

        Returns:
            list: Tokens
        """
        if by_character:
            # Character-level tokenization
            return list(text)
        else:
            # Word-level tokenization
            return text.split()

    def extract_ngrams(self, tokens, n):
        """
        Extract n-grams from a list of tokens.

        Args:
            tokens (list): Input tokens
            n (int): N-gram size

        Returns:
            list: N-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n]) if isinstance(tokens[i], str) else ''.join(tokens[i:i + n])
            ngrams.append(ngram)
        return ngrams

    def identify_pidgin_markers(self, text):
        """
        Identify potential Pidgin markers in text.

        Args:
            text (str): Input text

        Returns:
            list: Identified markers and their counts
        """
        # Clean and tokenize text
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)

        # Count occurrences of pidgin markers
        marker_counts = Counter()
        for token in tokens:
            if token in self.pidgin_markers:
                marker_counts[token] += 1

        return marker_counts

    def tag_potential_pidgin(self, text):
        """
        Tag potential Pidgin sections based on markers.
        This is a simple heuristic approach and not a replacement
        for proper language identification.

        Args:
            text (str): Input text

        Returns:
            str: Text with XML tags indicating potential Pidgin sections
        """
        # Clean text while preserving case and punctuation
        cleaned_text = self.clean_text(text, lowercase=False, remove_punctuation=False)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        tagged_sentences = []
        for sentence in sentences:
            # Check if sentence contains pidgin markers
            words = sentence.lower().split()
            has_pidgin = any(marker in words for marker in self.pidgin_markers)

            if has_pidgin:
                tagged_sentences.append(f"<pidgin>{sentence}</pidgin>")
            else:
                tagged_sentences.append(f"<english>{sentence}</english>")

        return " ".join(tagged_sentences)

    def process_file(self, input_file, output_file, lowercase=True, remove_urls=True,
                     remove_numbers=False, remove_punctuation=False, fix_spacing=True):
        """
        Process an entire file with preprocessing steps.

        Args:
            input_file (str): Input file path
            output_file (str): Output file path
            lowercase (bool): Convert to lowercase
            remove_urls (bool): Remove URLs
            remove_numbers (bool): Remove numbers
            remove_punctuation (bool): Remove punctuation
            fix_spacing (bool): Fix spacing issues

        Returns:
            bool: Success status
        """
        try:
            # Read input file
            text = read_text_file(input_file)

            # Apply preprocessing
            processed_text = self.clean_text(
                text,
                lowercase=lowercase,
                remove_urls=remove_urls,
                remove_numbers=remove_numbers,
                remove_punctuation=remove_punctuation,
                fix_spacing=fix_spacing
            )

            # Write output file
            write_text_file(output_file, processed_text)

            logger.info(f"Processed {input_file} -> {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            return False

    def prepare_data_for_training(self, input_file, output_dir, split_ratio=0.8):
        """
        Prepare data for training by splitting into train and test sets.

        Args:
            input_file (str): Input file path
            output_dir (str): Output directory
            split_ratio (float): Train/test split ratio

        Returns:
            tuple: Paths to train and test files
        """
        try:
            # Read and clean text
            text = read_text_file(input_file)
            clean_text_value = self.clean_text(text)

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', clean_text_value)

            # Shuffle sentences
            import random
            random.shuffle(sentences)

            # Calculate split
            split_idx = int(len(sentences) * split_ratio)
            train_sentences = sentences[:split_idx]
            test_sentences = sentences[split_idx:]

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Write train and test files
            train_file = os.path.join(output_dir, 'train.txt')
            test_file = os.path.join(output_dir, 'test.txt')

            write_text_file(train_file, '\n'.join(train_sentences))
            write_text_file(test_file, '\n'.join(test_sentences))

            logger.info(f"Split {input_file} into {train_file} and {test_file}")
            logger.info(f"Train set: {len(train_sentences)} sentences")
            logger.info(f"Test set: {len(test_sentences)} sentences")

            return train_file, test_file

        except Exception as e:
            logger.error(f"Error preparing data from {input_file}: {e}")
            return None, None

    def extract_tagged_sections(self, input_file, pidgin_output=None, english_output=None):
        """
        Extract Pidgin and English sections from a tagged file.

        Args:
            input_file (str): Input file path with XML tags
            pidgin_output (str): Output file for Pidgin sections
            english_output (str): Output file for English sections

        Returns:
            tuple: Extracted Pidgin and English text
        """
        try:
            # Read file
            text = read_text_file(input_file)

            # Extract Pidgin sections
            pidgin_pattern = r'<pidgin>(.*?)</pidgin>'
            pidgin_sections = re.findall(pidgin_pattern, text, re.DOTALL)
            pidgin_text = '\n'.join(pidgin_sections)

            # Extract English sections
            english_pattern = r'<english>(.*?)</english>'
            english_sections = re.findall(english_pattern, text, re.DOTALL)
            english_text = '\n'.join(english_sections)

            # Write output files if specified
            if pidgin_output:
                write_text_file(pidgin_output, pidgin_text)
                logger.info(f"Extracted Pidgin sections to {pidgin_output}")

            if english_output:
                write_text_file(english_output, english_text)
                logger.info(f"Extracted English sections to {english_output}")

            return pidgin_text, english_text

        except Exception as e:
            logger.error(f"Error extracting sections from {input_file}: {e}")
            return "", ""

if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()

    # Example text with code-switching
    text = "How mama be today? You no sabi book but you sabi plenty thing wey pass book, my dear girl what a waste of effort."

    # Clean text
    cleaned_text = preprocessor.clean_text(text)
    print("Cleaned text:", cleaned_text)

    # Identify pidgin markers
    markers = preprocessor.identify_pidgin_markers(text)
    print("Pidgin markers:", markers)

    # Tag potential pidgin
    tagged_text = preprocessor.tag_potential_pidgin(text)
    print("Tagged text:", tagged_text)
