"""
Text processing utilities for the Nigerian Pidgin/English code-switching project.
Handles cleaning, tokenization, and n-gram extraction.
"""

import re
import string
from typing import List, Dict, Set

# Common Pidgin markers that often indicate Pidgin usage
PIDGIN_MARKERS = {
    "na", "dey", "abi", "wetin", "oga", "sabi", "don", "dem",
    "una", "abeg", "wahala", "jare", "sha", "sef", "kuku", "shey"
}

def clean_text(text):
    """Basic text cleaning"""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Keep apostrophes in words but remove other special chars
    text = re.sub(r'[^\w\s\']', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def tokenize(text, lowercase=True):
    """Split text into tokens"""
    if lowercase:
        text = text.lower()
    return text.split()

def get_char_ngrams(text, n):
    """Extract character n-grams from text"""
    if len(text) < n:
        return []

    return [text[i:i+n] for i in range(len(text) - n + 1)]

def get_word_ngrams(tokens, n):
    """Extract word n-grams from tokens"""
    if len(tokens) < n:
        return []

    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def contains_pidgin_markers(text):
    """Check if text contains common Pidgin markers"""
    tokens = tokenize(text)
    return any(token in PIDGIN_MARKERS for token in tokens)

def tag_languages(text, labels):
    """Add XML-style tags based on language labels"""
    if len(text) != len(labels):
        raise ValueError("Text and labels must have the same length")

    result = []
    current_lang = labels[0]
    current_text = text[0]

    for i in range(1, len(text)):
        if labels[i] == current_lang:
            current_text += text[i]
        else:
            # Language switch detected
            tag = "pidgin" if current_lang == "P" else "English"
            result.append(f"<{tag}>{current_text}</{tag}>")

            # Start new segment
            current_lang = labels[i]
            current_text = text[i]

    # Add final segment
    tag = "pidgin" if current_lang == "P" else "English"
    result.append(f"<{tag}>{current_text}</{tag}>")

    return "".join(result)

def extract_tagged_text(tagged_text):
    """Extract text and language labels from tagged text"""
    text = ""
    labels = ""

    # Simple regex to extract text and labels
    pidgin_pattern = r"<pidgin>(.*?)</pidgin>"
    english_pattern = r"<English>(.*?)</English>"

    # Current position in the processed text
    pos = 0

    # Find all pidgin sections
    for match in re.finditer(pidgin_pattern, tagged_text):
        # Text before this match is English
        start, end = match.span()

        # Skip any already processed text
        if start < pos:
            continue

        # Add English text if there's gap before match
        if start > pos:
            english_text = tagged_text[pos:start]
            # Remove any XML tags
            english_text = re.sub(r"<.*?>", "", english_text)
            text += english_text
            labels += "E" * len(english_text)

        # Add pidgin text
        pidgin_text = match.group(1)
        text += pidgin_text
        labels += "P" * len(pidgin_text)

        # Update position
        pos = end

    # Add any remaining English text
    if pos < len(tagged_text):
        english_text = tagged_text[pos:]
        # Remove any XML tags
        english_text = re.sub(r"<.*?>", "", english_text)
        text += english_text
        labels += "E" * len(english_text)

    return text, labels