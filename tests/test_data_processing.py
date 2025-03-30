"""
Tests for data preprocessing functionality.
"""

import os
import pytest
import re

from src.data.preprocessing import TextPreprocessor
from src.data.corpus_builder import extract_sentences, create_mixed_code_switching_ground_truth

def test_text_preprocessor_initialization():
    """Test that TextPreprocessor initializes correctly."""
    preprocessor = TextPreprocessor()
    assert hasattr(preprocessor, 'pidgin_markers')
    assert len(preprocessor.pidgin_markers) > 0

def test_clean_text():
    """Test text cleaning functionality."""
    preprocessor = TextPreprocessor()
    
    # Test with URLs
    text_with_url = "Check this website https://example.com for more details."
    cleaned = preprocessor.clean_text(text_with_url, remove_urls=True)
    assert "https://example.com" not in cleaned
    
    # Test lowercasing
    text_with_caps = "This TEXT has CAPITALS."
    cleaned = preprocessor.clean_text(text_with_caps, lowercase=True)
    assert cleaned == "this text has capitals."
    
    # Test spacing
    text_with_spaces = "  Too   many    spaces   "
    cleaned = preprocessor.clean_text(text_with_spaces, fix_spacing=True)
    assert cleaned == "Too many spaces"

def test_extract_sentences(sample_text, temp_dir):
    """Test extracting sentences from mixed text."""
    # Save sample text to a temporary file
    sample_file = os.path.join(temp_dir, "sample.txt")
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    # Extract sentences
    pidgin_sentences, english_sentences = extract_sentences(sample_file)
    
    assert len(pidgin_sentences) == 1
    assert len(english_sentences) == 2
    assert "Dis na example Pidgin text." in pidgin_sentences
    assert "This is a sample English text." in english_sentences

def test_create_ground_truth():
    """Test creating ground truth with code-switching."""
    pidgin_sentences = ["Dis na Pidgin.", "How body dey?", "Make we go."]
    english_sentences = ["This is English.", "Hello there.", "Good day."]
    
    ground_truth = create_mixed_code_switching_ground_truth(
        pidgin_sentences, english_sentences, ground_truth_size=6
    )
    
    assert len(ground_truth) == 6
    
    # Check if code-switching patterns exist
    intra_switching = any("<english>" in gt and "<pidgin>" in gt for gt in ground_truth)
    assert intra_switching