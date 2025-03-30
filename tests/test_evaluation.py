
"""
Tests for evaluation metrics.
"""

import os
import pytest
import re

from src.evaluation.metrics import (
    normalize_tags, extract_text_without_tags, create_language_label_map,
    create_language_label_map_no_whitespace
)

def test_normalize_tags():
    """Test tag normalization."""
    text = "<English>Hello</English> <Pidgin>How body</Pidgin>"
    normalized = normalize_tags(text)
    
    assert "<english>Hello</english>" in normalized
    assert "<pidgin>How body</pidgin>" in normalized
    
    # Test fixing incorrect closing tags
    text = "<English>Hello</\\English> <Pidgin>How body</\\Pidgin>"
    normalized = normalize_tags(text)
    
    assert "<english>Hello</english>" in normalized
    assert "<pidgin>How body</pidgin>" in normalized

def test_extract_text_without_tags():
    """Test extracting text without tags."""
    text = "<english>Hello</english> <pidgin>How body</pidgin>"
    extracted = extract_text_without_tags(text)
    
    assert extracted == "Hello How body"
    assert "<" not in extracted
    assert ">" not in extracted

def test_create_language_label_map(sample_text):
    """Test creating language label map."""
    labels = create_language_label_map(sample_text)
    
    # Extract text without tags
    clean_text = extract_text_without_tags(sample_text)
    
    # Verify labels exist for characters
    assert len(labels) > 0
    assert all(isinstance(pos, int) for pos in labels.keys())
    assert all(label in ['E', 'P'] for label in labels.values())
    
    # Check if the number of labels is approximately correct
    # (There may be some differences due to whitespace handling)
    non_whitespace_chars = sum(1 for c in clean_text if not c.isspace())
    assert len(labels) <= non_whitespace_chars