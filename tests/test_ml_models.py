"""
Tests for ML model functionality.
"""

import os
import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# We'll mock the actual model training to avoid dependencies
# These tests focus on the infrastructure rather than the model performance
def test_vectorizer_creation():
    """Test creating a character n-gram vectorizer."""
    texts = ["This is English text.", "Dis na Pidgin text."]
    
    # Create vectorizer
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X = vectorizer.fit_transform(texts)
    
    # Check features
    assert X.shape[0] == 2  # Two documents
    assert X.shape[1] > 0  # Some features
    
    # Check vocabulary
    vocab = vectorizer.get_feature_names_out()
    assert len(vocab) > 0
    assert ' is' in vocab  # Space should be included in ngrams
    
def test_window_creation():
    """Test creating character windows for ML models."""
    text = "This is a test."
    window_size = 3
    
    windows = []
    for i in range(len(text)):
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
        
        windows.append(window)
    
    # Check window properties
    assert len(windows) == len(text)
    assert all(len(w) <= 2 * window_size + 1 for w in windows)