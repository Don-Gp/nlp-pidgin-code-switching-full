"""
pytest configuration file for shared fixtures.
"""

import os
import pytest
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_text():
    """Sample text with English and Pidgin content."""
    return """<english>This is a sample English text.</english>
<pidgin>Dis na example Pidgin text.</pidgin>
<english>Another English sentence here.</english>"""

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'english_corpus': 'data/corpus/english.txt',
            'pidgin_corpus': 'data/corpus/pidgin.txt',
            'mixed_corpus': 'data/corpus/mixed.txt'
        },
        'ml': {
            'validation_split': 0.1,
            'sample_rate': 3,
            'window_size': 5
        }
    }