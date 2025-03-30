"""
Tests for utility functions.
"""

import os
import pytest
import yaml
import json

from src.utils.file_io import (
    ensure_dir, read_file, write_file, read_config, write_config,
    read_json, write_json
)

def test_ensure_dir(temp_dir):
    """Test directory creation."""
    test_dir = os.path.join(temp_dir, "test_subdir")
    assert not os.path.exists(test_dir)
    
    # Create directory
    ensure_dir(test_dir)
    assert os.path.exists(test_dir)
    
    # Should not raise an error if directory already exists
    ensure_dir(test_dir)

def test_read_write_file(temp_dir):
    """Test reading and writing text files."""
    test_file = os.path.join(temp_dir, "test.txt")
    test_content = "This is a test content.\nWith multiple lines."
    
    # Write file
    write_file(test_file, test_content)
    assert os.path.exists(test_file)
    
    # Read file
    read_content = read_file(test_file)
    assert read_content == test_content

def test_read_write_config(temp_dir):
    """Test reading and writing YAML config files."""
    config_file = os.path.join(temp_dir, "config.yaml")
    config_data = {
        'section1': {
            'param1': 'value1',
            'param2': 42
        },
        'section2': {
            'param3': True,
            'param4': [1, 2, 3]
        }
    }
    
    # Write config
    write_config(config_data, config_file)
    assert os.path.exists(config_file)
    
    # Read config
    read_data = read_config(config_file)
    assert read_data == config_data

def test_read_write_json(temp_dir):
    """Test reading and writing JSON files."""
    json_file = os.path.join(temp_dir, "data.json")
    json_data = {
        'name': 'Test',
        'values': [1, 2, 3],
        'nested': {
            'key': 'value'
        }
    }
    
    # Write JSON
    write_json(json_data, json_file)
    assert os.path.exists(json_file)
    
    # Read JSON
    read_data = read_json(json_file)
    assert read_data == json_data