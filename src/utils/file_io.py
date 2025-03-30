"""Utility functions for file input/output operations."""

import os
import json
import pickle
import yaml
from typing import Any, Dict, List, Union


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_text_file(filepath: str, encoding: str = 'utf-8') -> str:
    """Read text from a file.

    Args:
        filepath: Path to the text file
        encoding: File encoding (default: utf-8)

    Returns:
        Contents of the file as a string
    """
    with open(filepath, 'r', encoding=encoding) as f:
        return f.read()


def write_text_file(filepath: str, content: str, encoding: str = 'utf-8') -> None:
    """Write text to a file.

    Args:
        filepath: Path to the output file
        content: Text content to write
        encoding: File encoding (default: utf-8)
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(content)


def read_json(filepath: str) -> Dict:
    """Read a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Parsed JSON content as a dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(filepath: str, data: Dict, indent: int = 4) -> None:
    """Write data to a JSON file.

    Args:
        filepath: Output JSON file path
        data: Data to serialize as JSON
        indent: Indentation spaces for pretty printing
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def load_yaml_config(config_path: str) -> Dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Configuration as a dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """Save an object to a pickle file.

    Args:
        obj: Object to pickle
        filepath: Output pickle file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load an object from a pickle file.

    Args:
        filepath: Path to the pickle file

    Returns:
        Unpickled object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)