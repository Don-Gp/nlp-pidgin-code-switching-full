#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nlp-pidgin-code-switching",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Nigerian Pidgin English Code-Switching Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp-pidgin-code-switching",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "colorlog>=6.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pidgin-code-switching=src.main:main",
        ],
    },
)