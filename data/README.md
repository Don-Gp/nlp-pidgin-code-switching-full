# Data Directory

This directory contains the corpus data and related files for the Nigerian Pidgin-English Code-Switching Detection project.

## Directory Structure
data/
├── corpus/         # Core corpus files
│   ├── mixed.txt            # Mixed corpus with language tags
│   ├── english.txt          # English corpus
│   ├── pidgin.txt           # Pidgin corpus
│   ├── ground_truth/  # Ground truth data for evaluation
├── raw/            # Raw, unprocessed text files
├── interim/        # Intermediate processed data
└── test/           # Test data files       # Plain text for testing
    ├── ml_test.txt     # English test corpus
    └── ppm_test.txt      # Pidgin test corpus

## File Formats

### Mixed Corpus (mixed.txt)
Contains tagged text indicating language boundaries:
<english>This is English text.</english>
<pidgin>Dis na Pidgin text.</pidgin>
Copy
### Language-Specific Corpora (english.txt, pidgin.txt)
Single-language corpus files, one sentence per line:
This is a sample English sentence.
Another English sentence follows.
Copy
### Ground Truth (ml_ground_truth.txt)
Reference data with proper language tagging used for evaluation:
<english>This is English.</english> <pidgin>Dis na Pidgin.</pidgin>
<pidgin>How body dey?</pidgin>
Copy
### Test Data (test_text.txt)
Plain text without language tags used for testing model predictions:
This is English. Dis na Pidgin.
How body dey?
Copy
## Corpus Statistics

- **English Corpus**: Approximately X sentences, Y words, Z% unique words
- **Pidgin Corpus**: Approximately X sentences, Y words, Z% unique words
- **Mixed Corpus**: Contains X English segments and Y Pidgin segments
- **Ground Truth**: Contains X code-switching instances

## Data Sources

The corpus data is derived from various sources including:
- Nigerian news articles
- Social media posts
- Transcribed conversations
- Literary texts

## Usage

The corpus files are used for training and evaluating the code-switching detection models:
- `mixed.txt` is used to extract language segments
- `english.txt` and `pidgin.txt` are used for training language-specific models
- `ml_ground_truth.txt` is used for evaluation
- `test_text.txt` is used for testing model predictions

## Data Processing

The raw data goes through several processing steps:
1. Text extraction and cleaning
2. Language identification and tagging
3. Corpus separation and organization
4. Ground truth preparation

To recreate the corpus from raw data, use the corpus builder:
python -m src.data.corpus_builder
Copy
## Contributing Data

To add new data to the corpus:
1. Place raw text files in the `raw/` directory
2. Tag language segments with `<english>` and `<pidgin>` tags
3. Run the corpus builder to update the corpus files