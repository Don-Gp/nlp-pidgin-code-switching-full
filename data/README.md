# Data

The `data/` directory contains all text files for training, testing, and evaluation.


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



## File Format Rules

- **Encoding:** UTF‑8 without BOM  
- **Sentences:** One sentence per line, plain text.  
- **Labels:** In ground truth files, each token is tagged as `token\tTAG` with a tab separator.  
- **Alignment:** Line *n* in input matches line *n* in its ground truth file.

## Data Sources & License

- **English & Pidgin corpora**: Public domain or permissively licensed sources.  
- **Mixed and annotations**: Created by this project team; available under CC BY‑SA.  

## Usage Notes

- **Quick checks:** Use files in `data/test/` for fast runs without loading full corpora.  
- **Do not commit:** Large or derived files (models, plots, logs) should be ignored via `.gitignore`.  
- **Adding data:** Place new corpora in `data/corpus/` and update references in `config/*.yaml`.


## Corpus Statistics
- **English Corpus:** 21,453 sentences, 374,129 words, 16.2% unique words

- **Pidgin Corpus:** 17,852 sentences, 283,541 words, 19.8% unique words

- **Mixed Corpus:** Contains 31,245 English segments and 28,763 Pidgin segments

- **Ground Truth:** Contains 3,574 code-switching instances


