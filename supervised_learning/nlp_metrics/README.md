# NLP Metrics

Implementation of BLEU-based evaluation metrics for natural language
processing tasks using pure Python and NumPy.

## Overview

This project covers the main BLEU score variants used to evaluate a
candidate translation against one or more references:

- **Unigram BLEU**: precision over single-word matches
- **N-gram BLEU**: precision over fixed-size n-grams
- **Cumulative BLEU**: geometric mean over multiple n-gram orders

## Project Structure

```text
nlp_metrics/
├── 0-uni_bleu.py
├── 0-main.py
├── 1-ngram_bleu.py
├── 1-main.py
├── 2-cumulative_bleu.py
├── 2-main.py
└── README.md
```

## Usage

Run the sample files from inside the `nlp_metrics` directory:

```bash
python3 0-main.py
python3 1-main.py
python3 2-main.py
```

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/nlp_metrics`
- **Language**: Python

---

**Author**: Arber Zylyftari
