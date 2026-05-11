# Error Analysis

Implementation of classification error metrics from scratch using NumPy,
covering confusion matrices and the key metrics derived from them.

## Overview

This project builds a toolkit for evaluating multi-class classifiers:

- **Confusion matrix**: count of true vs predicted labels for every class pair
- **Sensitivity** (recall): ability to detect actual positives — TP / (TP + FN)
- **Precision**: accuracy of positive predictions — TP / (TP + FP)
- **Specificity**: ability to reject negatives — TN / (TN + FP)
- **F1 score**: harmonic mean of precision and recall

Additional written tasks cover error handling strategies and a
comparison/contrast of error types.

## Project Structure

```text
error_analysis/
├── 0-create_confusion.py   # Build a confusion matrix from labels/logits
├── 1-sensitivity.py        # Per-class sensitivity (recall)
├── 2-precision.py          # Per-class precision
├── 3-specificity.py        # Per-class specificity
├── 4-f1_score.py           # Per-class F1 score
├── 5-error_handling        # Written: error handling strategies
├── 6-compare_and_contrast  # Written: comparing error metrics
├── 0-main.py
├── 1-main.py
├── 2-main.py
├── 3-main.py
├── 4-main.py
└── README.md
```

## Tasks

| File | Function | Description |
|---|---|---|
| `0-create_confusion.py` | `create_confusion_matrix` | Create a confusion matrix from one-hot arrays |
| `1-sensitivity.py` | `sensitivity` | Sensitivity (recall) per class |
| `2-precision.py` | `precision` | Precision per class |
| `3-specificity.py` | `specificity` | Specificity per class |
| `4-f1_score.py` | `f1_score` | F1 score per class |
| `5-error_handling` | — | Written response on error handling |
| `6-compare_and_contrast` | — | Written comparison of error metrics |

## Key Concepts

### Confusion Matrix

A `(classes, classes)` matrix where `confusion[i][j]` is the number of
samples whose true label is `i` but were predicted as `j`. The diagonal
contains correct predictions (true positives per class).

### Metrics Derived from the Confusion Matrix

For each class `i`, define:

| Symbol | Meaning |
|---|---|
| TP | Samples correctly predicted as class `i` |
| FP | Samples incorrectly predicted as class `i` |
| FN | Samples of class `i` incorrectly predicted as another |
| TN | All other correctly rejected samples |

| Metric | Formula |
|---|---|
| Sensitivity / Recall | TP / (TP + FN) |
| Precision | TP / (TP + FP) |
| Specificity | TN / (TN + FP) |
| F1 Score | 2 · Precision · Recall / (Precision + Recall) |

## Requirements

- Python 3.9
- NumPy 1.25.2

## Usage

```bash
python3 0-main.py    # Build and print confusion matrix
python3 1-main.py    # Sensitivity per class
python3 2-main.py    # Precision per class
python3 3-main.py    # Specificity per class
python3 4-main.py    # F1 score per class
```

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/error_analysis`
- **Language**: Python

---

**Author**: Arber Zylyftari
