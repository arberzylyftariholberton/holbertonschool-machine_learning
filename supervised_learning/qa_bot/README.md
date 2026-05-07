# QA Bot

Implementation of a question answering bot using BERT-based extractive
question answering and semantic search across a corpus of documents.

## Overview

This project combines transformer-based answer extraction with semantic
document retrieval:

- **Reference QA**: extract an answer span from one document
- **Interactive Loop**: simple question and answer prompt
- **Single-Reference QA Loop**: answer questions from one text
- **Semantic Search**: retrieve the most relevant document in a corpus
- **Multi-Reference QA**: answer questions across many documents

## Project Structure

```text
qa_bot/
├── 0-qa.py
├── 1-loop.py
├── 2-qa.py
├── 3-semantic_search.py
├── 4-qa.py
└── README.md
```

## Requirements

- Python 3.9
- NumPy 1.25.2
- TensorFlow 2.15
- tensorflow-hub 0.15.0
- transformers 4.44.2

Install dependencies with:

```bash
pip install --user tensorflow-hub==0.15.0
pip install --user transformers==4.44.2
```

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/qa_bot`
- **Language**: Python

---

**Author**: Arber Zylyftari
