# Word Embeddings

Implementation of classic and neural word embedding techniques using
NumPy, gensim, and Keras.

## Overview

This project introduces several core approaches for representing words
and sentences numerically:

- **Bag of Words**: count-based sentence embeddings
- **TF-IDF**: weighted term-frequency sentence embeddings
- **Word2Vec**: learned dense word embeddings
- **Gensim to Keras**: converts pretrained embeddings to Keras
- **FastText**: subword-aware dense word embeddings
- **ELMo Theory**: conceptual understanding of ELMo training

## Requirements

- Python 3.9
- NumPy 1.25.2
- TensorFlow 2.15
- Keras 2.15.0
- gensim 4.3.3
- pycodestyle 2.11.1

Install gensim with:

```bash
pip install --user gensim==4.3.3
```

## Project Structure

```text
word_embeddings/
├── 0-bag_of_words.py
├── 0-main.py
├── 1-tf_idf.py
├── 1-main.py
├── 2-word2vec.py
├── 2-main.py
├── 3-gensim_to_keras.py
├── 3-main.py
├── 4-fasttext.py
├── 4-main.py
├── 5-elmo
└── README.md
```

## Tasks

### Task 0: Bag of Words

Builds a count-based embedding matrix where each row represents a
sentence and each column represents a vocabulary feature.

### Task 1: TF-IDF

Builds a sentence embedding matrix using term frequency-inverse document
frequency weights, followed by row normalization.

### Task 2: Train Word2Vec

Creates and trains a gensim `Word2Vec` model using either:

- CBOW
- Skip-gram

### Task 3: Extract Word2Vec

Converts a trained gensim Word2Vec model into a trainable Keras
`Embedding` layer.

### Task 4: FastText

Creates and trains a gensim `FastText` model for word embeddings that
benefit from subword information.

### Task 5: ELMo

Identifies which parts of an ELMo model are trained when learning the
embedding model.

## Usage

Run the sample files from inside the `word_embeddings` directory:

```bash
python3 0-main.py
python3 1-main.py
python3 2-main.py
python3 3-main.py
python3 4-main.py
```

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/word_embeddings`
- **Language**: Python

---

**Author**: Arber Zylyftari
