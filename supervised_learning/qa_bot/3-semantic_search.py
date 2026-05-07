#!/usr/bin/env python3
"""Module that performs semantic search on a corpus of documents."""
import os
import tensorflow as tf
import tensorflow_hub as hub


MODEL = None


def load_model():
    """Load the Universal Sentence Encoder model.

    Returns:
        tensorflow_hub.KerasLayer: USE model.
    """
    global MODEL

    if MODEL is None:
        MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return MODEL


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents.

    Args:
        corpus_path (str): Path to the corpus of documents.
        sentence (str): Sentence to compare against the corpus.

    Returns:
        str: Reference text of the document most similar to sentence.
    """
    documents = []

    for file_name in sorted(os.listdir(corpus_path)):
        file_path = os.path.join(corpus_path, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())

    if len(documents) == 0:
        return None

    model = load_model()
    embeddings = model([sentence] + documents)

    sentence_embedding = embeddings[0]
    document_embeddings = embeddings[1:]

    similarities = tf.linalg.matmul(document_embeddings,
                                    tf.expand_dims(sentence_embedding, axis=1))
    similarities /= (
        tf.norm(document_embeddings, axis=1, keepdims=True) *
        tf.norm(sentence_embedding)
    )

    index = tf.argmax(tf.squeeze(similarities, axis=1)).numpy()
    return documents[index]
