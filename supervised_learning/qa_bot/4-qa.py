#!/usr/bin/env python3
"""Module that answers questions from multiple reference documents."""
single_question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer_loop(corpus_path):
    """Answer questions from multiple reference texts.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        answer = single_question_answer(question, reference)

        if answer is None:
            answer = "Sorry, I do not understand your question."
        print("A:", answer)


def question_answer(corpus_path):
    """Start the multi-reference question answering loop.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
    """
    question_answer_loop(corpus_path)
