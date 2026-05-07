#!/usr/bin/env python3
"""Module that answers questions from a single reference text."""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answer questions from a reference text.

    Args:
        reference (str): Reference text used to answer questions.
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        if answer is None:
            answer = "Sorry, I do not understand your question."
        print("A:", answer)
