#!/usr/bin/env python3
"""Script that prompts the user in a question and answer loop."""


def main():
    """Run the basic interactive question loop."""
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")
        if question.lower() in exit_words:
            print("A: Goodbye")
            break
        print("A:")


if __name__ == "__main__":
    main()
