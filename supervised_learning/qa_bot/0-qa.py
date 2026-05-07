#!/usr/bin/env python3
"""Module that answers questions from a reference document."""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


TOKENIZER = None
MODEL = None


def load_tokenizer():
    """Load the pretrained Bert tokenizer.

    Returns:
        BertTokenizer: Pretrained tokenizer.
    """
    global TOKENIZER

    if TOKENIZER is None:
        TOKENIZER = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )
    return TOKENIZER


def load_model():
    """Load the pretrained BERT question answering model.

    Returns:
        tensorflow_hub.KerasLayer: Question answering model.
    """
    global MODEL

    if MODEL is None:
        MODEL = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    return MODEL


def question_answer(question, reference):
    """Find a snippet of text within a reference document to answer a question.

    Args:
        question (str): Question to answer.
        reference (str): Reference document to search.

    Returns:
        str: Answer snippet, or None if no answer is found.
    """
    tokenizer = load_tokenizer()
    model = load_model()

    encoded = tokenizer.encode_plus(question, reference,
                                    return_tensors='tf',
                                    truncation=True,
                                    max_length=512)

    inputs = {
        "input_ids": encoded["input_ids"],
        "input_mask": encoded["attention_mask"],
        "segment_ids": encoded["token_type_ids"]
    }

    outputs = model(inputs)
    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    input_ids = encoded["input_ids"][0].numpy()
    token_type_ids = encoded["token_type_ids"][0].numpy()

    cls_score = start_logits[0] + end_logits[0]
    best_score = float("-inf")
    best_span = None

    for start_index in range(len(input_ids)):
        if token_type_ids[start_index] != 1:
            continue

        end_limit = min(len(input_ids), start_index + 16)
        for end_index in range(start_index, end_limit):
            if token_type_ids[end_index] != 1:
                break

            score = start_logits[start_index] + end_logits[end_index]
            if score > best_score:
                best_score = score
                best_span = (start_index, end_index)

    if best_span is None or best_score <= cls_score:
        return None

    start_index, end_index = best_span
    answer_tokens = tokenizer.convert_ids_to_tokens(
        input_ids[start_index:end_index + 1]
    )
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    if answer == "":
        return None

    return answer
