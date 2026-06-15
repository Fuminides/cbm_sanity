"""Small dependency-free metrics."""

from __future__ import annotations


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute accuracy on empty inputs")
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute macro F1 on empty inputs")

    scores = []
    for label in range(num_classes):
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return sum(scores) / num_classes


def concept_agreement(concepts_a: list[list[int]], concepts_b: list[list[int]]) -> float:
    if len(concepts_a) != len(concepts_b):
        raise ValueError("Concept prediction lists must have the same length")
    if not concepts_a:
        raise ValueError("Cannot compute concept agreement on empty inputs")

    total = 0
    matches = 0
    for vector_a, vector_b in zip(concepts_a, concepts_b):
        if len(vector_a) != len(vector_b):
            raise ValueError("Concept vectors must have the same length")
        for value_a, value_b in zip(vector_a, vector_b):
            total += 1
            matches += int(value_a == value_b)
    return matches / total

