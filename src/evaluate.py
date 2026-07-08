from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def build_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def build_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

