import numpy as np


def confusion_matrix(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn


def accuracy(y_true, y_pred):

    tp, tn, fp, fn = confusion_matrix(
        y_true,
        y_pred
    )

    total = tp + tn + fp + fn

    if total == 0:
        return 0.0

    return (tp + tn) / total


def precision(y_true, y_pred):

    tp, tn, fp, fn = confusion_matrix(
        y_true,
        y_pred
    )

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def recall(y_true, y_pred):

    tp, tn, fp, fn = confusion_matrix(
        y_true,
        y_pred
    )

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def f1_score(y_true, y_pred):

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    denominator = p + r

    if denominator == 0:
        return 0.0

    return 2 * (p * r) / denominator