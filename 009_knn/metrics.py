import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1, 1]
    fp = cm[0, 1]

    return tp / (tp + fp)

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1, 1]
    fn = cm[1, 0]

    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * (p * r) / (p + r)