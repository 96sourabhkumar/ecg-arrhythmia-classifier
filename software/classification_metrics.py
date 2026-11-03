"""
classification_metrics.py -- accuracy, macro-F1 (N, S, V, F), confusion matrix and per-class F1.
NumPy only, so evaluation runs without PyTorch.
"""
import numpy as np


def metrics(y, p):
    cm = np.zeros((5, 5), int)
    for t, q in zip(y, p): cm[t, q] += 1
    f1 = []
    for i in range(5):
        TP = cm[i, i]; FN = cm[i].sum() - TP; FP = cm[:, i].sum() - TP
        se = TP / (TP + FN) if TP + FN else 0; pp = TP / (TP + FP) if TP + FP else 0
        f1.append(2 * se * pp / (se + pp) if se + pp else 0)
    return np.trace(cm) / cm.sum(), float(np.mean(f1[:4])), cm, f1
