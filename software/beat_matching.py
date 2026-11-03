"""
beat_matching.py -- greedy one-to-one matching of detected peaks to annotated beats.

Used by beat_dataset.py, evaluate.py and make_board_vectors.py to label each detected beat with
the AAMI class of the annotation within 150 ms. (Moved verbatim out of the earlier
detector-dataset experiment, archive/python/build_dataset_det.py.)
"""
import numpy as np


def match(det, ann, tol):
    """Greedy 1-1 match. Returns list of (det_index, ann_index)."""
    used = np.zeros(len(det), bool); out = []
    for ai, a in enumerate(ann):
        if not len(det): break
        d = np.abs(det - a).astype(float); d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol: used[j] = True; out.append((j, ai))
    return out
