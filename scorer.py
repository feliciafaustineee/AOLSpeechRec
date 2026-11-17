"""Scorer berbasis template DTW: dipakai oleh train_scorer.py dan app.py.
Fitur MFCC dibuat manual (mfcc.py), jarak dihitung dengan DTW (dtw_knn.py).
"""
import pickle
import numpy as np
from dtw_knn import dtw_distance

OTHER = "__OTHER__"
STEP = 2  # setiap frame kedua, untuk mempercepat DTW


def load_scorer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def classify(feat, templates):
    """feat: fitur MFCC (sudah di-downsample). Kembalikan (label, jarak terkecil)."""
    best_lab, best_d = None, float("inf")
    for lab, feats in templates.items():
        for t in feats:
            d = dtw_distance(feat, t)
            if d < best_d:
                best_lab, best_d = lab, d
    return best_lab, best_d
