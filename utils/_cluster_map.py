import numpy as np
from munkres import Munkres
from sklearn.metrics import confusion_matrix


def cluster_map(true: np.ndarray, pred: np.ndarray, wildcard: int = None) -> np.ndarray:
    mask = true != wildcard

    if len(np.unique(true[mask])) != len(np.unique(pred[mask])):
        raise ValueError(f"True label has {len(np.unique(true[mask]))} classes "
                         f"but Prediction has {len(np.unique(pred[mask]))} classes!")

    # temporary replace dis-continuous label
    enc = dict([(i, j) for i, j in zip(np.unique(true), np.arange(len(np.unique(true))))])
    dec = dict([(j, i) for i, j in enc.items()])
    if np.any(np.unique(true) != np.arange(len(np.unique(true)))):
        true = np.vectorize(enc.get)(true)
    if np.any(np.unique(pred) != np.arange(len(np.unique(pred)))):
        pred = np.vectorize(enc.get)(pred)

    cm = len(true[mask]) - confusion_matrix(pred[mask], true[mask])
    idx = Munkres().compute(cm)
    idx = dict(idx)
    
    pred = np.vectorize(idx.get)(pred)
    pred = np.vectorize(dec.get)(pred)

    return pred
