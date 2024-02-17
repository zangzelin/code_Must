import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, confusion_matrix
from ._cluster_map import cluster_map


# TODO: str support
def aligned_accuracy_score(true: np.ndarray, pred: np.ndarray, wildcard: int = None) -> np.ndarray:
    """
    Provide accuracy score for unsupervised label, like cluster analysis.

    Currently, this method only support integer labels.

    Parameters
    ----------
    true    1-d int array
        True label.
    pred    1-d int array
        Prediction.

    Returns
    -------
    Accuracy score for matched unsupervised label.

    """
    mask = true != wildcard
    true = true[mask]
    pred = pred[mask]
    pred = cluster_map(true, pred, wildcard=999)

    if len(np.unique(true)) != len(np.unique(pred)):
        raise ValueError(f"True label has {len(np.unique(true))} classes "
                         f"but Prediction has {len(np.unique(pred))} classes!")

    return accuracy_score(true, pred)
