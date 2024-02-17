import numpy as np


def make_error_label(y_true, y_pred, wildcard=None):
    mask = y_true == y_pred
    if wildcard is not None:
        mask |= (y_true == wildcard)
    dic = {False: 0, True: 1}

    return np.vectorize(dic.__getitem__)(mask)
