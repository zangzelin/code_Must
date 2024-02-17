import numpy as np

from typing import Union

# For Py3.7- compatibility
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def refine_label(label: np.ndarray,
                 method: Union[Literal["hexagon"], str] = "hexagon",
                 random_state: int = 0,
                 corrds: np.ndarray = None,
                 radius: int = 30,) -> np.ndarray:

    if method == "hexagon":
        if corrds is None:
            raise ValueError('Hexagonal refinement supposed to be based on corrdinations.')
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(radius=radius).fit(corrds)
        _, index = nbrs.radius_neighbors(corrds)

        refined_label = []
        for vec in index:
            cnt = np.bincount(label[vec])
            refined_label.append(np.argmax(cnt))

        refined_label = np.array(refined_label, dtype=int)
        
    return refined_label