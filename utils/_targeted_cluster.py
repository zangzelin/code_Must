import numpy as np
from tqdm import trange
from typing import Optional, Union

# For Py3.7- compatibility
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ._cluster import cluster

def targeted_cluster(latent: np.ndarray,
                     target_n_clusters: int = None,
                     max_iter: int = 500,
                     method: Union[Literal["leiden", "louvain"], str] = "louvain",
                     random_state: int = 0,
                     n_neighbors: int = 12,
                     resolution: float = 0.55, ) -> np.ndarray:
    """
    Targeted clustering for specified cluster numbers.

    Parameters
    ----------
    latent:             2-d array
        2-d array to be clustered.
    target_n_clusters   int
        Optional target cluster numbers.
    max_iter            int
        Maximum iteration counts for specified cluster numbers.
    method:         "louvain" or "leiden"
        method used to cluster.
    n_neighbors:        int
        Number of neighborhood to be discovered.
    resolution:         float
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    random_state:       int
        Seed for reproducibility.

    Returns
    -------
    label:              1-d array
        Clustered label as integers.

    """
    label = None
    visited = {}
    tbar = trange(max_iter)
    delta_res = 0.02
    for run_idx in tbar:
        label = cluster(latent=latent,
                        method=method,
                        random_state=random_state,
                        n_neighbors=int(n_neighbors),
                        resolution=resolution)

        class_num = len(np.unique(label))
        diff = class_num - target_n_clusters
        tbar.set_description(f"[{class_num:2d}:{target_n_clusters:2d}] res:{resolution:.4f} n_nbrs:{n_neighbors:2d}")

        if diff == 0:
            break
        elif run_idx == max_iter - 1:
            raise RuntimeError("Hit iteration limit!")

        direct = 1 if diff > 0 else -1

        if np.abs(diff) > 7:
            n_neighbors += direct * 2 if n_neighbors > 4 else 0
            resolution -= direct * 0.1 if resolution > 0.1 else 0
        elif np.abs(diff) > 3:
            n_neighbors += direct * 1 if n_neighbors > 4 else 0
            resolution -= direct * 0.05 if resolution > 0.05 else 0
        else:
            if delta_res < 0.0001:
                delta_res = 0.02
                n_neighbors = max(n_neighbors + direct * 2, 1)
            elif resolution < 0.0001:
                resolution = 0.2
                n_neighbors = max(n_neighbors + direct * 2, 1)
            elif resolution > delta_res:
                resolution -= direct * delta_res
                if visited.get((n_neighbors, resolution), None) is True:
                    delta_res /= 2
            elif visited.get((n_neighbors, resolution), None) is True:
                n_neighbors += direct * 3 if n_neighbors > 3 else 0
            else:
                raise RuntimeError(f"Unable to cluster as {target_n_clusters} classes!")

        visited[(n_neighbors, resolution)] = True

    return label
