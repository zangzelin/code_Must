import anndata
import numpy as np

from typing import Optional, Union

# For Py3.7- compatibility
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def cluster(latent: np.ndarray,
            method: Union[Literal["leiden", "louvain", "kmeans", 'mclust'], str] = "louvain",
            random_state: int = 0,
            n_neighbors: int = 10,
            n_clusters: int = 10,
            resolution: float = 0.5,
            pca_dim: int = None,
            mclust_model: str = 'EEE') -> np.ndarray:
    """
    Easy clustering for clarity.

    Parameters
    ----------
    latent:             2-d array
        2-d array to be clustered.
    method:         "louvain", "leiden", "kmeans" or "mclust"
        method used to cluster.
    random_state:       int
        Seed for reproducibility.
    n_neighbors:        int
        Only in Louvain or Leiden. Number of neighborhood to be discovered.
    n_clusters:         int
        Only in KMeans. Number of cluster to be clustered.
    resolution:         float
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    pca_dim:            int
        Target dimesion after PCA, None for no PCA.
    mclust_model:       str
        Model mclusted to use.

    Returns
    -------
    label:              1-d array
        Clustered label as integers.

    """

    if pca_dim is not None:
        from sklearn.decomposition import PCA
        latent = PCA(n_components=pca_dim, random_state=random_state).fit_transform(latent)

    if method in ["louvain", "leiden"]:
        import scanpy as sc

        data = anndata.AnnData(latent, dtype=np.float32)
        sc.pp.neighbors(data, n_neighbors=n_neighbors, use_rep='X', random_state=random_state)
        if method == "louvain":
            sc.tl.louvain(data, resolution=resolution, random_state=random_state)
            label = data.obs["louvain"].to_numpy()
        elif method == "leiden":
            sc.tl.leiden(data, resolution=resolution, random_state=random_state)
            label = data.obs["leiden"].to_numpy()
    elif method == "kmeans":
        from sklearn.cluster import KMeans
        label = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(latent)
    elif method == "mclust":
        from rpy2.robjects import r, numpy2ri
        
        r.library("mclust")
        numpy2ri.activate()
        r_random_seed = r['set.seed']
        r_random_seed(random_state)

        rmclust = r['Mclust']
        res = rmclust(numpy2ri.numpy2rpy(latent), n_clusters, mclust_model)
        
        label = np.asarray(res[-2], dtype=int) - 1
    else:
        raise ValueError(f"Invalid Method {method}!")

    return label.astype(int)