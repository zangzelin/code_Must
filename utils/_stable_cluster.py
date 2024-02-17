from ._cluster import cluster
import numpy as np

def stable_cluster(latent: np.ndarray,
                    method: str = "louvain",
                    attempt_num: int = 3,
                    target_metric: str = "silhouette",
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

    latent_original = latent
    if pca_dim is not None:
        from sklearn.decomposition import PCA
        latent = PCA(n_components=pca_dim, random_state=0).fit_transform(latent)

    metrics = []
    labels = []
    for seed in range(attempt_num):
        label = cluster(latent=latent,
                        method=method,
                        random_state=seed,
                        n_neighbors=n_neighbors,
                        n_clusters=n_clusters,
                        resolution=0.5,
                        pca_dim=None,
                        mclust_model='EEE')
        if target_metric == "silhouette":
            from sklearn.metrics import silhouette_score
            metrics.append(silhouette_score(latent_original, label))

        labels.append(label)

    metrics = np.array(metrics)
    if target_metric == "silhouette":
        return labels[np.argmax(metrics)].astype(int)
    else:
        return None