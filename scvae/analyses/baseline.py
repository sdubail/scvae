from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import igraph as ig
    import leidenalg

    LEIDEN_AVAILABLE = True
except ImportError:
    print(
        "Leiden algorithm not available. Install with: pip install leidenalg python-igraph"
    )
    LEIDEN_AVAILABLE = False


# Set global matplotlib parameters
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 15
plt.rcParams["lines.linewidth"] = 3


# Define the custom color palette
CLASS_PALETTE = {
    0: [0, 0, 0],
    1: [1, 0, 0],
    2: [0, 1, 0],
    3: [0, 0, 1],
    4: [1, 1, 0],
    5: [1, 0, 1],
    6: [0, 1, 1],
    7: [1, 1, 1],
    8: [0.5, 0, 0],
    9: [0, 0.5, 0],
    10: [0, 0.5, 0.5],
    11: [0.5, 0.5, 0.5],
    12: [0.5, 0.5, 0],
}


def get_colors_from_labels(labels):
    """Convert labels to RGB colors using the custom palette."""
    return np.array([CLASS_PALETTE[label % len(CLASS_PALETTE)] for label in labels])


def create_knn_graph(X, n_neighbors=15):
    # Create KNN graph using UMAP's internal function
    from umap.umap_ import nearest_neighbors

    knn = nearest_neighbors(
        X,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds={},
        angular=False,
        random_state=42,
    )

    n_samples = X.shape[0]
    rows = np.repeat(np.arange(n_samples), n_neighbors)
    cols = knn[0].ravel()
    data = np.ones_like(cols)

    graph = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    return graph


def leiden_clustering(X, resolution=1.0):
    # Create KNN graph
    knn_graph = create_knn_graph(X)

    # Convert to igraph
    sources, targets = knn_graph.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))

    # Create graph with the correct number of vertices
    g = ig.Graph(n=X.shape[0], edges=edges, directed=False)

    # Run Leiden algorithm
    partitions = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution
    )

    return np.array(partitions.membership)


def run_clustering_baseline(data_directory, evaluation_set):
    X = evaluation_set.values.toarray()
    print("Data shape", X.shape)
    true_labels = evaluation_set.labels

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)

    print("Performing PCA...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    print("Performing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap = reducer.fit_transform(X_pca)

    results = {}

    # K-means
    print("Running K-means...")
    n_clusters = len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    results["K-means"] = {
        "labels": kmeans_labels,
        "ari": adjusted_rand_score(true_labels, kmeans_labels),
        "silhouette": silhouette_score(X_scaled, kmeans_labels, sample_size=2000),
    }

    # DBSCAN
    print("Running DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca)

    if len(np.unique(dbscan_labels)) > 1:
        try:
            results["DBSCAN"] = {
                "labels": dbscan_labels,
                "ari": adjusted_rand_score(true_labels, dbscan_labels),
                "silhouette": silhouette_score(X_pca, dbscan_labels, sample_size=2000),
            }
        except Exception as e:
            print("Error - DBSCAN failed to find meaningful clusters")
    else:
        print("DBSCAN failed to find meaningful clusters")

    # Leiden clustering
    if LEIDEN_AVAILABLE:
        print("Running Leiden clustering...")
        try:
            leiden_labels = leiden_clustering(X_pca)
            results["Leiden"] = {
                "labels": leiden_labels,
                "ari": adjusted_rand_score(true_labels, leiden_labels),
                "silhouette": silhouette_score(X_pca, leiden_labels, sample_size=2000),
            }
        except Exception as e:
            print(f"Leiden clustering failed: {e}")

    # Print results
    print("\nClustering Results:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"Adjusted Rand Index: {result['ari']:.4f}")
        print(f"Silhouette Score: {result['silhouette']:.4f}")
        print(f"Number of clusters found: {len(np.unique(result['labels']))}")

    # Visualize results
    n_methods = len(results) + 1
    plt.figure(figsize=(5 * n_methods, 10))

    # First row: PCA projections
    plt.subplot(2, n_methods, 1)
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=get_colors_from_labels(true_labels.astype(int)),
        s=5,
        linewidth=3,
    )
    plt.title("True Labels (PCA)", fontsize=15, fontfamily="Helvetica")

    for i, (method, result) in enumerate(results.items(), 1):
        plt.subplot(2, n_methods, i + 1)
        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=get_colors_from_labels(result["labels"]),
            s=5,
            linewidth=3,
        )
        plt.title(f"{method} (PCA)", fontsize=15, fontfamily="Helvetica")

    # Second row: UMAP projections
    plt.subplot(2, n_methods, n_methods + 1)
    plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=get_colors_from_labels(true_labels.astype(int)),
        s=5,
        linewidth=3,
    )
    plt.title("True Labels (UMAP)", fontsize=15, fontfamily="Helvetica")

    for i, (method, result) in enumerate(results.items(), 1):
        plt.subplot(2, n_methods, n_methods + i + 1)
        plt.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=get_colors_from_labels(result["labels"]),
            s=5,
            linewidth=3,
        )
        plt.title(f"{method} (UMAP)", fontsize=15, fontfamily="Helvetica")

    plt.tight_layout()
    plt.savefig(Path(data_directory) / "baseline_clustering.png", dpi=300)

    # Save results to file
    filename = "baseline_clustering_results.txt"
    with open(Path(data_directory) / filename, "w") as f:
        f.write("Clustering Results:\n")
        f.write("=" * 50 + "\n")

        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"Adjusted Rand Index: {result['ari']:.4f}\n")
            f.write(f"Silhouette Score: {result['silhouette']:.4f}\n")
            f.write(f"Number of clusters found: {len(np.unique(result['labels']))}\n")

        f.write("\nDataset Details:\n")
        f.write(f"Data shape: {evaluation_set.values.shape}\n")
        f.write(f"Number of true labels: {len(np.unique(evaluation_set.labels))}\n")
        f.write(
            f"Sparsity: {evaluation_set.values.nnz / (evaluation_set.values.shape[0] * evaluation_set.values.shape[1]):.4f}\n"
        )

    return results
