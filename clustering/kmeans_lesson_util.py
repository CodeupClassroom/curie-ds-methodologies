import math
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import minmax_scale


def distance(p, q):
    return math.sqrt((p.x1 - q.x1) ** 2 + (p.x2 - q.x2) ** 2)


def find_cluster(centroids: pd.DataFrame, row: pd.Series):
    distances = centroids.apply(lambda center: distance(center, row), axis=1)
    return distances.idxmin()


np.random.seed(73)
X, clusters = make_blobs(cluster_std=1.5)

df = pd.DataFrame(minmax_scale(X), columns=["x1", "x2"])
centroids = pd.DataFrame(np.random.rand(3, 2), columns=["x1", "x2"])

# For the sake of easier demonstration, these functions modify and refer to
# global variables.
# For a more practical data pipeline, we'd probably want to explicitly pass in
# dataframes and return transformed dataframes, as opposed to mutating and
# relying on global variables.
def assign_clusters():
    global df
    df = df.assign(
        cluster=lambda df: df.apply(partial(find_cluster, centroids), axis=1)
    )


def update_centroids():
    global centroids
    centroids = df.groupby("cluster").mean()

def viz_initial(include_centroids=False):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(df.x1, df.x2)
    if include_centroids:
        centroids.plot.scatter(
            x="x1", y="x2", s=600, c="red", label="centroids", marker='x', ax=ax
        )


def viz():
    fig, ax = plt.subplots(figsize=(15, 8))
    for cluster, subset in df.groupby("cluster"):
        ax.scatter(subset.x1, subset.x2, label=f"cluster {cluster}")
    centroids.plot.scatter(
        x="x1", y="x2", s=600, c="red", label="centroids", marker='x', ax=ax
    )
