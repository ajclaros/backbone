import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from utils import *
from tqdm import tqdm


df = pd.read_pickle("rand_100_embeds.pkl")
df = df[df['text'].str.len() > 100]
df = df[df['text'].str.len() < 1000]
df = df.reset_index(drop=True)
network, doc_boundaries = get_matrix2(df, metric=euclidean)


matrix = nx.adjacency_matrix(network).todense()
matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
matrix[matrix > 0] = 1 - matrix[matrix > 0]
plt.imshow(matrix, cmap="jet", interpolation="nearest")
plt.show()

# find neighbors of each node in the network. order by distance. if part of the same document it is true, else false


def bfs_limited(G, source, depth_limit, max_nodes=10):
    """
    Return the set of nodes in the graph `G` that are within `depth_limit` steps
    of the source node.
    """
    visited = set()
    current_level = [source]
    while current_level:
        next_level = []
        for node in current_level:
            if node not in visited:
                visited.add(node)
                next_level.extend(
                    [
                        neighbor
                        for neighbor in G.neighbors(node)
                        if neighbor not in visited
                    ]
                )
        if not next_level:
            break
        current_level = next_level
        if len(visited) >= depth_limit:
            break
    distances = [
        euclidean(df.iloc[source]["embedding"], df.iloc[node]["embedding"])
        for node in visited
    ]
    visited, distances = zip(*sorted(zip(visited, distances), key=lambda x: x[1]))
    return visited[:max_nodes], distances[:max_nodes]


def return_distance(df, i, j):
    """
    Given two indices i, j, return the distance between them.
    """
    return euclidean(df.iloc[i]["embedding"], df.iloc[j]["embedding"])


def same_doc(df, i, j):
    """
    Given two indices i, j, return True if they are part of the same document.
    """
    doc_i = df.iloc[i]["paper_id"]
    doc_j = df.iloc[j]["paper_id"]
    return doc_i == doc_j


def extract_weight_from_graph(G, i, j):
    return G[i][j]["weight"]


for node in tqdm(network.nodes):
    neighbors, distances = bfs_limited(network, node, 10, max_nodes=6)
    network.nodes[node]["neighbors"] = neighbors
    network.nodes[node]["distances"] = distances
    network.nodes[node]["same_doc"] = [
        same_doc(df, node, neighbor) for neighbor in neighbors
    ]

in_doc_arr = []
for node in tqdm(network.nodes):
    in_doc_arr.append(network.nodes[node]["same_doc"])
in_doc_arr = np.array(in_doc_arr)
in_doc_df = pd.DataFrame(in_doc_arr, columns=[f"k_{i}" for i in range(6)])
in_doc_df.drop(columns=["k_0", 'k_5'], inplace=True)

# plot bar of mean in_doc_arr
in_doc_df.mean().plot(kind="bar")

in_doc_knn.drop(columns=["k_0"], inplace=True)
in_doc_knn.mean().plot(kind="bar")
plt.title("Probability of in_doc for k=1 to k=4| HNSW")
plt.xlabel("k")
plt.ylabel("Probability of in_doc")
plt.savefig("../notes/in_doc_prob_knn.png")
plt.show()

in_doc_knn = pd.DataFrame(r, columns=[f"k_{i}" for i in range(5)])
# histogram of text length

df["text"].str.len().hist(bins=1000); plt.title("Histogram of text length"); plt.xlabel("Text length"); plt.ylabel("Frequency")
plt.xlim(0, 1500)
plt.axvline(0, color='k', label="dataset bounds\n 0<x<1000"
plt.show()
