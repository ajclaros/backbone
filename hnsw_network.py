import os
import jsonlines
import sys
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from joblib import parallel_backend
import pickle
import concurrent.futures
import argparse
import tqdm
from utils import *
from scipy.spatial.distance import euclidean
from sklearn.cluster import HDBSCAN

parser = argparse.ArgumentParser()
parser.add_argument("-p", type=int, default=10, help="Number of processes")
args = parser.parse_args()
nj = args.p

years = [5, 6, 7, 8, 9]
columns = [
    "paper_id",
    "section",
    "sec_number",
    "text",
    "embedding",
]
# df, filename = process_files(years, -1, columns, domain="Physics", save=True)
filename = "rand_100_embeds.pkl"
df = pd.read_pickle(filename)
df = df[df["text"].str.len() > 0]


df["text"].str.len().value_counts().sort_index()
# cumulative histogram of text length
fig, ax = plt.subplots()
df["text"].str.len().plot.hist(
    bins=5000, cumulative=False, density=True, ax=ax, label="text length"
)
ax.set_xlim(0, 1500)
ax.axvline(50, color="k", label="dataset bounds\n50 < x < 350")
ax.axvline(350, color="k")
ax.set_title("Text Length Histogram")
plt.legend()
plt.savefig("../notes/histogram.png")
plt.show()

# log log histogram of text length
df["text"].str.len().plot.hist(bins=100, loglog=True)
plt.show()
df_copy = df.copy()
network, doc_boundaries = get_matrix2(df.iloc[:2000], metric=euclidean)
matrix = nx.adjacency_matrix(network).todense()
# normalize matrix

matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
# invert non-zero values
matrix[matrix > 0] = 1 - matrix[matrix > 0]
plt.imshow(matrix)
plt.show()
print("Calculating distance matrix")


# papers per year
papers = df.groupby("year")[["paper_id"]].nunique()

# remove rows with text length less than 50 and greater than 350
# get distance matrix
dist_matrix = distance_matrix(matrix)
df_copy = df[df["text"].str.len() > 50]
df_copy = df_copy[df_copy["text"].str.len() < 350]
df_copy = df_copy.reset_index(drop=True)
df = df_copy.copy()
del df_copy
# get nearest neighbors
neighbors = 9
knn = NearestNeighbors(n_neighbors=neighbors, metric="precomputed")
print("Fitting nearest neighbors")
with parallel_backend("threading", n_jobs=nj):
    knn.fit(dist_matrix)
print("Finding nearest neighbors")
with parallel_backend("threading", n_jobs=nj):
    distances, indices = knn.kneighbors(dist_matrix)

# get in-doc neighbors
knnlist = [in_doc_list(i, indices[i], df) for i in range(df.shape[0])]
neigh_df = pd.Dataframe(columns=["doc_i", "doc_j", "distance", "in_doc"])
for i, knn in enumerate(knnlist):
    for j, distance in enumerate(knn):
        neigh_df = neigh_df.append(
            {
                "doc_i": i,
                "doc_j": j,
                "distance": distance,
                "in_doc": df["paper_id"][i] == df["paper_id"][j],
            },
            ignore_index=True,
        )
# calculate probabilities of in-doc neighbors
probs = []
for k in range(1, neighbors + 1):
    probs.append(neigh_df[neigh_df["distance"] < k].shape[0] / neigh_df.shape[0])
