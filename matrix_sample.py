from utils import *
import os
import jsonlines
import importlib
import sys
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from flexible_clustering import hnsw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from joblib import parallel_backend
import pickle
import concurrent.futures

importlib.reload(sys.modules["utils"])


years = [5, 6, 7, 8, 9]
num_docs = 1
columns = [
    "paper_id",
    "section",
    "sec_number",
    "text",
    "embedding",
]
# df, filename = process_files(
#     years, num_docs, columns, domain="Physics", save=True, process_by="year"
# )
filename = "aggregated_physics_5_9/rand_100_embeds.pkl"
df = pd.read_pickle(filename)

# df_copy = drop_if_within_distance(df, 0.01)
# del df
# df['text'].str.len().value_counts()
# bins = df['text'].str.len().value_counts(bins=10)
# plt.hist(df_copy["text"].str.len(), bins=10)
# plt.show()
# matrix, doc_boundaries = get_matrix(df)
# num_docs = 10
# fig_row = num_docs // 2
# fig_col = 2
# fig, ax = plt.subplots(fig_row, fig_col, figsize=(20, 20))
# for i in range(num_docs):
#     test = df[df["paper_id"] == df["paper_id"].unique()[i]]
#     # drop rows with text length less than 1000
#     # test = test[test['text'].str.len() > 10]
#     # test = test[test['text'].str.len() < 1000]
#     # & test['text'].str.len() < 1000]]
#     print(test["paper_id"].unique())
#     matrix, doc_ids = get_matrix(test)
#     matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
#     matrix[matrix > 0] = 1 - matrix[matrix > 0]
#     ax[i // fig_col, i % fig_col].imshow(matrix, cmap="jet", interpolation="nearest")
# df_copy = df[df["text"].str.len() > 1000]
# df_copy = df[(df["text"].str.len() > 100) & (df["text"].str.len() < 350)]
df_copy = df.copy()
matrix, doc_boundaries = get_matrix(df_copy, shuffle=False, level=-1)


# doc_ids = df["paper_id"].unique()
matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
matrix[matrix > 0] = 1 - matrix[matrix > 0]
matrix2, doc_ids2 = get_matrix(df_copy, shuffle=True)
matrix2 = (matrix2 - matrix2.min()) / (matrix2.max() - matrix2.min())
matrix2[matrix2 > 0] = 1 - matrix2[matrix2 > 0]
fig1, ax1 = plt.subplots(figsize=(10, 10), ncols=3)
opacity = 0.5
ax1[1].imshow(matrix, cmap="jet", interpolation="nearest")
ax1[2].imshow(matrix2, cmap="jet", interpolation="nearest")
lw = 0.5
for idx in doc_boundaries:
    ax1[1].axhline(idx, color="black", linewidth=lw, alpha=opacity, linestyle="--")
    ax1[1].axvline(idx, color="black", linewidth=lw, alpha=opacity, linestyle="--")
    ax1[2].axhline(idx, color="k", linewidth=lw, alpha=opacity, linestyle="--")
    ax1[2].axvline(idx, color="k", linewidth=lw, alpha=opacity, linestyle="--")
plt.show()
# ax1[0].axhline(0, color="black", linewidth=2)


# corr matrix using l2 distance

corr_matrix = np.zeros(df_copy.shape[0] ** 2).reshape(
    (df_copy.shape[0], df_copy.shape[0])
)
# num_workers = 16
# results = []
for i in range(df_copy.shape[0]):
    end = "\n" * (i % 50 == 0)
    row = i // 50
    row = str(row) + " :\n"
    row *= i % 50 == 0
    print(f"{row}{i} ", end=end)
    for j in range(df_copy.shape[0]):
        if i == j:
            corr_matrix[i, j] = 0
        else:
            corr_matrix[i, j] = l2_distance(
                df_copy.iloc[i]["embedding"], df_copy.iloc[j]["embedding"]
            )

corr_matrix = (corr_matrix - corr_matrix.min()) / (
    corr_matrix.max() - corr_matrix.min()
)
corr_matrix[corr_matrix > 0] = 1 - corr_matrix[corr_matrix > 0]
# for idx in doc_boundaries:
#     corr_matrix[idx, :] = 0
#     corr_matrix[:, idx] = 0
ax1[0].imshow(corr_matrix, cmap="jet", interpolation="nearest")
# set title
ax1[0].set_title("L2 Distance")
ax1[1].set_title("HNSW, no shuffle")
ax1[2].set_title("HNSW, shuffle")
plt.savefig("../notes/hnsw_shuffle_compare.png")


def in_doc(doc_i, doc_j, df):
    distance = l2_distance(df.iloc[doc_i]["embedding"], df.iloc[doc_j]["embedding"])
    return (df.iloc[doc_i]["paper_id"] == df.iloc[doc_j]["paper_id"], doc_j, distance)


def in_doc_list(doc_i, doc_list, df):
    return [in_doc(doc_i, doc_j, df) for doc_j in doc_list]


def knn_list(input_vec, df, k=5):
    distances, indices = find_k_nearest_neighbors(input_vec, df["embedding"], k)
    return in_doc_list(doc_i, indices, df)


# knn_list(df.iloc[0]["embedding"], df_copy, k=10)
# df_copy = df[df["text"].str.len() > 1000]
df = df[df["text"].str.len() > 50]
df = df[df["text"].str.len() < 350]
df.reset_index(drop=True, inplace=True)
neighbors = 8
knn = NearestNeighbors(n_neighbors=neighbors, metric="l2", n_jobs=-1)
print("Fitting KNN")
with parallel_backend("multiprocessing"):
    knn.fit(df["embedding"].to_list())

print("Finding KNN")
with parallel_backend("multiprocessing"):
    distances, indices = knn.kneighbors(df["embedding"].to_list())

knnlist = [in_doc_list(i, indices[i], df) for i in range(df.shape[0])]
# knnlist is organized as:
# knnlist[doc_i] = [(in_doc, distance), ...], where in_doc is a boolean if doc_i and doc_j are in the same document
# and distance is the l2 distance between doc_i and doc_j
# need to plot the probability of in_doc for each k as a function of distance
fig, ax = plt.subplots()
# neigh_df = pd.DataFrame(columns=["doc_i", "doc_j", "in_doc", "distance"])
neigh_df2 = pd.DataFrame(columns=["doc_i", "doc_j", "in_doc", "distance"])
neigh_df2["doc_i"] = neigh_df.index // neighbors
neigh_df2["doc_j"] = neigh_df.index % neighbors
neigh_df2["in_doc"] = neigh_df["in_doc"]
neigh_df2["distance"] = neigh_df["distance"]
neigh_df.reset_index(drop=True, inplace=True)
# drop all rows where doc_i and doc_j are in the same document
neigh_df2 = neigh_df2[neigh_df2["distance"] != 0.0].reset_index(drop=True)
# save knnlist as pickle
# with open("./aggregated_physics_5_9/knnlist_500_sample.pkl", "wb") as f:
#     pickle.dump(knnlist, f)

# plot histogram of text length
# plot probability of in_doc as a function of k
probs = []
for k in range(1, neighbors + 1):
    probs.append(neigh_df2[neigh_df2["doc_j"] < k]["in_doc"].mean())

fig, ax = plt.subplots()
ax.plot(range(neighbors), probs, marker="o")
ax.set_xlabel("k")
ax.set_ylabel("Probability of in_doc")
ax.set_title("P(in_doc|k)")
plt.savefig("../notes/knn_in_doc.png")
plt.show()

# plot probability of in_doc as a function of distance
fig, ax = plt.subplots()
neigh_df2["distance"] = neigh_df2["distance"].round(2)
bins = 20
max_distance = neigh_df2["distance"].max()
x = np.linspace(0, max_distance, bins)
y = []
for i in range(bins - 1):
    # grab all rows where distance is between bins[i] and bins[i+1]
    # calculate the mean of all boolean values in in_doc (True = 1, False = 0)
    y.append(
        neigh_df2[(neigh_df2["distance"] >= x[i]) & (neigh_df2["distance"] < x[i + 1])][
            "in_doc"
        ].mean()
    )
ax.plot(x[:-1], y, marker="o")
ax.set_xlabel("Distance")
ax.set_ylabel("Probability of in_doc")
ax.set_title("P(in_doc|distance)")
plt.savefig("../notes/knn_in_doc_distance.png")
plt.show()

# plot distribution of distances of the first neighbor
fig, ax = plt.subplots()
# neigh_df2[(neigh_df2["doc_j"] == 1)]["distance"]
hist, bins = np.histogram(neigh_df2[(neigh_df2["doc_j"] == 1)]["distance"], bins=50)
ax.bar(bins[:-1], hist / hist.sum(), width=0.35)
ax.set_xlabel("Distance")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of distances of the first neighbor")
plt.savefig("../notes/knn_distance_distribution.png")
plt.show()

# stacked bar plot
# fig, ax = plt.subplots()
max_distance = neigh_df2["distance"].max()
bins = np.linspace(0, max_distance, 20)
neigh_df2["distance_bin"] = pd.cut(neigh_df2["distance"], bins)
neigh

neig
