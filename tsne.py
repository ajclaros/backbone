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
from utils import *
import pickle

np.random.seed(42)
filename = "rand_100_embeds.pkl"
df = pd.read_pickle(filename)
num_papers = 30
papers = df["paper_id"].unique()
papers = np.random.choice(papers, num_papers, replace=False)
df = df[df["paper_id"].isin(papers)]
df = df[df["text"].str.len() > 50]
df = df[df["text"].str.len() < 350]
df = df.reset_index(drop=True)
# apply tsno
embeds = np.array(df["embedding"].to_list())
print("Fitting tsne")
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(embeds)
print("Done fitting tsne")
df["tsne1"] = tsne_results[:, 0]
df["tsne2"] = tsne_results[:, 1]
# given the tsne results, we plot them and color by attribute
ncols = 2
attributes = [
    "year",
    "paper_id",
    "binned_len",
]
df["binned_len"] = pd.cut(df["text"].str.len(), bins=10)
metadata = pd.read_csv("metadata_00.csv")

df_copy = distance_cluster(df, 0.25)
df_copy["binned_len"] = pd.cut(df_copy["text"].str.len(), bins=350)

import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 16), ncols=1, nrows=3)
sns.scatterplot(
    hue="paper_id", data=df_copy, x="tsne1", y="tsne2", palette="viridis", ax=ax[0]
)
ax[0].title.set_text("Color by paper_id")
ax[0].legend([], frameon=False)
ax[0].set_xlabel("tsne1")
ax[0].set_ylabel("tsne2")
plt.tight_layout()
sns.scatterplot(
    hue="year", data=df_copy, x="tsne1", y="tsne2", palette="viridis", ax=ax[1]
)
ax[1].set_title("Color by year")
ax[1].legend([], frameon=False)
ax[1].set_xlabel("tsne1")
ax[1].set_ylabel("tsne2")
plt.tight_layout()
df_copy["length"] = df_copy["text"].str.len()
sns.scatterplot(
    hue="length", data=df_copy, x="tsne1", y="tsne2", palette="viridis", ax=ax[2]
)
ax[2].set_title("Color by text length")
ax[2].set_xlabel("tsne1")
ax[2].set_ylabel("tsne2")
# add viridis colorbar manually
import matplotlib.cm as cm
import matplotlib.colors as mcolors

norm = mcolors.Normalize(vmin=0, vmax=350)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax[2], label="Text Length")
plt.tight_layout()

# only show the first 10 papers in the legend
plt.legend([], frameon=False)
fig.savefig("../plots/tsne_params.png")
plt.show()

# plot histogram of text length log log scale

fig, ax = plt.subplots(figsize=(8, 4))
df["text_len"] = df["text"].str.len()
df["text_len"].hist(bins=1500, ax=ax, density=True)
ax.set_title("Text Length Histogram")
plt.tight_layout()
plt.xlim(0, 1000)
lw = 5
plt.axvline(50, color="r", label="Dataset Bounds\n50 < x < 350", ls="--", lw=lw)
plt.axvline(350, color="r", ls="--", lw=lw)
plt.legend()
plt.savefig("../plots/text_len_hist.png")
plt.show()


# all to all matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

df_copy = df.copy()
papers = df_copy["paper_id"].unique()[:7]
df_copy_t = df[df["paper_id"].isin(papers)]
metric = euclidean
dist_matrix = distance_matrix(df_copy, metric=l2_distance)
# import cosine similarity and euclidean distance
clusterer = hnsw.HNSW(metric, m=2, ef=200)
network, doc_boundaries = get_matrix2(df_copy, metric=metric)
# get the adjacency matrix
matrix = nx.adjacency_matrix(network).todense()
# normalize matrix
matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
# invert non-zero values
matrix[matrix > 0] = 1 - matrix[matrix > 0]
fig, ax = plt.subplots(figsize=(8, 8), ncols=2, nrows=1)
ax[0].imshow(dist_matrix, cmap="jet", interpolation="nearest")
ax[0].set_title("L2 Distance")
ax[0].set_xlim(0, 400)
ax[0].set_ylim(0, 400)
ax[1].imshow(matrix, cmap="jet", interpolation="nearest")
ax[1].set_title("HNSW")
ax[1].set_xlim(0, 400)
ax[1].set_ylim(0, 400)
# add horizontal color bar underneath plots
cax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
fig.colorbar(
    ax[0].imshow(dist_matrix, cmap="jet", interpolation="nearest"),
    cax=cax,
    orientation="horizontal",
)
plt.savefig("../plots/hnsw_compare.png")
plt.tight_layout()
plt.show()
