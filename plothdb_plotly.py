# replace with UMAP
# use 128 dimensions
# add categories from metadata as extra parameters to color on
# sampling on categories (metadata)
# increase number of papers
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils import distance_cluster
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
import textwrap
import plotly.express as px
import plotly.graph_objects as go
import sys
import importlib
import seaborn as sns

sns.set()


np.random.seed(42)
sns.set_context("poster")
sns.set_style("white")
sns.set_color_codes()

importlib.reload(sys.modules["utils"])
df = pd.read_pickle("rand_100_embeds.pkl")
df = df[df["year"] == 2009]
num_papers = 30
papers = df["paper_id"].unique()
papers = np.random.choice(papers, num_papers, replace=False)
df = df[df["paper_id"].isin(papers)]
df = df[df["text"].str.len() < 600]
df = df[df["text"].str.len() > 50]

tsne_reduce = TSNE(
    n_components=127, perplexity=40, n_iter=300, early_exaggeration=12, method="exact"
)
embeds = np.array(df["embedding"].values.tolist())
tsne_reduce_results = tsne_reduce.fit_transform(embeds).astype(np.float64)


hdb_params = {
    "algorithm": "generic",
    "min_cluster_size": 5,
    "min_samples": 10,
    "gen_min_span_tree": True,
    "approx_min_span_tree": False,
    "metric": "cosine",
}
clusterer = HDBSCAN(**hdb_params)
embeds = np.array(embeds, dtype=np.float64).reshape(-1, 768)
clusterer.fit(tsne_reduce_results)
labels = clusterer.labels_
tsne = TSNE(n_components=2, perplexity=40, n_iter=300, early_exaggeration=12)
tsne_results = tsne.fit_transform(embeds)
df["tsne1"] = tsne_results[:, 0]
df["tsne2"] = tsne_results[:, 1]

plot_kwds = {"alpha": 0.5, "s": 80, "linewidths": 0, "s": 40}
fig, ax = plt.subplots(figsize=(10, 10))
df["cluster"] = labels

ax.scatter(df["tsne1"], df["tsne2"], c=df["cluster"], cmap="viridis", **plot_kwds)
ax.set_title(f"Number of Clusters: {len(set(labels))}")
plt.show()
