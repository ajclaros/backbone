import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils import distance_cluster
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import textwrap
import plotly.express as px
import plotly.graph_objects as go
import sys
import importlib
import seaborn as sns

np.random.seed(42)
sns.set_context("poster")
sns.set_style("white")
sns.set_color_codes()
plot_kwds = {"alpha": 0.5, "s": 80, "linewidths": 0}

importlib.reload(sys.modules["utils"])

t0 = time()
df = pd.read_pickle("rand_100_embeds.pkl")
# randomly sample 30 papers
num_papers = 30
papers = df["paper_id"].unique()
papers = np.random.choice(papers, num_papers, replace=False)
df = df[df["paper_id"].isin(papers)]
df = df[df["text"].str.len() < 600]
df = df[df["text"].str.len() > 50]

tsne = TSNE(n_components=2, perplexity=40, n_iter=300, early_exaggeration=12)
embeds = np.array(df["embedding"].values.tolist())
tsne_results = tsne.fit_transform(embeds)
df["tsne1"] = tsne_results[:, 0]
df["tsne2"] = tsne_results[:, 1]

hdb_params = {
    "algorithm": "generic",
    "min_cluster_size": 5,
    "min_samples": 10,
    "gen_min_span_tree": True,
    "approx_min_span_tree": False,
    "metric": "cosine",
}
clusterer = HDBSCAN(**hdb_params)
embeds = np.array(embeds).reshape(-1, 768)
tsne_arr = np.array(tsne_results)
# reshape the embeddings to n_samples x n_features

clusterer.fit(tsne_arr)
labels = clusterer.labels_
# clusterer.minimum_spanning_tree_.plot(
#     edge_cmap="viridis", edge_alpha=0.6, node_size=80, edge_linewidth=2
# )

df["cluster"] = labels
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df["tsne1"], df["tsne2"], c=df["cluster"], cmap="viridis", s=10)
ax.title.set_text(f"Number of clusters: {len(set(labels))}")
plt.show()
