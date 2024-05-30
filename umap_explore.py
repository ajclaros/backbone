import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import pandas as pd
import umap
import os
import sys
import importlib
import plotly.express as px
import plotly.graph_objects as go
import textwrap

np.random.seed(42)

df = pd.read_pickle("rand_100_embeds.pkl")
df = df[df["year"] == 2009]
metadata = pd.read_csv("metadata_09.csv")
metadata["paper_id"] = metadata["paper_id"].astype(str)
metadata["paper_id"] = "0" + metadata["paper_id"]
metadata = metadata[metadata["discipline"] == "Physics"]
num_papers = 100
papers = df["paper_id"].unique()
papers = np.random.choice(papers, num_papers, replace=False)
df["words_per_paragraph"] = df["text"].apply(lambda t: len(t.split())).values
df = df.query("words_per_paragraph > 50 and words_per_paragraph < 600")
df = df.query("paper_id in @papers")
# df = df.query("text.str.len() > 1000")
df = df.query("text.str.len() > 500")
data = df["embedding"].tolist()
fit = umap.UMAP()
u = fit.fit_transform(data).astype(np.float64)
colors = df["paper_id"].astype("category").cat.codes
df["umap1"] = u[:, 0]
df["umap2"] = u[:, 1]
df["text_wrap"] = df["text"].apply(lambda t: "<br>".join(textwrap.wrap(t, width=50)))
df["text_len"] = df["text"].apply(len)
paper_id = df["paper_id"].iloc[0]
categories_meta = metadata[["paper_id", "categories_meta"]]
# for every row, map categories_meta to the paper_id
df["categories_meta"] = df["paper_id"].map(
    categories_meta.set_index("paper_id")["categories_meta"]
)

# apply hdb to the umap embeddings
hdb_params = {
    "algorithm": "generic",
    "min_cluster_size": 5,
    "min_samples": 10,
    "gen_min_span_tree": True,
    "approx_min_span_tree": False,
    "metric": "cosine",
}
clusterer = hdbscan.HDBSCAN(**hdb_params)
embeds = np.array(data, dtype=np.float64).reshape(-1, 768)
clusterer.fit(u)
labels = clusterer.labels_
df["cluster"] = labels
df["cluster"] = df["cluster"].astype(float)
fig = px.scatter(
    df,
    x="umap1",
    y="umap2",
    color="cluster",
    hover_data=["categories_meta", "paper_id", "words_per_paragraph"],
    labels={"color": "paper_id"},
    title="UMAP of Embeddings",
    render_mode="svg",
)
fig.update_traces(marker=dict(size=20))
fig.show()

# find the paper with the most paragraphs
