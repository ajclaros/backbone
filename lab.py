import plotly.express as px
import plotly.graph_objects as go
import textwrap
import os
import jsonlines
import sys
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from joblib import parallel_backend
import pickle
import concurrent.futures
import argparse
import tqdm
from utils import *
from scipy.spatial.distance import euclidean


filename = "rand_100_embeds.pkl"
df = pd.read_pickle(filename)
# filter out empty text
df = df[df["text"].str.len() > 0]
df = df[df["text"].str.len() < 350]
df = df[df["text"].str.len() > 50]
df["year"] = df["year"].apply(lambda x: int(x))
df["month"] = df["month"].astype(int)
df["date"] = df.apply(lambda x: datetime(x["year"], x["month"], 1).timestamp(), axis=1)
X = np.array(df["embedding"].tolist())
tsne = TSNE(n_components=2, perplexity=40, n_iter=300, early_exaggeration=40)
tsne_results = tsne.fit_transform(X)
df["tsne1"] = tsne_results[:, 0]
df["tsne2"] = tsne_results[:, 1]
df_copy = distance_cluster(df, 2.5)
network, doc_boundaries = get_matrix2(df_copy, metric=euclidean)

embeds = df_copy["embedding"].values.tolist()
cluster, indices = add_embeds(embeds, cluster, shuffle=shuffle)
# drop_nan
df_copy = df_copy.dropna()
df_copy["text_wrap"] = df_copy["text"].apply(lambda t: "<br>".join(textwrap.wrap(t)))
px.scatter(
    df_copy,
    x="tsne1",
    y="tsne2",
    color="cluster_size",
    hover_data=["text_wrap"],
    render_mode="svg",
)
