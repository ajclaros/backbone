"""
This script is used to take the distances for all paragraphs and bins them into a histogram.
The histogram is then used to calculate the probability of a binned distance belonging to the same paper that the paragraph is from.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

# use l2 distance and cosine similarity
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from joblib import parallel_backend
import seaborn as sns
import os
from tqdm import trange
import concurrent.futures

np.random.seed(42)
# Plotting the histogram of distances
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
dist_matrix = distance_matrix(df, cosine)
# bin the distances into a histogram
hist, bins = np.histogram(dist_matrix, bins=100)
# plotting hist and bins
dist_matrix = np.unique(dist_matrix)
n_bins = 30
plt.hist(dist_matrix, bins=n_bins)
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title(f"Histogram of Distances |{n_bins} bins")
plt.savefig("../plots/histogram_of_distances.png")
plt.show()
# find the probability of a distance belonging to the same paper

# for each bin, find the number of distances that belong to the same paper
# divide by the total number of distances in that bin

# plotting stacked histogram of distances
# where the height of the bars is the proportion of k-nearest neighbors that belong to the same paper
from sklearn.neighbors import NearestNeighbors

k = 20
# data = data.reshape(-1, 768)
nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(data)
distances, indices = nn.kneighbors(data)
distances = distances[:, 1:]
indices = indices[:, 1:]
flat_indices = indices.flatten()
flat_distances = distances.flatten()
data_k = {"distance": flat_distances, "neighbor": np.tile(np.arange(1, k), len(df))}
df_plot = pd.DataFrame(data_k)
df_plot["bin"] = pd.cut(df_plot["distance"], bins=n_bins)
bin_counts = df_plot.groupby(["bin", "neighbor"]).size().unstack(fill_value=0)
bin_percentages = bin_counts.div(bin_counts.sum(axis=1), axis=0)

sns.histplot(
    df_plot,
    x="distance",
    bins=n_bins,
    hue="neighbor",
    multiple="stack",
    palette="viridis",
)
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title(f"Stacked Histogram of Distances |{n_bins} bins")
plt.savefig("../plots/stacked_histogram_of_distances.png")
plt.show()
