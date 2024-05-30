import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist

# use l2 distance and cosine similarity
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from joblib import parallel_backend
import seaborn as sns
import os
from tqdm import trange, tqdm
import concurrent.futures

np.random.seed(42)

df = pd.read_pickle("rand_100_embeds.pkl")
papers = df["paper_id"].unique()
# Randomly sample 100 papers from years 2005-2009
paper_sample = []
years = [2005, 2006, 2007, 2008, 2009]
num_papers = 50
for year in years:
    papers = df.query("year == @year")["paper_id"].unique()
    paper_sample.extend(np.random.choice(papers, num_papers, replace=False))
df = df.query("paper_id in @paper_sample")
paper_groups = df.groupby("paper_id")
papers = df["paper_id"].unique()
in_paper_distances = {}
for paper_id, group in tqdm(paper_groups):
    distances = pdist(group["embedding"].tolist(), cosine)
    if len(distances) <= 1:
        df = df.query("paper_id != @paper_id")
        continue
    in_paper_distances[paper_id] = distances
# identify the greatest and smallest distances of paragragraphs in whole dataset
min_distance = np.inf
max_distance = -np.inf
for distances in in_paper_distances.values():
    min_distance = min(min_distance, min(distances))
    max_distance = max(max_distance, max(distances))
# bin the minimum and maximum distances
n_bins = 30
bins = np.linspace(min_distance, max_distance, n_bins)
# given the distance, find the probability of a paragraph being in the same paper
probabilities = np.zeros(n_bins - 1)
for distances in tqdm(in_paper_distances.values()):
    for distance in distances:
        for i in range(1, n_bins):
            if bins[i - 1] <= distance < bins[i]:
                probabilities[i - 1] += 1
                break

probabilities /= len(df)
plt.bar(bins[:-1], probabilities, width=bins[1] - bins[0])
plt.xlabel("Distance")
plt.ylabel("Probability")
plt.title("Probability of Distance Belonging to Same Paper")
plt.savefig("../plots/probability_of_distance.png")
plt.show()

from joblib import Parallel, delayed

np.random.seed(42)

min_distance = np.inf
max_distance = -np.inf
n_bins = 30
bins = np.linspace(min_distance, max_distance, n_bins)
probabilities = np.zeros(n_bins - 1)


def process_paper(paper_id, group, in_paper_distances):
    distances = pdist(group["embedding"].tolist(), cosine)
    if len(distances) <= 1:
        return None

    neigh = NearestNeighbors(n_neighbors=15, metric="cosine")
    embeddings = group["embedding"].tolist()
    neigh.fit(embeddings)
    paper_probabilities = np.zeros(n_bins - 1)
    for i in range(len(group)):
        _, indices = neigh.kneighbors([distances[i]])
        for j, idx in enumerate(indices[0]):
            if idx == i:
                continue
            for k in range(1, n_bins):
                if bins[k - 1] <= distances[idx] < bins[k]:
                    paper_probabilities[k - 1] += 1
                    break
    return paper_probabilities


num_threads = 4
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(
        tqdm(
            executor.map(
                lambda x: process_paper(x[0], x[1], in_paper_distances),
                in_paper_distances.items(),
            ),
            total=len(in_paper_distances),
        )
    )

for result in results:
    if result is None:
        continue
    probabilities += result
