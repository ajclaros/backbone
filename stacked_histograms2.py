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

df = pd.read_pickle("physics_5_9/all_embeds.pkl")
# plot histogram of number of paragraphs per paper


embeds = df['embedding'].tolist()
embeds = np.array(embeds)

paper_ids = df['paper_id'].tolist()
unique_paper_ids = list(set(paper_ids))
# del df
intra_doc_dists = {}
intra_doc_dist_ix = {}
paper_dist = []
cutoff = 200
neighbor_distances = {}
n_distance_bins = 30
distance_bins = np.linspace(0, 0.1, n_distance_bins).tolist()
distance_bins = np.array(distance_bins)
def cosine_distance(x, y):
    return abs(cosine(x, y))
def extract_neighbors(index):
    global unique_paper_ids, embeds, df, distance_bins, cutoff
    if index % 100 == 0:
        print(index)
    intra_doc_dist_matrix = np.zeros((cutoff, len(distance_bins)))
    paper_id = unique_paper_ids[index]
    df_t = df[df['paper_id'] == paper_id]
    embeds_paper = embeds[np.array(paper_ids) == paper_id]
    # calculate knn to extract ranked distances from each point
    knn = NearestNeighbors(n_neighbors=df_t.shape[0], metric=cosine_distance )
    knn.fit(embeds_paper)
    distances, indices = knn.kneighbors(embeds_paper)
    # count up how many neighbors are in each bin
    for paragraph_ix in range(indices.shape[0]):
        for neighbor_ix in range(2, indices.shape[1]): # skip the first one, which is the paragraph itself
            distance = distances[paragraph_ix, neighbor_ix]
            for i in range(len(distance_bins) - 1):
                if distance < distance_bins[i + 1]:
                    if neighbor_ix < cutoff:
                        intra_doc_dist_matrix[neighbor_ix, i] += 1
                    else:
                        intra_doc_dist_matrix[-1, i] += 1
                    break

    return intra_doc_dist_matrix

intra_doc_matrix = np.zeros((cutoff, len(distance_bins)))
results = []
matrices = []
print("Number of unique papers:")
print(len(unique_paper_ids))
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    for i in trange(len(unique_paper_ids)):
    # for i in trange(100):
        results.append(executor.submit(extract_neighbors, i))

    for future in concurrent.futures.as_completed(results):
        matrices.append(future.result())
for matrix in matrices:
    intra_doc_matrix += matrix
# normalize the histogram
intra_doc_matrix/= len(unique_paper_ids)
# plot the histogram
cmap = plt.get_cmap("cividis", intra_doc_matrix.shape[0])
colors = [cmap(i) for i in range(intra_doc_matrix.shape[0])]
# fig, ax = plt.subplots()
# using seaborn, we can plot histograms for each bin and the prevalence of neighbors in that bin being their
# width for the histogram value
intra_df = pd.DataFrame(intra_doc_matrix, columns=distance_bins[:])
# now we plot a histogram of each column, let the hue be the row index
sns.set()
sns.set_context("talk")
sns.set_style("white")
sns.set_palette("cividis")
# legend off
df = intra_df.T

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=intra_df.T, kde=False, ax=ax, palette=colors, bins=n_distance_bins, binrange=(0, 1), legend=False)
plt.show()
# plot but use the index as the x axis
# sns.histplot(data=intra_df.T, bins=len(distance_bins), kde=False, ax=ax, palette=colors, legend=False)
# set the xtick labels to be the bin centers
print("showing plot")
plt.xlabel("Distance")
plt.ylabel("Probability")
plt.title("Probability of Distance Belonging to Same Paper")
# plt.savefig("plots/probability_of_distance.png")
plt.show()
