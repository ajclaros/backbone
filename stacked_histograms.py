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
n_distance_bins = 50
distance_bins = np.linspace(0, 1, n_distance_bins).tolist()
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for i in trange(len(unique_paper_ids)):
        paper_id = unique_paper_ids[i]
        df_t = df[df['paper_id'] == paper_id]
        embeds_paper = embeds[np.array(paper_ids) == paper_id]
        # distances = pdist(embeds_paper, metric='cosine')
        distances = []
        for pgh_i in range(len(embeds_paper)):
            for pgh_j in range(pgh_i + 1, len(embeds_paper)):
                dist = cosine(embeds_paper[pgh_i], embeds_paper[pgh_j])
                distances.append((pgh_i, pgh_j, dist))
                if dist<distance_bins[1]:
                    text1 = df_t.iloc[pgh_i]['text']
                    text2 = df_t.iloc[pgh_j]['text']
                    if paper_id not in paper_dist:
                        paper_dist.append((paper_id, pgh_i, pgh_j, dist, text1, text2))
        dists = [dist[2] for dist in distances]

        dists = sorted(distances, reverse=True)
        intra_doc_dists[paper_id] = dists
        # following line wrong
        # following line correct
        intra_doc_dist_ix[paper_id] = [j if j < cutoff else cutoff for j in range(len(dists))]
# distance_bins.append(1)
distance_bins = np.array(distance_bins)
# logarithmic bins
# distance_bins = np.logspace(-3, 0, n_distance_bins)
# distance_bins.append(1)
distance_probs = np.zeros(len(distance_bins))
# matrix of indices ranked pairwise distances size (n_papers, cutoff)
intra_doc_dist_ix_matrix = np.zeros((cutoff, len(distance_bins)))
print('Calculating distance probabilities')
for i in trange(len(distance_bins[:-1])):
    b = distance_bins[i]
    for paper_id in unique_paper_ids:
        # paper_dist[paper_id] = []
        for j, dist in enumerate(intra_doc_dists[paper_id]):
            if dist < distance_bins[i + 1] and dist >= distance_bins[i]:
                if j < cutoff-1:
                    intra_doc_dist_ix_matrix[j, i] += 1
                    distance_probs[i] += 1
                else:
                    intra_doc_dist_ix_matrix[cutoff-1, i] += 1
                    distance_probs[i] += 1
# distance_probs /= distance_probs.sum()
# intra_doc_dist_ix_matrix /=intra_doc_dist_ix_matrix.sum()
# verify that each bin sums to the same bin in distance_probs
print(intra_doc_dist_ix_matrix.sum(axis=0))
print(distance_probs)

print('Plotting histogram')
# use gradient for coloring
cmap = plt.cm.get_cmap('cividis', intra_doc_dist_ix_matrix.shape[0])
colors = [cmap(i) for i in range(intra_doc_dist_ix_matrix.shape[0])]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(intra_doc_dist_ix_matrix.T, stacked=True, align='left', bins=range(n_distance_bins), color=colors, density=True)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, orientation='vertical')
dist_bin_rounded = [round(b, 6) for b in distance_bins]
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels([dist_bin_rounded[0], dist_bin_rounded[15], dist_bin_rounded[-1]])
cbar.set_label('cosine distance')
ax.set_xlabel('distance bin')
ax.set_ylabel('frequency')
# relabel x-axis. Each bin name is the upper bound of the bin
ax.set_xticks(range(n_distance_bins))
ax.set_xticklabels(dist_bin_rounded)
ax.set_title('Stacked histogram of pairwise cosine distances') # rotate x-axis labels
plt.xticks(rotation=-45)
plt.savefig('../plots/stacked_histogram_neighbors.png')

plt.show()


# iterate through paper_dist and create a new df with the indices of each paper
df = pd.read_pickle("physics_5_9/all_embeds.pkl")
smallest_bin = pd.DataFrame(columns=df.columns)
for paper, group in tqdm(df.groupby('paper_id')):
    indices = paper_dist[paper]
    df.loc[indices]
    print(indices)
    break


df[df['paper_id'] == 'quant-ph/0507274']


# reimplement pdist but retaining indices
paper_dist_df = pd.DataFrame(paper_dist, columns=['paper_id','pgh_i', 'pgh_j', 'dist', 'text1', 'text2'])
# append text1 and text2 as a single column
text = paper_dist_df['text1'] + paper_dist_df['text2']
distances = paper_dist_df['dist']
text_hist = pd.DataFrame(text, columns=['text'])
# plot the histogram of the text
fig, ax = plt.subplots(figsize=(10, 6), ncols=1, nrows=2)
# distances.hist(bins=100, ax=ax)
text_hist['text'].apply(str.split).apply(len).hist(bins=100, ax=ax[0])
ax[0].set_xlabel('word count')
ax[0].set_ylabel('frequency')
ax[0].set_title('Histogram of text length')
text_hist['text'].apply(str.split).apply(len).hist(bins=100, ax=ax[1])
ax[1].set_xlabel('word count')
ax[1].set_ylabel('frequency')
ax[1].set_title('Histogram of text length Zoom')
plt.show()





text_hist['text'].apply(str.split).apply(len).hist(bins=100, ax=ax)
# text_hist['text'].str
ax.set_xlabel('text length')
ax.set_ylabel('frequency')
ax.set_title(f'Histogram of text length within distance 0 and {distance_bins[1]}')
# plt.savefig('../plots/histogram_text_length_under02.png')
plt.show()
fig, ax = plt.subplots(figsize=(10, 6))
paper_dist_df['dist'].hist(bins=100, density=True, ax=ax)
ax.title.set_text(f'Histogram of cosine distances within distance 0 and {distance_bins[1]}')
ax.set_xlabel('cosine distance')
ax.set_ylabel('frequency')
# plt.savefig('../plots/histogram_distances_under02.png')
plt.show()
