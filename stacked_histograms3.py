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
# find k = 200 nearest neighbors for each paragraph in each paper
cutoff = 200
neighbor_distances = {}
def return_neighbors(index):
    global unique_paper_ids, embeds, df, cutoff
    if index % 100 == 0:
        print(index)
    paper_id = unique_paper_ids[index]
    df_t = df[df['paper_id'] == paper_id]
    if df_t.shape[0] < cutoff:
        cutoff = df_t.shape[0]
    embeds_paper = embeds[np.array(paper_ids) == paper_id]
    # calculate knn to extract ranked distances from each point
    knn = NearestNeighbors(n_neighbors=cutoff, metric='cosine' )
    knn.fit(embeds_paper)
    distances, indices = knn.kneighbors(embeds_paper)
    return distances, indices
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(return_neighbors, range(len(unique_paper_ids))), total=len(unique_paper_ids)))
    for i in range(len(results)):
        distances, indices = results[i]
        paper_id = unique_paper_ids[i]
        neighbor_distances[paper_id] = distances
# create df, where columns are the neighbor ordered by distance, and rows are the paragraphs in each paper
df = pd.DataFrame(columns=['paper_id', 'paragraph_ix', 'distance', 'neighbor_ix'])
def get_row_data(ix):
    global neighbor_distances, unique_paper_ids
    paper_id = unique_paper_ids[ix]
    print(ix)
    distances = neighbor_distances[paper_id]
    rows = []
    for paragraph_ix in range(len(distances)):
        for neighbor_ix in range(1, distances.shape[1]):
            rows.append([paper_id, paragraph_ix, distances[paragraph_ix, neighbor_ix], neighbor_ix])
    return pd.DataFrame(rows, columns=['paper_id', 'paragraph_ix', 'distance', 'neighbor_ix'])

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(get_row_data, range(len(unique_paper_ids))), total=len(unique_paper_ids)))
    for result in results:
        df = pd.concat([df, result])
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.histplot(data=df, x='distance', hue='neighbor_ix', bins=30, ax=ax, multiple='stack', palette='cividis', linewidth=0, edgecolor=None)
plt.xlabel('absolute cosine distance')
plt.ylabel('number of paragraphs')
plt.title('Histogram of cosine distances between paragraphs')
plt.show()
