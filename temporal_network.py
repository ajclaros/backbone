import pandas as pd
import numpy as np
import os
import jsonlines
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from flexible_clustering import hnsw
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import networkx as nx
from utils import *

# get slategray color as a array
slategray = np.array([112, 128, 144]) / 255
# add alpha channel
slategray = np.append(slategray, 0.05)

domain = "Physics"
years = [
    5,
    6,
    7,
    8,
    9,
]
num_docs = 50
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}


columns = [
    "paper_id",
    "section",
    "sec_number",
    "text",
    "embedding",
]


df = pd.DataFrame(columns=columns)
for year in years:
    print(f"Processing year {year}")
    list_files = get_files_from_year(year, num_docs)
    total_files = group_files_by_month(list_files, num_docs)
    for filename in total_files.keys():
        print(f"Processing {filename}")
        location = filename.split(".")[0].split("_")[2:]
        with jsonlines.open(f"aggregated_physics_5_9/data/{filename}") as f:
            for i, line in enumerate(f):
                npy_file = "_".join(location) + f"_{i}.npy"
                if npy_file in total_files[filename]:
                    embeds = np.load(f"aggregated_physics_5_9/embeds/{npy_file}")
                    print(
                        f"{npy_file} has {embeds.shape[0]} embeddings and {len(line['body_text'])} paragraphs"
                    )
                    for j, paragraph in enumerate(line["body_text"]):
                        # embed = embeds[count]
                        embed = embeds[j]
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        "paper_id": line["paper_id"],
                                        "section": paragraph["section"],
                                        "sec_number": paragraph["sec_number"],
                                        "text": paragraph["text"],
                                        "year": year,
                                        "month": folder_month_from_filename(filename),
                                        "embedding": [embed],
                                    }
                                ),
                            ]
                        )

df.to_pickle("./aggregated_physics_5_9/rand_{num_docs}_embeds.pkl")
df.reset_index(inplace=True, drop=True)

# using year and month, encode date as a number

df["year"] = df["year"].apply(lambda x: int(x))
df["month"] = df["month"].astype(int)
df["date"] = df.apply(lambda x: datetime(x["year"], x["month"], 1).timestamp(), axis=1)
X = np.array(df["embedding"].tolist())
tsne = TSNE(n_components=2, perplexity=40, n_iter=300, early_exaggeration=40)
tsne_results = tsne.fit_transform(X)
df_tsne = pd.DataFrame(tsne_results, columns=["tsne1", "tsne2"])
# add tsne results to dataframe
df["tsne1"] = df_tsne["tsne1"]
df["tsne2"] = df_tsne["tsne2"]
df["cluster_size"] = 1
# df["date"] = df.apply(lambda x: datetime(x["year"], 1, 1).timestamp(), axis=1)

df_copy = distance_cluster(df, 2.5)
embeds = df_copy[["tsne1", "tsne2"]].values
cluster = hnsw.HNSW(l2_distance, m=2, ef=200)
for e in embeds:
    cluster.add(np.array(e, dtype=np.float32))
graphs = cluster._graphs
print("creating network")
matrix = np.zeros((embeds.shape[0], embeds.shape[0]))
for graph in graphs:
    for key, value in graph.items():
        if value == {}:
            continue
        for key2, value2 in value.items():
            matrix[key, key2] = value2
            matrix[key2, key] = value2
network = nx.from_numpy_array(matrix)
nx.set_node_attributes(network, df_copy["paper_id"].to_dict(), "paper_id")
nx.set_node_attributes(network, df_copy["tsne1"].to_dict(), "tsne1")
nx.set_node_attributes(network, df_copy["tsne2"].to_dict(), "tsne2")
nx.set_node_attributes(network, df_copy["date"].to_dict(), "date")
louvain_communities = nx.community.louvain_communities(network)
# nx.set_node_attributes(network, louvain_communities, "louvain_community")
# assign each node to a community
for i, community in enumerate(louvain_communities):
    for node in community:
        network.nodes[node]["louvain_community"] = i
options = {
    "node_size": 5,
    "width": 0.5,
    "with_labels": False,
}
# colors = plt.cm.viridis(np.linspace(0, 1, len(df["paper_id"])))
# colors = plt.cm.tab20(np.linspace(0, 1, len(df["paper_id"].unique())))
colors = plt.cm.Set1(np.linspace(0, 1, len(louvain_communities)))
node_color = []
opacities = []
show_community = 0
# all_paper_ids = df["paper_id"].unique()
for node in network.nodes:
    if network.nodes[node]["louvain_community"] == show_community:
        node_color.append(colors[0])
        opacities.append(1)
    else:
        node_color.append(slategray)

# for each node, set the color to the paper id

pos = {}
for node in network.nodes:
    pos[node] = (
        network.nodes[node]["date"],
        network.nodes[node]["tsne1"],
        network.nodes[node]["tsne2"],
    )
node_xyz = np.array([pos[v] for v in network])
edge_xyz = []
for u, v in network.edges:
    if (
        network.nodes[u]["louvain_community"] == show_community
        and network.nodes[v]["louvain_community"] == show_community
    ):
        edge_xyz.append([pos[u], pos[v], "red"])
    # if the starting node in the edge is in the show_community but the ending node is not
    # color yellow
    elif (
        network.nodes[u]["louvain_community"] == show_community
        and network.nodes[v]["louvain_community"] != show_community
    ):
        edge_xyz.append([pos[u], pos[v], "green"])
    # if the ending node in the edge is in the show_community but the starting node is not
    # color red
    elif (
        network.nodes[u]["louvain_community"] != show_community
        and network.nodes[v]["louvain_community"] == show_community
    ):
        edge_xyz.append([pos[u], pos[v], "blue"])
    else:
        edge_xyz.append([pos[u], pos[v], "black"])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# ax.scatter(node_xyz[:, 0], node_xyz[:, 1], c=node_color)
ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], c=node_color, alpha=0.5)
for edge in edge_xyz:
    if edge[2] == "black":
        continue
    ax.plot(
        [edge[0][0], edge[1][0]],
        [edge[0][1], edge[1][1]],
        [edge[0][2], edge[1][2]],
        c=edge[2],
        alpha=0.2,
        ls="--",
        lw=1.5,
    )


fig.tight_layout()
# date axis ticks set to year
ax.set_xticks(df["date"].unique())
# set label to year-month
ax.set_xticklabels(
    [datetime.fromtimestamp(date).strftime("%Y-%m") for date in df["date"].unique()]
)
ax.set_title("Semantic Network")
ax.set_xlabel("date")
# rotate x axis labels
plt.xticks(rotation=90)
ax.set_ylabel("tsne1")

ax.set_zlabel("tsne2")
plt.show()
