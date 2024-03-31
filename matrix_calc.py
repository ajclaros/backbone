import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle


rows = 10001
matrix = np.load(
    [name for name in os.listdir() if "l2" in name and str(rows) in name][0]
)
embeds = np.load(
    [name for name in os.listdir() if "embeds" in name and str(rows) in name][0]
)


# threshold values to 1
path_matrix = matrix.copy()
path_matrix[path_matrix > 0] = 1
nx.draw(nx.from_numpy_array(path_matrix))
# find paths of length 2
path_matrix = np.dot(path_matrix, path_matrix)

matrix2 = matrix.copy()
# normalize
matrix2 = (matrix2 - matrix2.min()) / (matrix2.max() - matrix2.min())
# set non-zero values to 1-x
matrix2[matrix2 > 0] = 1 - matrix2[matrix2 > 0]
# set non-zero values to their negative
plt.imshow(matrix2, cmap="jet", interpolation="nearest")
plt.colorbar()
plt.legend()
plt.show()
plt.colorbar()
plt.legend()
# load pickle file
graphs = pickle.load(
    open(
        [name for name in os.listdir() if name.endswith(".pkl") and str(rows) in name][
            0
        ],
        "rb",
    )
)

for val in graphs[0][0]:
    print(val, graphs[0][0][val])

plt.imshow(path_matrix, cmap="jet", interpolation="nearest")
plt.colorbar()
plt.legend()
plt.show()


def l2_distance(a, b):
    return np.linalg.norm(a - b)


for key, val in graphs[0][0].items():
    print(key, val)
print(l2_distance(embeds[0], embeds[1]))

# find number of connected components
G = nx.from_numpy_array(matrix)
print(nx.number_connected_components(G))
# find the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
# subgraphs of each connected component as adjacency matrix
# subgraphs = [nx.to_numpy_array(G.subgraph(c)) for c in nx.connected_components(G)]
# # find average shortest path length of each connected component
# for sg in subgraphs:
#     print(nx.average_shortest_path_length(nx.from_numpy_array(sg)))
# get degree distribution
# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degrees = [d for n, d in G.degree()]
degree_counts = np.bincount(degrees)
ccdf = 1 - np.cumsum(degree_counts) / sum(degree_counts)
plt.loglog(ccdf, "b-", marker="o")
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")
plt.show()
