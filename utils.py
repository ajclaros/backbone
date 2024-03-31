import jsonlines
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from sklearn.manifold import TSNE
from flexible_clustering import hnsw

slategray = np.array([112, 128, 144]) / 255
slategray = np.append(slategray, 0.1)


def find_k_nearest_neighbors(input_vec, dataset, k=5):
    knn = NearestNeighbors(n_neighbors=k, metric="l2")
    knn.fit(dataset)
    distances, indices = knn.kneighbors([input_vec])
    return distances, indices


def search_year(ix, year, df, k=5):
    vec = np.array(df.iloc[ix]["embedding"])
    mask = df["year"] < year
    dataset = df[mask]
    dataset = np.array(dataset["embedding"].values.tolist())
    distances, indices = find_k_nearest_neighbors(vec, dataset, k)
    return list(indices[0])


def backward_search_year(ix, year, df, k=5):
    from_indices = [ix]
    if year == min(df["year"].unique()):
        print("No previous year")
        return []
    ixs = search_year(ix, year, df, k)
    to_indices = {i: [] for i in ixs}
    for i in ixs:
        from_indices.append(i)
        t_year = df.iloc[i]["year"]
        print(f"Searching year {t_year}")
        next_indices = backward_search_year(i, t_year, df, k)
        if next_indices == []:
            continue
        tmp = []
        if len(next_indices) > 0:
            for n in next_indices:
                tmp += n
        to_indices[i] = tmp
    return from_indices, to_indices


def month_number_to_string(month):
    return datetime(2000, month, 1).strftime("%B")


def int_year(year):
    if year > 24:
        return int(year + 1900)
    else:
        return int(year + 2000)


def l2_distance(a, b):
    return np.linalg.norm(a - b)


def cosine_distance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distance_cluster(df, distance):
    index = 0
    df_copy = df.copy()
    # mask = df["text"].str.len() > 100
    # df_copy = df[mask].copy()
    while index < df_copy.shape[0]:
        # if index<=3:
        #     index += 1
        #     continue
        # if index % 100 == 0:
        #     print(index, end=" ")
        row = df_copy.iloc[index]
        # df_doc = df_copy[df_copy['id'] == row['id']]
        df_doc = df_copy[df_copy["paper_id"] == row["paper_id"]].copy()
        mask = (
            df_doc[["tsne1", "tsne2"]].apply(
                lambda x: np.linalg.norm(x - row[["tsne1", "tsne2"]]), axis=1
            )
            < distance
        )
        df_doc = df_doc.loc[mask].copy()
        if df_doc.shape[0] == 1:
            index += 1
            continue
        avg = df_doc[["tsne1", "tsne2"]].mean(axis=0)
        df_copy.at[index, "tsne1"] = avg["tsne1"]
        df_copy.at[index, "tsne2"] = avg["tsne2"]
        # df_copy.at[index, 'text'] = string + "...(continued)"
        df_copy.at[index, "cluster_size"] = df_doc.shape[1]
        df_copy.drop(df_doc.index[1:].tolist(), inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        index += 1
    return df_copy


def arg_sort_array(arr, files):
    """
    arr is the numerical array to sort by, files is the list to return sorted
    each element of arr is meant to represent year, month, num, index. Sort accordingly "
    """
    mask = np.array(arr)
    indices = np.lexsort(mask.T)
    print(indices)
    return [files[i] for i in indices]


def folder_year_from_filename(filename):
    return filename.split("_")[2][:2]


def folder_month_from_filename(filename):
    return filename.split("_")[2][2:4]


def random_file_given_year(year, years=[5, 6, 7, 8, 9], num_files=1):
    files = os.listdir("aggregated_physics_5_9/embeds")
    files = [f for f in files if "0" + str(year) in f.split("_")[0][:2]]
    # files = [f for f in files if "0" + str(year) in f.split("_")[0][:2]]
    # print(f.split("_")[0][:2])
    sample_files = np.random.choice(files, num_files)
    print(f"{sample_files}, {year}: length {len(files)}")
    print(f"{sample_files}")
    return sample_files


def embed_paragraphs(docs):
    embeds = []
    for doc in docs:
        embeds.append(embeddings.embed_query(doc.page_content))
    return np.array(embeds)


def get_files_from_year(year, num_docs=200):
    files = os.listdir("aggregated_physics_5_9/data")
    files = [f for f in files if "0" + str(year) in f.split("_")[2][:2]]
    # select a random sample of files
    files = list(set(np.random.choice(files, num_docs)))
    return files


def group_files_by_month(files, num_docs=200):
    month_groups = {}
    for i in range(1, 13):
        s = f"{i:02d}"
        month_groups[s] = [f for f in files if s in f.split("_")[2][2:4]]
    total_docs = {}
    for month, files in month_groups.items():
        file_npy_dict = fetch_files_by_group(files, month, int(np.ceil(num_docs / 12)))
        for k, v in file_npy_dict.items():
            total_docs[k] = v

    # sort by {year}{month}_{num}_{index} with year being the highest priority and index the lowest
    # mask = [f.split(".")[0].split("_") for f in total_docs]
    # # turn all elements into integers
    # # split yearmonth num index
    # for i, m in enumerate(mask):
    #     tmp = [m[0][:2], m[0][2:4], m[1], m[2]]
    #     mask[i] = tmp
    # mask = [[int(i) for i in m] for m in mask]
    # # sort by year, month, num, index
    # # sort by year, month, num, index
    # # sort indices by year, month, num, index
    # mask = np.array(mask)
    # total_docs = arg_sort_array(mask, total_docs)
    # total_docs = [total_docs[i] for i in mask]
    return total_docs


def fetch_files_by_group(files, month, num_docs=10):
    total_docs = {}
    for name in files:
        npy_name = name.split(".")[0].split("_")[2:]
        npy_name = "_".join(npy_name)
        npy_files = os.listdir("aggregated_physics_5_9/embeds")
        npy_files = [f for f in npy_files if npy_name in f]
        npy_files = np.random.choice(npy_files, num_docs).tolist()
        # organize by year, month, num, index
        temp = [f.split(".")[0].split("_") for f in npy_files]
        for i, t in enumerate(temp):
            temp[i] = [t[0][:2], t[0][2:4], t[1], t[2]]
        temp = [[int(i) for i in t] for t in temp]
        npy_files = arg_sort_array(temp, npy_files)
        # npy_files = [npy_files[i] for i in indices]
        total_docs[name] = npy_files
    return total_docs


def process_files(
    years, num_docs, columns, domain="Physics", save=True, process_by="year"
):
    df = pd.DataFrame(columns=columns)
    min_year = min(years)
    max_year = max(years)
    for year in years:
        print(f"Processing year {year}")
        list_files = get_files_from_year(year, num_docs)
        print(list_files)
        total_files = group_files_by_month(list_files, num_docs)
        for filename in total_files.keys():
            print(f"Processing {filename}")
            location = filename.split(".")[0].split("_")[2:]
            with jsonlines.open(
                f"aggregated_{domain.lower()}_{min_year}_{max_year}/data/{filename}"
            ) as f:
                for i, line in enumerate(f):
                    npy_file = "_".join(location) + f"_{i}.npy"
                    if npy_file in total_files[filename]:
                        embeds = np.load(f"aggregated_physics_5_9/embeds/{npy_file}")
                        print(
                            f"{npy_file} has {embeds.shape[0]} embeddings and {len(line['body_text'])} paragraphs"
                        )
                        for j, paragraph in enumerate(line["body_text"]):
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
                                            "month": folder_month_from_filename(
                                                filename
                                            ),
                                            "embedding": [embed],
                                        }
                                    ),
                                ]
                            )
    df.reset_index(inplace=True, drop=True)
    df["year"] = df["year"].apply(lambda x: int_year(x))
    df = process_date(df, by="year")
    filename = f"./aggregated_{domain.lower()}_{min_year}_{max_year}/rand_{num_docs}_embeds.pkl"
    if save:
        df.to_pickle(f"{filename}")
    return df, filename


def process_date(df, by="year", save=True):
    if by == "year":
        df["date"] = df.apply(
            lambda x: datetime(x["year"] + 2000, 1, 1).timestamp(), axis=1
        )
    elif by == "month":
        df["date"] = df.apply(
            lambda x: datetime(x["year"] + 2000, x["month"], 1).timestamp(), axis=1
        )
    return df


def apply_tsne(df):
    X = np.array(df["embedding"].values.tolist())
    tsne = TSNE(n_components=2, random_state=42, early_exaggeration=12, perplexity=40)
    tsne_results = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(tsne_results, columns=["tsne1", "tsne2"])
    df["tsne1"] = df_tsne["tsne1"]
    df["tsne2"] = df_tsne["tsne2"]


def get_matrix(df, m=2, ef=200, level=0, shuffle=False):
    embeds = df["embedding"].values.tolist()
    doc_ids = df["paper_id"].unique()
    doc_boundaries = [
        df[df["paper_id"] == doc_ids[i]].shape[0] for i in range(len(doc_ids))
    ]
    doc_boundaries = np.cumsum(doc_boundaries)
    cluster = hnsw.HNSW(l2_distance, m=m, ef=ef)
    print("Adding embeddings")
    cluster, indices = add_embeds(embeds, cluster, shuffle=shuffle)
    graphs = cluster._graphs
    print(f"Number of graphs: {len(graphs)}")
    print("Filling matrix")
    matrix = np.zeros((len(embeds), len(embeds)))
    for key, value in graphs[level].items():
        if value == {}:
            continue
        for key2, value2 in value.items():
            x = indices[key]
            y = indices[key2]
            matrix[x, y] = value2
            matrix[y, x] = value2
            # matrix[key, key2] = value2

            # matrix[key2, key] = value2

    # for graph in graphs:
    #     for key, value in graph.items():
    #         if value == {}:
    #             continue
    #         for key2, value2 in value.items():
    #             matrix[key, key2] = value2
    #             matrix[key2, key] = value2
    return matrix, doc_boundaries


def add_embeds(embeds, cluster, shuffle=False):
    # if shuffle, add embeds in random order
    # keep track of original indices and reassemble matrix in original order

    if shuffle:
        indices = np.random.permutation(len(embeds))
        embeds = [embeds[i] for i in indices]
    else:
        indices = np.arange(len(embeds))

        embeds = [embeds[i] for i in indices]
    for e in embeds:
        cluster.add(np.array(e, dtype=np.float32))
    return cluster, indices


def drop_if_within_distance(df, distance=0.5):
    "Drops on embedding distance, not tsne distance"
    index = 0
    df_copy = df.copy()
    mask = (df["text"].str.len() > 300) & (df["text"].str.len() < 1000)
    df_copy = df[mask].copy()
    while index < df_copy.shape[0]:
        row = df_copy.iloc[index]
        df_doc = df_copy[df_copy["paper_id"] == row["paper_id"]].copy()
        mask = (
            df_doc["embedding"].apply(lambda x: np.linalg.norm(x - row["embedding"]))
            < distance
        )
        df_doc = df_doc.loc[mask].copy()
        if df_doc.shape[0] == 1:
            index += 1
            continue
        # drop all but the first row
        df_copy.drop(df_doc.index[1:].tolist(), inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        index += 1

    return df_copy


# year = 7
# tmp = df_copy[df_copy["year"] == year]
# randomly select row
# row = tmp.sample()
# get index in original dataframe


# df_copy["date"] = df_copy.apply(
#     lambda x: datetime(x["year"] + 2000, 1, 1).timdf_copy["date"] = df_copy.apply(lambda x: datetime(x["year"]+2000, 1, 1).timestamp(), axis=1)
# values.tolist()), 5)
# indices = backward_search_year(row, 6, df_copy, 5)
# from_indices, to_indices = backward_search_year(row.index[0], year, df_copy, 10)
# colors = plt.cm.jet(np.linspace(0, 1, len(indices)))
# network = nx.Graph()
# network = nx.from_numpy_array(matrix)
# nx.set_node_attributes(network, df_copy["paper_id"].to_dict(), "paper_id")
# nx.set_node_attributes(network, df_copy["tsne1"].to_dict(), "tsne1")
# nx.set_node_attributes(network, df_copy["tsne2"].to_dict(), "tsne2")
# nx.set_node_attributes(network, df_copy["date"].to_dict(), "date")
# for i, ix in enumerate(indices):
#     for index in ix:
#         df_copy.at[index, "color"] = colors[i]

# colors = plt.cm.viridis(np.linspace(0, 1, len(to_indices)))
# df_copy["color"] = [slategray for i in range(df_copy.shape[0])]
# df_copy.at[row.index[0], "color"] = [1, 0, 0, 1]
# nx.set_node_attributes(network, df_copy["color"].to_dict(), "color")
# node_color = [network.nodes[n]["color"] for n in network.nodes]

# pos = {}
# for node in network.nodes:
#     pos[node] = (
#         network.nodes[node]["date"],
#         network.nodes[node]["tsne1"],
#         network.nodes[node]["tsne2"],
#     )
# node_xyz = np.array([pos[v] for v in network])
# # edge_xyz = np.array([(pos[u], pos[v]) for u, v in network.edges])


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], c=node_color, s=10)

# plt.show()

# for ix, val in to_indices.items():
#     for v in val:
#         ax.plot(
#             [pos[ix][0], pos[v][0]],
#             [pos[ix][1], pos[v][1]],
#             [pos[ix][2], pos[v][2]],
#             color="k",
#             ls="--",
#         )

# ax.set_xticks(df_copy["date"].unique())
# # set label to year-month
# ax.set_xticklabels(
#     [datetime.fromtimestamp(date).strftime("%Y") for date in df_copy["date"].unique()]
# )
# ax.set_title("Semantic Network")
# ax.set_xlabel("date")
# # rotate x axis labels
# # plt.xticks(rotation=90)
# ax.set_ylabel("tsne1")

# ax.set_zlabel("tsne2")
# plt.show()
