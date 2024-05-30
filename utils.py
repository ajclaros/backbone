import jsonlines
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from datetime import datetime
from sklearn.manifold import TSNE
from flexible_clustering import hnsw
from tqdm import tqdm, trange

slategray = np.array([112, 128, 144]) / 255
slategray = np.append(slategray, 0.1)


def find_k_nearest_neighbors(input_vec, dataset, k=5):
    """
    Finds the k nearest neighbors in the dataset
    """
    knn = NearestNeighbors(n_neighbors=k, metric="l2")
    knn.fit(dataset)
    distances, indices = knn.kneighbors([input_vec])
    return distances, indices


def search_year(ix, year, df, k=5):
    """
    Searches for the year and returns the indices of the documents
    """
    vec = np.array(df.iloc[ix]["embedding"])
    mask = df["year"] < year
    dataset = df[mask]
    dataset = np.array(dataset["embedding"].values.tolist())
    distances, indices = find_k_nearest_neighbors(vec, dataset, k)
    return list(indices[0])


def backward_search_year(ix, year, df, k=5):
    """
    Searches for the previous year and returns the indices of the documents
    """
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
    """
    Converts month number to string
    """
    return datetime(2000, month, 1).strftime("%B")


def str_year(year):
    if year < 10:
        return f"0{year}"
    return str(year)


def int_year(year):
    """
    Returns the year as an integer
    """
    if year > 24:
        return int(year + 1900)
    else:
        return int(year + 2000)


def str_month(month):
    """
    Returns the month as an integer
    """
    if month > 9:
        return int(month)
    else:
        return "0" + str(month)


def l2_distance(a, b):
    return np.linalg.norm(a - b)


def cosine_distance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distance_cluster(df, distance):
    """
    Clusters the dataframe based on the distance.
    If the distance between two points is less than the distance, the two points are considered to be in the same cluster
    """
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


def distance_cluster2(df, distance):
    """
    Clusters the dataframe based on the distance.
    If the distance between two points is less than the distance, the two points are considered to be in the same cluster "
    """
    df_copy = df.copy()
    t_df = pd.DataFrame(columns=df.columns)

    for group, data in tqdm(df_copy.groupby("paper_id")):
        for i, row in data.iterrows():
            mask = (
                data[["tsne1", "tsne2"]].apply(
                    lambda x: np.linalg.norm(x - row[["tsne1", "tsne2"]]), axis=1
                )
                < distance
            )
            data.loc[mask, "tsne1"] = data[mask]["tsne1"].mean()
            data.loc[mask, "tsne2"] = data[mask]["tsne2"].mean()
            data.loc[mask, "cluster_size"] = data[mask].shape[0]
            data.drop(data[mask].index[1:].tolist(), inplace=True)
            t_df = pd.concat([t_df, data])
    return t_df


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
    """
    Returns the year from the filename
    """
    return filename.split("_")[2][:2]


def folder_month_from_filename(filename):
    """
    Returns the month from the filename
    """
    return filename.split("_")[2][2:4]


def random_file_given_year(year, years=[5, 6, 7, 8, 9], num_files=1):
    """
    Returns a random file given a year
    """
    files = os.listdir("physics_5_9/embeds")
    files = [f for f in files if "0" + str(year) in f.split("_")[0][:2]]
    # files = [f for f in files if "0" + str(year) in f.split("_")[0][:2]]
    # print(f.split("_")[0][:2])
    sample_files = np.random.choice(files, num_files)
    print(f"{sample_files}, {year}: length {len(files)}")
    print(f"{sample_files}")
    return sample_files


def embed_paragraphs(docs):
    """
    Embeds paragraphs in a list of documents
    """
    embeds = []
    for doc in docs:
        embeds.append(embeddings.embed_query(doc.page_content))
    return np.array(embeds)


def get_files_from_year(year, num_docs=200):
    """
    Fetches files from a given year. If num_docs is -1, fetches all files
    """
    files = os.listdir("physics_5_9/data")
    files = [f for f in files if "0" + str(year) in f.split("_")[2][:2]]
    # select a random sample of files
    if num_docs == -1:
        files = list(set(files))
    else:
        files = list(set(np.random.choice(files, num_docs)))
    return files


def group_files_by_month(files, num_docs=200):
    """
    Groups files by month. If num_docs is -1, fetches all files
    """
    month_groups = {}
    if num_docs == -1:
        num_docs = len(files)
    for i in range(1, 13):
        s = f"{i:02d}"
        month_groups[s] = [f for f in files if s in f.split("_")[2][2:4]]
    total_docs = {}
    for month, files in month_groups.items():
        if num_docs == -1:
            num_docs = len(files)
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
    """
    Fetches a random sample of files from the list of files
    if num_docs is -1, fetches all files
    :param files: list of files
    :param month: month to fetch if num_docs is -1, fetches all files, skip for now
    :param num_docs: number of files to fetch if -1, fetches all files
    """
    total_docs = {}
    for name in files:
        npy_name = name.split(".")[0].split("_")[2:]
        npy_name = "_".join(npy_name)
        npy_files = os.listdir("physics_5_9/embeds")
        npy_files = [f for f in npy_files if npy_name in f]
        print(f"Number of files: {len(npy_files)}")
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


def process_files(years, num_docs, columns, domain="Physics", save=True):
    """
    Processes the files and returns a dataframe
    :param years: list of years to process
    :param num_docs: number of documents to process
    """
    from tqdm import tqdm

    df = pd.DataFrame(columns=columns)
    min_year = min(years)
    max_year = max(years)
    for year in years:
        print(f"Processing year {year}")
        list_files = get_files_from_year(year, num_docs)
        print(list_files)
        total_files = group_files_by_month(list_files, num_docs)
        for filename in tqdm(total_files.keys()):
            print(f"Processing {filename}")
            location = filename.split(".")[0].split("_")[2:]
            with jsonlines.open(
                f"{domain.lower()}_{min_year}_{max_year}/data/{filename}"
            ) as f:
                for i, line in enumerate(f):
                    npy_file = "_".join(location) + f"_{i}.npy"
                    if npy_file in total_files[filename]:
                        embeds = np.load(f"physics_5_9/embeds/{npy_file}")
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
    if num_docs == -1:
        filename = f"./{domain.lower()}_{min_year}_{max_year}/{num_docs}_embeds.pkl"
    else:
        filename = (
            f"./{domain.lower()}_{min_year}_{max_year}/rand_{num_docs}_embeds.pkl"
        )
    df = process_date(df, by="year")
    if save:
        df.to_pickle(f"{filename}")
    return df, filename


def process_date(df, by="year", save=True):
    """
    Adds a date column to the dataframe based on the year or month
    """
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
    """
    Applies tsne to the embeddings and adds the tsne1 and tsne2 columns to the dataframe
    """
    X = np.array(df["embedding"].values.tolist())
    tsne = TSNE(n_components=2, random_state=42, early_exaggeration=12, perplexity=40)
    tsne_results = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(tsne_results, columns=["tsne1", "tsne2"])
    df["tsne1"] = df_tsne["tsne1"]
    df["tsne2"] = df_tsne["tsne2"]


def get_matrix(df, m=2, ef=200, level=0, shuffle=False):
    """
    Returns a matrix and doc boundaries
    """
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
    return matrix, doc_boundaries


def get_matrix2(df, m=2, ef=200, level=0, shuffle=False, metric=l2_distance):
    """
    Returns a networkx graph and doc boundaries
    """
    embeds = df["embedding"].values.tolist()
    doc_ids = df["paper_id"].unique()
    doc_boundaries = [
        df[df["paper_id"] == doc_ids[i]].shape[0] for i in range(len(doc_ids))
    ]
    doc_boundaries = np.cumsum(doc_boundaries)
    cluster = hnsw.HNSW(metric, m=m, ef=ef)
    print("Adding embeddings")
    cluster, indices = add_embeds(embeds, cluster, shuffle=shuffle)
    graphs = cluster._graphs
    print(f"Number of graphs: {len(graphs)}")
    print("Filling matrix")
    for key, value in graphs[level].items():
        if value == {}:
            continue
        for key2, value2 in value.items():
            graphs[level][key][key2] = {"weight": value2}
    network = nx.from_dict_of_dicts(graphs[level])
    return network, doc_boundaries


def add_embeds(embeds, cluster, shuffle=False):
    """
    Adds embeddings to the cluster
    """

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


def distance_matrix(df, metric=cosine_distance):
    dist_matrix = np.zeros((df.shape[0], df.shape[0]))
    for i in trange(df.shape[0]):
        for j in range(df.shape[0]):
            if i == j:
                dist_matrix[i, j] = 0
            x = metric(df.iloc[i]["embedding"], df.iloc[j]["embedding"])
            dist_matrix[i, j] = x
            dist_matrix[j, i] = x
    dist_matrix = (dist_matrix - dist_matrix.min()) / (
        dist_matrix.max() - dist_matrix.min()
    )
    dist_matrix[dist_matrix > 0] = 1 - dist_matrix[dist_matrix > 0]
    return dist_matrix


def distance_matrix_parallel(df, metric=cosine_distance):
    import concurrent.futures

    dist_matrix = np.zeros((df.shape[0], df.shape[0]))

    def distance(i, j):
        if i == j:
            return 0
        return metric(df.iloc[i]["embedding"], df.iloc[j]["embedding"])

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = {
            executor.submit(distance, i, j): (i, j)
            for i in trange(df.shape[0])
            for j in range(df.shape[0])
        }
        for future in concurrent.futures.as_completed(results):
            i, j = results[future]
            dist_matrix[i, j] = future.result()
    return dist_matrix


def in_doc(doc_i, doc_j, df):
    """
    Returns a tuple of whether the two documents are in the same paper, the index of the second document, and the distance between the two documents
    """
    distance = l2_distance(df.iloc[doc_i]["embedding"], df.iloc[doc_j]["embedding"])
    return (df.iloc[doc_i]["paper_id"] == df.iloc[doc_j]["paper_id"], doc_j, distance)


def in_doc_list(doc_i, doc_list, df):
    """
    Returns a list of tuples of whether the two documents are in the same paper, the index of the second document, and the distance between the two documents
    """
    return [in_doc(doc_i, doc_j, df) for i, doc_j in enumerate(doc_list) if i != doc_i]


def knn_list(input_vec, df, k=5):
    """
    Returns a list of tuples of whether the two documents are in the same paper, the index of the second document, and the distance between the two documents
    """
    distances, indices = find_k_nearest_neighbors(input_vec, df["embedding"], k)
    return in_doc_list(doc_i, indices, df)


def grab_doc(paper_id, year, month, part, agg_path=None):
    """
    Retrieves a document from the jsonl file
    """
    if path is None:
        paragraphs = []
        part = f"00{part}"
        with jsonlines.open(
            f"../{str_year(year)}/arXiv_src_{str_year(year)}{str_month(month)}_{part}.jsonl"
        ) as f:
            for i, line in enumerate(f, 1):
                if line["paper_id"] == paper_id:
                    return line
    if agg_ath:
        path = [
            f
            for f in os.listdir(path)
            if f"{str_year(year)}_{str_month(month)}_{part}" in f
        ]

        paragraphs = []
        with jsonlines.open(
            f"{agg_path}/{str_year(year)}_{str_month(month)}_{part}.jsonl"
        ) as f:
            for i, line in enumerate(f, 1):
                if line["paper_id"] == paper_id:
                    return line


def embed_paper(paper_obj):
    """
    Embeds a paper object
    """
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    embeds = []
    for section in paper_obj["body_text"]:
        embeds.append(embeddings.embed_query(section["text"]))
    return np.array(embeds)


# physics/0501005
