import os
import jsonlines
import networkx as nx
import matplotlib.pyplot as plt
from langchain_community.document_loaders import JSONLoader
import pickle

# from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from pprint import pprint

# import sqlalchemy
# from sqlalchemy import func
# from sqlalchemy.dialects.postgresql import JSON, UUID
# from sqlalchemy.orm import Session, relationship
import time

domain = "Physics"
file_path = "aggregated_physics_5_9/data"
files = os.listdir(file_path)
data = []
num_docs = 0
collection_string = "unarXive"

ex = False
for ix, name in enumerate(files[:]):
    if ex:
        break
    # print(f"\nTotal documents: {num_docs}")
    print(f"\nProcessing {name} ({ix+1}/{len(files)})")
    with jsonlines.open(file_path + f"/{name}") as f:
        for i, line in enumerate(f):
            if ex:
                break
            for j, section in enumerate(line["body_text"]):
                # print(j, end=" ")
                if len(data) == 10:
                    ex = True
                    print(f"Stopping at {name}, paper {i}, id: {line['paper_id']}, {j}")
                    break
                # if (
                #     section["section"] is None
                #     or section["section"].lower() != "introdution"
                # ):
                #     continue
                # num_docs += 1

                metadata = {
                    "paper_id": line["paper_id"],
                    "section": section["section"],
                    "sec_number": section["sec_number"],
                }
                doc = Document(
                    page_content=line["body_text"][j]["text"].lower(),
                    metadata=metadata,
                )
                data.append(doc)
print(f"Number of documents: {len(data)}")
print("Creating vector store")
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
)
CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vector_db"
COLLECTION_NAME = "unarXive"
query = "quantum mechanics"
# x = pg_embedding.similarity_search_with_score(query, k=20)


# extend pgvector class for create_hnsw_index
# pg_embedding = PGVector.from_documents(
#     connection_string=CONNECTION_STRING,
#     collection_name=COLLECTION_NAME,
#     embedding=embeddings,
#     documents=data,
# )
# query = "quantum mechanics"
# embeddings.embed_query(query)
# import hnswlib

# ids = np.arange(len(data))
total_embeds = 0


def embedding_fun(doc):
    global total_embeds
    total_embeds += 1
    if total_embeds % 100 == 0:
        print(f"Embedding {total_embeds} documents")
    return embeddings.embed_query(doc.page_content)


print(f"number of documents: {len(data)}")
# t1 = time.time()
embeds = np.float32([embeddings.embed_query(doc.page_content) for doc in data])
# t2 = time.time()
# print(f"Embedding took {t2-t1} seconds")
# print("Deleting documents")
# del data
# np.save(f"embeds_{embeds.shape[0]}.npy", embeds)

# embeds = np.float32([embedding_fun(doc) for doc in data])
# p = hnswlib.Index(space="cosine", dim=embeds.shape[1])
# p.init_index(max_elements=embeds.shape[0], ef_construction=200, M=16)
# p.add_items(embeds, ids)
# p.set_ef(50)
# labels, distances = p.knn_query(embeds, k=20)
# import flexible_clustering as fc
# from flexible_clustering import hnsw


# # clusterer = fc.FISHDBC(
# #     np.float32([embeddings.embed_query(doc.page_content) for doc in data])
# # )
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def l2_distance(a, b):
#     return np.linalg.norm(a - b)


# clusterer = hnsw.HNSW(cosine_similarity, m=5, ef=200)
# clusterer = hnsw.HNSW(l2_distance, m=5, ef=200)
# for e in embeds:
#     clusterer.add(e)
# graphs = clusterer._graphs

# matrix = np.zeros((embeds.shape[0], embeds.shape[0]))
# for graph in graphs:
#     for key, value in graph.items():
#         if value == {}:
#             continue
#         for key2, value2 in value.items():
#             matrix[key, key2] = value2
#             matrix[key2, key] = value2


# plt.imshow(matrix, cmap="jet", interpolation="nearest")
# plt.colorbar()
# plt.legend()
# plt.show()
# # check location of array if is 0
# # print(len(np.where(matrix == 0)))
# # save matrix
# np.save(f"l2matrix_{embeds.shape[0]}.npy", matrix)
# # save graphs
# with open(f"graphs_{embeds.shape[0]}.pkl", "wb") as f:
#     pickle.dump(graphs, f)

# # Number of documents: 21.047.453 *
