# This script iterates over the data files and checks if the number of embeddings matches the number of paragraphs
# This script does not generate any data


import os
import jsonlines
# import networkx as nx
# import matplotlib.pyplot as plt
# from langchain_community.document_loaders import JSONLoader
# import pickle

# from langchain_community.embeddings import GPT4AllEmbeddings
# from langchain.text_splitter import CharacterTextSplitter

# from langchain_community.vectorstores import PGVector
# from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
# from pprint import pprint

domain = "Physics"
data_path = "aggregated_physics_5_9/data"
embed_path = "aggregated_physics_5_9/embeds"
files = os.listdir(data_path)

# model_kwargs = {"device": "cuda"}
# encode_kwargs = {"normalize_embeddings": False}
# embeddings = HuggingFaceEmbeddings(
#     model_name="allenai/scibert_scivocab_uncased",
# )
no_files = []


def printx(x):
    for i in x:
        print(i[:10])


def embed_paragraphs(docs):
    embeds = []
    for doc in docs:
        embeds.append(embeddings.embed_query(doc.page_content))
    return np.array(embeds)


ex = False
for ix, name in enumerate(files[:]):
    if ex:
        break
    # print(f"\nTotal documents: {num_docs}")
    print(f"\nProcessing {name} ({ix+1}/{len(files)})")
    with jsonlines.open(data_path + f"/{name}") as f:
        for i, line in enumerate(f):
            if ex:
                break
            # npy_name = name.split(".")[0] + f"_{i}.npy"
            npy_file = name.split(".")[0].split("_")[2:]
            npy_file = "_".join(npy_file) + f"_{i}.npy"
            if not os.path.exists(embed_path + f"/{npy_file}"):
                print(f"Error: {npy_file} does not exist")
                no_files.append(npy_file)
                continue
            emedding = np.load(embed_path + f"/{npy_file}")
            # if npy_file == "0501_001_85.npy":
            #     ex = True
            #     print(f"name: {name}")
            #     break
            if emedding.shape[0] != len(line["body_text"]):
                print(
                    f"Error: {npy_file} has {emedding.shape[0]} embeddings, but {len(line['body_text'])} paragraphs"
                )
                docs = []
                continue
                for j, section in enumerate(line["body_text"]):
                    metadata = {
                        "paper_id": line["paper_id"],
                        "section": section["section"],
                        "sec_number": section["sec_number"],
                    }
                    doc = Document(
                        page_content=section["text"].lower(),
                        metadata=metadata,
                    )
                    docs.append(doc)
                print(f"Embedding {npy_file}")
                new_embedding = embed_paragraphs(docs)
                if new_embedding.shape[0] != len(line["body_text"]):
                    print(f"Error: again, wrong number of embeddings")
                    print(f"names: {npy_file}, {name}")
                    print(f"Length of embeddings: {new_embedding.shape[0]}")
                    print(f"Length of paragraphs: {len(line['body_text'])}")
                    print("Skipping")
                    continue
                print(f"Saving {npy_file}")
                np.save(embed_path + f"/{npy_file}", new_embedding)
            # else:
            #     print(f"Correct: {npy_file} has {emedding.shape[0]} embeddings")
            #     print(f"Length of paragraphs: {len(line['body_text'])}")
            #     print("\n\n\n")
print(f"Files with no embeddings: {no_files}")
with open("no_files.txt", "w") as f:
    f.write("\n".join(no_files))
