import os
import jsonlines
import networkx as nx
from langchain_community.document_loaders import JSONLoader
import pickle
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

import time


domain = "Physics"
data_path = "aggregated_physics_5_9/data"
embed_path = "aggregated_physics_5_9/embeds"
files = os.listdir(data_path)
collection_string = "unarXive"

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
)


def printx(x):
    for i in x:
        print(i[:10])


for ix, name in enumerate(files[:]):
    # if ex:
    #     break
    # print(f"\nTotal documents: {num_docs}")
    print(f"\nProcessing {name} ({ix+1}/{len(files)})")
    with jsonlines.open(data_path + f"/{name}") as f:
        for i, line in enumerate(f):
            if f"{name}_{i}.npy" in os.listdir(embed_path):
                continue
            data = []
            for j, section in enumerate(line["body_text"]):

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
            embeds = []
            for doc in data:
                embeds.append(embeddings.embed_query(doc.page_content))
            embeds = np.float32(embeds)

            name = name.split(".")[0].split("_")[-2:]
            name = "_".join(name)
            np.save(f"{embed_path}/{name}_{i}.npy", embeds)

data = []
for j, section in enumerate(line["body_text"]):
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

embeds = []
for doc in data:
    embeds.append(embeddings.embed_query(doc.page_content))
    embeds = np.float32(embeds)
