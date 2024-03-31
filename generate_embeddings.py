import os
import jsonlines
import networkx as nx
import matplotlib.pyplot as plt
from langchain_community.document_loaders import JSONLoader
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from pprint import pprint
import time
import concurrent.futures

domain = "Physics"
data_path = "aggregated_physics_5_10/data"
embed_path = "aggregated_physics_5_10/embeds"
files = os.listdir(data_path)
collection_string = "unarXive"

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
)


def process_line(name, i, line, embed_path):
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


results = []
num_processes = 30
for ix, name in enumerate(files[:]):
    print(f"\nProcessing {name} ({ix+1}/{len(files)})")
    with jsonlines.open(data_path + f"/{name}") as f:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_processes
        ) as executor:
            for i, line in enumerate(f):
                results.append(executor.submit(process_line, name, i, line, embed_path))
                if len(results) == num_processes:
                    for future in concurrent.futures.as_completed(results):
                        future.result()
                    results = []
            for future in concurrent.futures.as_completed(results):
                future.result()
                results = []
