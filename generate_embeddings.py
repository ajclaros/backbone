import os
import jsonlines
import orjson
from tqdm import tqdm
import argparse

print("Starting script")
parser = argparse.ArgumentParser()
parser.add_argument("-p", type=int, default=10, help="Number of processes")
parser.add_argument("-d", type=str, default="Physics", help="Domain to aggregate")
parser.add_argument("-y", type=str, default="5 6 7 8 9", help="Years to aggregate")
parser.add_argument("-dir", type=int, default=1, help="Direction to process files")
args = parser.parse_args()
domain = args.d
years = [int(y) for y in args.y.split(" ")]
direction = args.dir
num_processes = args.p


# import networkx as nx
# import matplotlib.pyplot as plt
# from langchain_community.document_loaders import JSONLoader
# import pickle
# from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from pathlib import Path
# from pprint import pprint
# import time
import concurrent.futures

# domain = "Physics"
# data_path = "physics_5_9/data"
# embed_path = "physics_5_9/embeds"

domain = domain.replace(" ", "_")
data_path = f"{domain.lower()}_{years[0]}_{years[-1]}/data"
embed_path = f"{domain.lower()}_{years[0]}_{years[-1]}/embeds"



files = os.listdir(data_path)
collection_string = "unarXive"

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
# embeddings = HuggingFaceEmbeddings(
#     model_name="allenai/scibert_scivocab_uncased",
#     encode_kwargs=encode_kwargs,
# )
print("Embedding model loaded")



def process_line(name, i, line, embed_path):
    # if f"{name}_{i}.npy" in os.listdir(embed_path):
    #     return
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
num_cores = 10
num_threads = 2
# num_processes = 20
work_jsonls = []
print(f"Processing {len(files)} files")
print(f"First file: {files[0]}")
if direction == -1:
    print("Reversing order")
for ix, name in enumerate(tqdm(files[::direction])):
    # print(f"\nProcessing {name} ({ix+1}/{len(files)})")
    with open(data_path + f"/{name}") as f:
        work_jsonls = f.readlines()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_processes
        ) as executor:
            for i, line in enumerate(tqdm(work_jsonls)):
                save_name = name.split(".")[0].split("_")[-2:]
                save_name = "_".join(save_name)
                if Path(f"{embed_path}/{save_name}_{i}.npy").exists() :
                    continue
                line = orjson.loads(line)
                results.append(executor.submit(process_line, name, i, line, embed_path))
                if len(results) == num_processes:
                    for future in concurrent.futures.as_completed(results):
                        future.result()
                    results = []
            for future in concurrent.futures.as_completed(results):
                future.result()
                results = []
print("Done!")
