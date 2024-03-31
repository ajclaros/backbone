import os
import jsonlines
import pandas as pd
import pyalex
from pyalex import Works
import concurrent.futures

current_folder_name = os.path.basename(os.getcwd())

json_files = [
    pos_json for pos_json in os.listdir(foldername) if pos_json.endswith(".jsonl")
]
folders = os.listdir("..")
folders = [x for x in folders if x.isdigit()]


def get_discipline(input_path) -> list:
    data = []
    with jsonlines.open(input_path) as f:
        for line in f.iter():
            data.append(line["metadata"]["categories"])
    return data


def load_jsonl(input_path) -> list:
    data = []
    with jsonlines.open(input_path) as f:
        for line in f.iter():
            data.append(line)
    return data


def load_jsonl_metadata(folder, filename) -> list:
    data = []
    with jsonlines.open(f"../{folder}/{filename}") as f:
        for i, line in enumerate(f.iter()):
            filtered = {
                key: line[key]
                for key in line.keys()
                if key
                not in [
                    "body_text",
                    "bib_entries",
                    "ref_entries",
                    "abstract",
                    "metadata",
                ]
            }
            filtered["folder"] = folder
            filtered["filename"] = filename
            filtered["file_index"] = i
            for key in line["metadata"].keys():
                filtered[key + "_meta"] = line["metadata"][key]
            data.append(filtered)
    # print(f"Loaded {len(data)} records from {input_path}")
    return data


def load_from_discipline(input_path, discipline) -> list:
    data = set()
    with jsonlines.open(input_path) as f:
        for line in f.iter():
            if line["discipline"] == discipline:
                data.add(line["paper_id"])
    return data


def find_all_disciplines(input_path) -> set:
    data = set()
    with jsonlines.open(input_path) as f:
        for line in f.iter():
            data.add(line["discipline"])
    return data


def process_jsonl(folder, json_file):
    data = load_jsonl_metadata("../" + folder + "/", json_file)
    return data


# creating first level metadata csv
df = pd.DataFrame()
num_processes = 12
folders = sorted(folders)
for folder in folders:
    df = pd.DataFrame()
    print()
    print(f"Processing {folder}")
    json_files = [
        pos_json
        for pos_json in os.listdir(f"../{folder}")
        if pos_json.endswith(".jsonl")
    ]
    print(f"{len(json_files)} files in folder")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        dfl = []
        for i, json_file in enumerate(json_files):
            print(f"{i}", end=" ")
            if i % 50 == 0 and i > 0:
                print()
            results.append(executor.submit(process_jsonl, folder, json_file))
            if len(results) == num_processes:
                for future in concurrent.futures.as_completed(results):
                    data = future.result()
                    dfl.append(pd.DataFrame(data))
                results = []
        for future in concurrent.futures.as_completed(results):
            dfl.append(pd.DataFrame(future.result()))
            # df = pd.concat([df, pd.DataFrame(data)])
        df = pd.concat(dfl)
        print(f"Saving as file: metadata_{folder}.csv")
        df.to_csv(f"../{current_folder_name}/metadata_{folder}.csv", index=False)
        results = []
