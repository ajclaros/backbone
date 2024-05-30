import pandas as pd
import os
import jsonlines
import json
import re
import concurrent.futures


domain = "Physics"
years = [5, 6, 7, 8, 9]


def folder_year_from_filename(filename):
    return filename.split("_")[2][:2]


# csvs = [f for f in os.listdir() if f.endswith('.csv')
def process_data(domain, years):
    # loads metadata csvs
    csvs = [f for f in os.listdir() if f.endswith(".csv") and "meta" in f]
    data_locations = {}
    for csv in csvs:
        csv_year = int(csv.split(".")[0].split("_")[1])
        if csv_year not in years:
            continue
        df = pd.read_csv(csv)
        df.loc[df["discipline"] == domain][["filename", "file_index"]]
        groups = df.groupby("filename")["file_index"].apply(list).to_dict()
        data_locations.update(groups)

    # save data_locations to json
    full_years = []
    for year in years:
        if year < 23:
            full_years.append(2000 + year)
        else:
            full_years.append(1900 + year)

    # create folder for aggregated data
    if not os.path.exists(f"aggregated_{domain.lower()}_{years[0]}_{years[-1]}"):
        os.mkdir(f"aggregated_{domain.lower()}_{years[0]}_{years[-1]}")
    # save data_locations to json
    with open(
        f"./aggregated_{domain.lower()}_{years[0]}_{years[-1]}/data_locations.json", "w"
    ) as f:
        json.dump(data_locations, f)


def process_file(filename, folder_year, data_locations, ix):
    dataset = []
    print(f"\nProcessing: {ix}")

    with jsonlines.open(f"../{folder_year}/{filename}") as f:
        for i, line in enumerate(f):
            if i in data_locations[filename]:
                body_text = []
                for j, section in enumerate(line["body_text"]):
                    if section["content_type"] != "paragraph":
                        continue
                    body_text.append(
                        {
                            "section": section["section"],
                            "sec_number": section["sec_number"],
                            "sec_type": section["sec_type"],
                            "content_type": section["content_type"],
                            "text": section["text"],
                        }
                    )
                dataset.append({"paper_id": line["paper_id"], "body_text": body_text})
    print(f"done: {ix} ")

    return dataset, filename


def write_to_jsonl(dataset, filename):
    print(f"\nWriting {filename} to jsonl")
    with jsonlines.open(
        f"aggregated_{domain.lower()}_{years[0]}_{years[-1]}/data/{filename}",
        "w",
    ) as f:
        f.write_all(dataset)


def aggregate_data(domain, years, threads=4):
    # given a domain, years, and data_locations file, aggregate data
    with open(
        f"aggregated_{domain.lower()}_{years[0]}_{years[-1]}/data_locations.json", "r"
    ) as f:
        data_locations = json.load(f)
    filenames = list(data_locations.keys())
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        for i, filename in enumerate(filenames):
            results.append(
                executor.submit(
                    process_file,
                    filename,
                    folder_year_from_filename(filename),
                    data_locations,
                    i,
                )
            )
            if len(results) == threads:
                write_tasks = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=threads
                ) as executor1:
                    for future in concurrent.futures.as_completed(results):
                        dataset, filename = future.result()
                        write_task = executor1.submit(write_to_jsonl, dataset, filename)
                        write_tasks.append(write_task)
                concurrent.futures.wait(write_tasks)
                results = []

        if results:
            write_tasks = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=threads
            ) as executor1:
                for future in concurrent.futures.as_completed(results):
                    dataset, filename = future.result()
                    write_task = executor1.submit(write_to_jsonl, dataset, filename)
                    write_tasks.append(write_task)
            concurrent.futures.wait(write_tasks)

process_data(domain, years)
aggregate_data(domain, years, threads=30)
