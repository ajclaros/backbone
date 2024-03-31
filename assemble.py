import pandas as pd
import jsonlines
import json
import os
import ast

csvs = [f for f in os.listdir() if f.endswith(".csv") and "meta" in f]
disciplines = {}
categories = {}
for i, csv in enumerate(csvs):
    print(f"Processing {csv}")
    df = pd.read_csv(csv)
    file_discipline = df["discipline"].value_counts()
    for discipline in file_discipline.index:
        if discipline in disciplines:
            disciplines[discipline].append(file_discipline[discipline])
        else:
            # disciplines[discipline] = [file_discipline[discipline]]
            disciplines[discipline] = [0] * i + [file_discipline[discipline]]
    for discipline in disciplines:
        if discipline not in file_discipline.index:
            disciplines[discipline].append(0)

    file_categories = df["categories_meta"].apply(
        lambda x: x.split(" ") if type(x) == str else "nan"
    )
    file_categories = file_categories.explode().value_counts()
    for category in file_categories.index:
        if category in categories:
            categories[category].append(file_categories[category])
        else:
            categories[category] = [0] * i + [file_categories[category]]
    for category in categories:
        if category not in file_categories.index:
            categories[category].append(0)
for discipline in disciplines:
    disciplines[discipline].append(sum(disciplines[discipline]))
for category in categories:
    categories[category].append(sum(categories[category]))
disciplines["filename"] = csvs + ["total"]
categories["filename"] = csvs + ["total"]

disciplines = pd.DataFrame(disciplines)
categories = pd.DataFrame(categories)
# add final row of total
# disciplines.to_csv("disciplines.csv", index=False)
# categories.to_csv("categories.csv", index=False)
# print(disciplines['filename'] = disciplines['filename'].apply(lambda x: x.split(".")[1].split("_")[0] if x != "total" else "total"))
disciplines = pd.read_csv("disciplines.csv")
disciplines["year"] = disciplines["filename"].apply(
    lambda x: int(x.split(".")[0].split("_")[1]) if x != "total" else -1
)
disciplines["year"] = disciplines["year"].apply(
    lambda x: 2000 + x if x < 24 and x > 0 else 1900 + x if x >= 0 else x
)
categories = pd.read_csv("categories.csv")
categories["year"] = categories["filename"].apply(
    lambda x: int(x.split(".")[0].split("_")[1]) if x != "total" else -1
)
categories["year"] = categories["year"].apply(
    lambda x: 2000 + x if x < 24 and x > 0 else 1900 + x if x >= 0 else x
)
disciplines.to_csv("disciplines.csv", index=False)
categories.to_csv("categories.csv", index=False)
