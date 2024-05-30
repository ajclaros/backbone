import os
import jsonlines
import json
import re
import argparse

"""
This script takes the all files from data_locations and extracts the specific lines from the json lines file
located in the folder_year. The purpose is to create a subst of the original data from the original json lines
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="Computer Science", help="Domain to aggregate")
parser.add_argument("-y", type=str, default="5 6 7 8 9", help="Years to aggregate")
args = parser.parse_args()
domain = args.d
years = [int(y) for y in args.y.split(" ")]

def process_text(text):
    cite_num = 0
    formula_num = 0
    ix_curly = [
        (m.start(), m.end(), text[m.start() : m.end()])
        for m in re.finditer("{{(.+?)}}", text)
    ]
    for i, c in enumerate(ix_curly):
        curly_type = formula_or_citation(c[2])
        curly_str = f" {curly_type}_{i} "
        if curly_type == "formula":
            formula_num += 1
        elif curly_type == "citation":
            cite_num += 1
        if curly_type != "None":
            text = text.replace(c[2], curly_str)
    return text


def folder_year_from_filename(filename):
    return filename.split("_")[2][:2]


def find_curly_braces(text):
    # curly = re.findall('{{(.+?)}}', text)
    # return occurrences of text in curly braces and their indices
    ix_curly = [(m.start(), m.end()) for m in re.finditer("{{(.+?)}}", text)]
    # for each occurence, identify if it is a formula or a citation
    for i, c in enumerate(ix_curly):
        curly = text[c[0] : c[1]]
        curly_type = formula_or_citation(curly)
        ix_curly[i] = (c[0], c[1], curly, curly_type)
    return ix_curly


def formula_or_citation(text):
    return "formula" if "formula" in text else "citation" if "cite" in text else "None"


# domain = "Physics"
# years = [5, 6, 7, 8, 9, 10]
domain = domain.replace(" ", "_")
with open(
    f"{domain.lower()}_{years[0]}_{years[-1]}/data_locations.json", "r"
) as f:
    data_locations = json.load(f)
filenames = list(data_locations.keys())

for i, filename in enumerate(filenames):
    folder_year = folder_year_from_filename(filename)
    print(f"Processing {filename} ({i+1}/{len(filenames)})")
    documents = []
    with jsonlines.open(f"../{folder_year}/{filename}") as f:
        print(f"Total number of documents: {len(data_locations[filename])}")
        lines = []
        for j, line in enumerate(f):
            if j in data_locations[filename]:
                metadata = {
                    "paper_id": line["paper_id"],
                    "body_text": [],
                }
                for k, section in enumerate(line["body_text"]):
                    if section["content_type"] != "paragraph":
                        continue
                    body_text = {}
                    body_text["section"] = section["section"]
                    body_text["sec_number"] = section["sec_number"]
                    body_text["sec_type"] = section["content_type"]
                    body_text["text"] = section["text"]
                    metadata["body_text"].append(body_text)
                lines.append(metadata)
        with jsonlines.open(
            f"{domain.lower()}_{years[0]}_{years[-1]}/data/{filename}",
            "w",
        ) as f:
            f.write_all(lines)
