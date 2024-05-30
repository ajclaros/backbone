import os
import numpy as np
import pandas as pd
from utils import process_files
import argparse
import sys
import importlib

importlib.reload(sys.modules["utils"])

print("Building dataframe ...")
parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="Physics", help="Domain to aggregate")
parser.add_argument("-y", type=str, default="5 6 7 8 9", help="Years to aggregate")
args = parser.parse_args()
domain = args.d
years = [int(y) for y in args.y.split(" ")]
columns = [
    "paper_id",
    "section",
    "sec_number",
    "text",
    "embedding",
]

process_files(years, -1, columns, domain=domain, save=True, max_workers=30)
print("Dataframe built")
