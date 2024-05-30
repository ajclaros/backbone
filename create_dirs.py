import os
import pandas as pd
import pathlib

# find all folders in the current directory
folders = [f for f in os.listdir(".") if os.path.isdir(f)]
disciplines = pd.read_csv("disciplines.csv")
years = disciplines.year
disciplines = disciplines.columns[:-2]
years = [str(year) for year in years if year > 0]
for year in year:
    for discipline in disciplines:
        path = f"{discipline}_{year}"
