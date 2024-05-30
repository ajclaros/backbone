import numpy as np
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import re


filenames = [name for name in os.listdir() if name.endswith(".csv")]
disciplines = pd.read_csv("disciplines.csv")
categories = pd.read_csv("categories.csv")

# Filter, rows where year is in [2005, 2009]















