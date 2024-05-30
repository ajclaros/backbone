import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import orjson
from utils import str_year

disciplines = pd.read_csv("disciplines.csv")
# replace 1900 with 2000 manually

df = disciplines.iloc[1:, :]
pd.DataFrame()
years = df["year"].values.tolist()
years[years.index(1900)] = 2000
df.drop(columns={"filename", "year"}, inplace=True)
# df.set_index("year", inplace=True)
# Every column is a discipline, every row is a year
# Plot a histogram of the number of papers in each discipline, stacked on top of each other
# overlay with an alpha=0.5 histogram of the number of papers in all disciplines

# plot the percentage of papers in each discipline over time
fig, ax = plt.subplots()
for i, dis in enumerate(df.T.iterrows()):
    # print(dis)
    values = dis[1].values / sum(dis[1].values)
    ax.scatter(years, values, label=dis[0])
    # ax.bar(dis[1].index, dist, label=dis[0]) # papers in each discipline as a fraction of total papers in that year
    # ax.bar(dis[1].index, dis[1].values, label=dis[0])
    # ax.fill_between(dis[1].index, dis[1].values, alpha=0.5)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Percentage of Papers")
plt.title("Percentage of Papers in Each Discipline Over Time")
plt.savefig("../plots/disciplines_percentage.png")
# ax.set_xticklabels([int(t.get_text()) for t in tick_labels])

plt.show()

# plot the number of papers in each discipline over time
fig, ax = plt.subplots()
for i, dis in enumerate(df.T.iterrows()):
    ax.scatter(years, dis[1].values, label=dis[0])
plt.legend()
plt.xlabel("Year")
plt.ylabel("Number of Papers Added")
plt.title("Number of Papers in Each Discipline Each Year")
plt.savefig("../plots/disciplines_number.png")
plt.show()

# plot the change in number of papers in each discipline over time
fig, ax = plt.subplots()
for i, dis in enumerate(df.T.iterrows()):
    ax.scatter(years, dis[1].diff().values, label=dis[0])
plt.legend()
plt.xlabel("Year")
plt.ylabel("Change in Number of Papers")
plt.title("Change in Number of Papers in Each Discipline Each Year")
plt.savefig("../plots/disciplines_change.png")
plt.show()
