import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_parquet("./augmented_data/full_coords.parquet")
data.fillna(0, inplace=True)
data = data.apply(lambda x: pd.to_numeric(x, errors="coerce"))
data.dropna(axis=1, inplace=True)
data = data[(data["lat"].astype(float) != 0.) & (data["long"].astype(float) != 0.)]
km = KMeans(n_clusters=5)
km.fit(data)
cents = km.cluster_centers_[:, -2:]

for val, df in data.groupby("Issuer Precinct"):

    plt.scatter(-df["lat"][df["lat"] > 0], df["long"][df["long"] > 0], s=1,  alpha=(1./255.))#c="orange",

plt.title("Heatmap of parking tickets in NYC, by coordinates")
plt.xlabel("Latitude")
plt.ylabel("Longitude")

plt.scatter(-cents[:,0], cents[:, 1], c="r", s=20, marker="o")

plt.savefig("coords_data_precincts_centers.png")
#apparently, location does NOT affect clustering
print()