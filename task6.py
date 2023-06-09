import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
set_alpha = False
if set_alpha:
    im = cv2.imread("coords_data_precincts.png")
    rgba = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    print()
    mr = (im[:, :, 0] == 255)
    mg = (im[:, :, 1] == 255)
    mb = (im[:, :, 2] == 255)

    #rgba = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = 255 * (1 - (mr + mg + mb))
    cv2.imwrite("alpha.png", rgba)
    quit()











data = pd.read_parquet("./augmented_data/full_coords.parquet").sort_values(by="Issue Date")
data["Issue Date"] = pd.to_datetime(data["Issue Date"])
mask = (data['Issue Date'] > '2022-6-1') & (data['Issue Date'] <= '2023-5-15')
x = data[["Issue Date", "Start Date"]].loc[mask].groupby("Start Date").count()

for val, df in data.groupby("Issuer Precinct"):

    plt.scatter(-df["lat"][df["lat"] > 0], df["long"][df["long"] > 0], s=1,  alpha=(1./255.))#c="orange",

plt.title("Heatmap of parking tickets in NYC, by coordinates")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.savefig("coords_data_precincts.png")
plt.clf()
plt.scatter(-data["lat"][data["lat"] > 0], data["long"][data["long"] > 0], s=1, c="orange", alpha=(1./255.))#c="orange",

plt.title("Heatmap of parking tickets in NYC, by coordinates")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.savefig("coords_data.png")

print(x)
plt.plot(x.index,x)
plt.xticks(rotation=90)
plt.locator_params(axis='both', nbins=7)
plt.tight_layout()
plt.show()
#datas = [(x, val) for val, x in data.groupby("Issuer Precinct") if len(x) > 1000]
#print(datas[4][0].columns, datas[4][1])