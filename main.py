import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
if not os.path.exists("./parking.parquet"):
    df = pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', dtype=str, header=0)
    df.to_parquet('parking.parquet')
if not os.path.exists("./parking.hdf5"):
    with pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', dtype=str, header=0, chunksize=100000) as reader:
        for i, chunk in enumerate(reader):
            chunk.to_hdf("parking.hdf5", key="chunk_" + str(i), mode="a")
chunked = False
if chunked:
    df_list = pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', dtype=str, header=0, chunksize=100000)
else:
    df_list = [pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', dtype=str, header=0)]
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
for i in df_list:
    #print(i.head())
    plates = i["Vehicle Make"].value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("vehicle_make.png")
    plt.clf()
    plates = i["Vehicle Body Type"].value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("bodytype.png")
    plt.clf()
    plates = i["Vehicle Color"].value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("color.png")
    plt.clf()
    plates = i["Vehicle Year"].value_counts()[1:21]
    print(plates.head())
    idx = np.argsort(plates.index)
    plt.bar(plates.index[idx], plates[idx])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("year.png")
    plt.clf()
    plates = i["Violation County"].value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("county.png")
    plt.clf()
    plates = i["Issue Date"].value_counts()
    print(plates.head())

    idx = np.argsort([int(x.split("/")[2]) * 10000000 + int(x.split("/")[0]) * 1000 + int(x.split("/")[1]) for x in plates.index])
    plt.bar(plates.index[idx], plates[idx])
    plt.yscale("log")
    plt.xticks(plates.index[idx], rotation=90)
    #plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=10)
    plt.tight_layout()
    plt.savefig("date.png")
    plt.clf()
    plates = i["Plate ID"].value_counts()[1:21]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("plates.png")
    plt.show()
    # print(i["Vehicle Make"].value_counts())
    print("done")
    quit()
