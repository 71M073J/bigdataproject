import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import plotly.express as px
import dask.dataframe as dd


join_dfs = False
if join_dfs:
    business = pd.concat([pd.read_parquet(os.path.join("./augmented_data/with_business_dask.parquet/", x)) for x in
                          os.listdir("./augmented_data/with_business_dask.parquet/")], axis=0, ignore_index=True)
    events = pd.concat([pd.read_parquet(os.path.join("./augmented_data/with_events.parquet/", x)) for x in
                        os.listdir("./augmented_data/with_events.parquet/")], axis=0, ignore_index=True)
    schools = pd.concat([pd.read_parquet(os.path.join("./augmented_data/with_schools_dask.parquet/", x)) for x in
                         os.listdir("./augmented_data/with_schools_dask.parquet/")], axis=0, ignore_index=True)
    weather = pd.read_parquet("./augmented_data/with_weather_dask.parquet")
    # centerlines = pd.read_pickle("./augmented_data/processed_centerlines.pkl")

    # cols = ['Issue Date', 'Issuer Precinct', 'Violation Time', 'Street Name', 'datetime_issue',
    # 'datetime', 'temp', 'humidity', 'snowdepth', 'windspeed', 'conditions', 'description']
    # weather.columns = cols
    business["Summons Number"] = business["Summons Number"].astype(str)
    events["Summons Number"] = events["Summons Number"].astype(str)
    schools["Summons Number"] = schools["Summons Number"].astype(str)
    weather["Summons Number"] = weather["Summons Number"].astype(str)
    print("business\n", business.columns, "events\n", events.columns, "schools\n", schools.columns, "weather\n",
          weather.columns)
    business.drop(labels=["Issue Date", "Issuer Precinct", "Violation Time", "Street Name"], inplace=True, axis=1)
    events.drop(labels=["Issue Date", "Issuer Precinct", "Violation Time", "Street Name"], inplace=True, axis=1)
    schools.drop(labels=["Issue Date", "Issuer Precinct", "Violation Time", "Street Name"], inplace=True, axis=1)

    print("business\n", business.columns, "events\n", events.columns, "schools\n", schools.columns, "weather\n",
          weather.columns)
    # quit()
    import functools as ft

    full = ft.reduce(lambda left, right: pd.merge(left, right, on="Summons Number"),
                     [business, events, schools, weather])
    print(full.columns)
    full["Schools"].fillna(0, inplace=True)
    full.dropna()
    full.to_parquet("./augmented_data/full.parquet")
    quit()

edit_streets = False
if edit_streets:
    data = dd.read_parquet("./augmented_data/full_w_housenum.parquet")
    count = 0


    def process_names(x):
        global count
        if x is None:
            return x
        lx = x.upper().replace(" DRIVE", " DR").replace(" DRV", " DR") \
            .replace(" PLACE", " PL").replace(" AVENUE", " AVE").replace(" STREET", " ST") \
            .replace("1ST", "1").replace("2ND", "2").replace("3RD", "3").replace("4TH", "4").replace("5TH", "5") \
            .replace("6TH", "6").replace("7TH", "7").replace("8TH", "8").replace("9TH", "9").replace("0TH", "0")
        count += 1
        if count % 100000 == 0:
            print(count)

        return lx


    data["Street Name"] = data["Street Name"].apply(process_names)
    data.compute()
    data.to_parquet("./augmented_data/full3.parquet")
    quit()


aaa = True
if aaa:
    def cleanup(x):
        if str(x).lower() == "nan":
            return 0
        else:
            return int(str(x).split(".")[0].split("-")[0])
    centerlines = dd.read_csv("./augmented_data/processed_centerlines.csv")
    centerlines["L_LOW_HN"] = centerlines["L_LOW_HN"].apply(cleanup)
    centerlines["L_HIGH_HN"] = centerlines["L_HIGH_HN"].apply(cleanup)
    centerlines["R_LOW_HN"] = centerlines["R_LOW_HN"].apply(cleanup)
    centerlines["R_HIGH_HN"] = centerlines["R_HIGH_HN"].apply(cleanup)
    centerlines.compute()
    linedict = {}
    print("building dict")
    for i in range(len(centerlines)):
        if centerlines["ST_LABEL"].iloc[i] in linedict:
            linedict[centerlines["ST_LABEL"].iloc[i]].append(centerlines.iloc[i])
        else:
            linedict[centerlines["ST_LABEL"].iloc[i]] = [centerlines.iloc[i]]


    print("reading data")
    data = dd.read_parquet("./augmented_data/full3.parquet")
    data["House Number"].fillna(0, inplace=True)
    lat = np.zeros(len(data), dtype=np.float32)
    long = np.zeros(len(data), dtype=np.float32)


    for i in range(len(data)):
        x = data.iloc[i]
        if i % 100000 == 0:
            print(i)
        st = x["Street Name"]
        if st in linedict:
            try:
                hn = int(x["House Number"])
            except ValueError:
                continue
            for line in linedict[st]:
                if min(int(line["L_LOW_HN"]), int(line["R_LOW_HN"])) <= hn <= max(int(line["L_HIGH_HN"]), int(line["R_HIGH_HN"])):
                    linestr = line["averaged_long_lat"][2:-2].split(", ")
                    lat[i] = float(linestr[0])
                    long[i] = float(linestr[1])
    data.compute()
    data["lat"] = lat
    data["long"] = long
    data.to_parquet("./augmented_data/full_coords.parquet")
    print(len(data))
    print(data.iloc[1])

    timetemp = False
    if timetemp:
        def get_timeofday(x):
            if x is None:
                return 0
            try:
                if len(x) == 5:
                    if (-1 < int(x[0]) < 3) and (-1 < int(x[1]) < 10) and (-1 < int(x[2]) < 7) and (
                            -1 < int(x[3]) < 10) and (x[4] in ["A", "P"]):
                        return ((x[-1] == "P") and (int(x[:2]) < 12)) * 12 * 60 + int(x[:2]) * 60 + int(x[2:4])
            except:
                return 0
            return 0


        plt.title("Correlation between time of day and temperature")
        plt.ylabel("Temp (C)")
        plt.xlabel("Time of day in minutes since midnight")
        plt.tight_layout()
        plt.savefig("timeofday_vs_temp.png")
quit()

#Task 6?=


data = pd.read_parquet("./augmented_data/full_coords.parquet")
data = data.groupby("Issuer Precinct")










save_files = False
if save_files:
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
    df_list = pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv',  header=0)
    print(len(df_list))
    df_list = df_list.dropna()
    print(len(df_list))

fix_order = False
if fix_order:
    if chunked:
        df_list = next(df_list)
    df_list["Issue Date"] = pd.to_datetime(df_list["Issue Date"])
    df_list = df_list.sort_values("Issue Date")
    #df_list.to_parquet('parking.parquet')
    df_list.to_csv("Parking_Violations_Issued_-_Fiscal_Year_2023.csv", index=False)
    quit()
clustering = False
if clustering:
    ...
    #k-means
    km = KMeans()
    preds = km.fit_predict(df_list[0].to_numpy())


#pd.set_option('display.max_rows', 500)
#pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
for i in df_list:
    print(i.head())
    if False:
        plates = i["Registration State"].value_counts().rename_axis('unique_values').reset_index(name='counts')
        plates["logs"] = np.log2(plates["counts"])
        print("plotting...")
        fig = px.choropleth(plates,
                            locations='unique_values',
                            locationmode="USA-states",
                            scope="usa",
                            color='logs',
                            range_color=[8, 24],
                            color_continuous_scale="Viridis",
                            hover_name="unique_values",
                            hover_data="counts"
                            )
        #fig.colo
        fig.show()
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

    plt.clf()
    plates = i["Issuer Code"].value_counts()[1:21]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("issuers.png")
    plt.show()
    plt.show()
    # print(i["Vehicle Make"].value_counts())
    print("done")
    quit()
