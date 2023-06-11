import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import plotly.express as px
import dask.dataframe as dd
import pyarrow.parquet as pq

import json
import matplotlib.cm as cm

newvis = True

if newvis:
    data = pd.read_parquet("./augmented_data/full_tickettypes.parquet")[["Issue Date", "temp"]]
    data["Issue Date"] = pd.to_datetime(data["Issue Date"])
    mask = (data['Issue Date'] > '2022-6-1') & (data['Issue Date'] <= '2023-5-15')
    data = data[["Issue Date", "temp"]].loc[mask]
    for val, df in data.groupby("Issue Date"):
        plt.scatter(df["temp"].iloc[0],len(df), )
    plt.title("Temperature vs daily number of tickets")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Number of daily tickets")
    plt.savefig("temp_vs_num_tickets.png")
    plt.show()
    quit()
    #data = pq.ParquetFile("./augmented_data/full_tickettypes.parquet")
    #for batch in data.iter_batches():
    #    batch = batch.to_pandas()[["Issue Date", "temp"]]
    #    for val, df in batch.groupby("Issue Date"):
    #        plt.scatter(len(df), df["temp"].iloc[0])









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

advanced_visualisations = False
if advanced_visualisations:
    with open("asda.json", "rb") as f:
        polys = json.load(f)
    dataset = pq.ParquetFile("./augmented_data/full_coords.parquet")
    polygons = {}
    for i in polys["features"]:
        borough = i["properties"]["borough"]
        poly = [np.array(x) for x in i["geometry"]["coordinates"]]
        if borough not in polygons:
            polygons[borough] = [poly]
        else:
            polygons[borough].append(poly)


    def borough_to_code(x):
        if x == 'MANHATTAN':
            return 1
        elif x == "BRONX":
            return 2
        elif x == "BROOKLYN":
            return 3
        elif x == " QUEENS":
            return 4
        else:
            return 5
    def county_to_borough(x):
        if (x == 'K') | (x.upper() == 'KINGS') | (x == 'BK'):
            return 'BROOKLYN'
        elif (x == 'BX') | (x.upper() == 'BRONX'):
            return 'BRONX'
        elif (x == 'R') | (x.upper() == 'RICH') | (x == 'ST') | (x.upper() == 'RICHM'):
            return 'STATEN ISLAND'
        elif (x == 'NY') | (x == 'MN') | (x == "MS"):
            return 'MANHATTAN'
        elif (x == 'Q') | (x == 'QN') | (x.upper() == 'QNS'):
            return 'QUEENS'
        else:
            return x
    addressd = {}
    def count_viols(x):
        if x in addressd:
            addressd[x] += 1.
        else:
            addressd[x] = 1.

    for i, batch in enumerate(dataset.iter_batches()):
        batch = batch.to_pandas()
        batch["Address Borough"] = batch["Address Borough"].apply(lambda x: "MANHATTAN" if x == "MS" else ("QUEENS" if x == "QUEEN" else ("BRONX" if x == "ABX" else x)))
        batch["Address Borough"].apply(count_viols)
        h = int(i * 2**16/130000)
        if h % 10 == 0:
            print(h, "%")
        #batch["Violation County"] = batch["Violation County"].apply(county_to_borough)


    surface = {"QUEENS":280., "BRONX":110.,"STATEN ISLAND": 152., "MANHATTAN": 59.1, "BROOKLYN": 180.}
    ns = {x:addressd[x]/surface[x] for x in addressd if x != "nan"}
    mx = max([ns[x] for x in ns])
    ns2 = {x: ns[x]/mx for x in ns}
    print(ns, ns2, addressd)
    for borough in polygons:
        for shape in polygons[borough]:
            for s in shape:
                plt.fill(s[:, 0], s[:, 1], c=cm.plasma(ns2[borough.upper()]))
    from matplotlib.lines import Line2D
    colors = [cm.plasma(ns2[borough.upper()]) for borough in list(polygons)]
    custom_lines = [Line2D([0], [0], color=colors[3], lw=4),
                    Line2D([0], [0], color=colors[4], lw=4),
                    Line2D([0], [0], color=colors[1], lw=4),
                    Line2D([0], [0], color=colors[0], lw=4),
                    Line2D([0], [0], color=colors[2], lw=4)]


    plt.legend(custom_lines, [f"{x}:{int(ns[x]*10)/10} Tickets/km²" for x in ns])
    plt.title("Heatmap of tickets over boroughs, normalized for surface area")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.tight_layout()
    plt.savefig("ticket_heatmap_boroughs.png")
    plt.show()



advanced_visualisations2 = True
if advanced_visualisations2:
    data = None
    if not os.path.exists("./augmented_data/full_tickettypes.parquet"):
        data = pd.read_parquet("parking.parquet")[["Summons Number", "Violation Code"]]
        data["Summons Number"] = data["Summons Number"].astype(str)
        data2 = pd.read_parquet("./augmented_data/full3.parquet")
        data2["Summons Number"] = data2["Summons Number"].astype(str)
        data2 = data2.merge(data, on="Summons Number")
        data2["Address Borough"] = data2["Address Borough"].apply(
            lambda x: "MANHATTAN" if x == "MS" else ("QUEENS" if x == "QUEEN" else ("BRONX" if x == "ABX" else x)))

        data2.to_parquet("./augmented_data/full_tickettypes.parquet")
    data = pq.ParquetFile("./augmented_data/full_tickettypes.parquet")
    prices = pd.read_csv("ticket_code_to_fine.csv", dtype={"VIOLATION CODE":str, "VIOLATION DESCRIPTION":str, "96th":int,"OTHER STREETS":int})
    pr = {}
    for i in range(len(prices)):
        x = prices.iloc[i]
        pr[x["VIOLATION CODE"]] = x["OTHER STREETS"]
    polygons = {}

    with open("asda.json", "rb") as f:
        polys = json.load(f)
    for i in polys["features"]:
        borough = i["properties"]["borough"]
        poly = [np.array(x) for x in i["geometry"]["coordinates"]]
        if borough not in polygons:
            polygons[borough] = [poly]
        else:
            polygons[borough].append(poly)
    boroughs = {}
    per_type = {}
    def get_prices(x):
        b = x["Address Borough"]
        if b in boroughs:
            xv = x["Violation Code"]
            if xv in pr:
                boroughs[b] += int(pr[xv])
                per_type[xv] += 1
        else:
            xv = x["Violation Code"]
            boroughs[b] = int(pr[xv])
            per_type[xv] = 1

    for i, batch in enumerate(data.iter_batches()):
        batch = batch.to_pandas()[["Address Borough", "Violation Code"]]
        batch = batch.apply(get_prices, axis=1)
        h = int(i * 2**16/130000)
        if h % 10 == 0:
            print(h, "%")


    surface = {"QUEENS":280., "BRONX":110.,"STATEN ISLAND": 152., "MANHATTAN": 59.1, "BROOKLYN": 180.}
    ns = {x:boroughs[x] for x in boroughs if x != "nan"}
    mx = max([ns[x] for x in ns])
    ns_normalised = {x: ns[x]/mx for x in ns}
    print(ns, ns_normalised, boroughs)
    for borough in polygons:
        for shape in polygons[borough]:
            for s in shape:
                plt.fill(s[:, 0], s[:, 1], c=cm.plasma(ns_normalised[borough.upper()]))
    from matplotlib.lines import Line2D
    colors = [cm.plasma(ns_normalised[borough]) for borough in list(ns_normalised)]
    custom_lines = [Line2D([0], [0], color=colors[x], lw=4) for x in range(5)]


    plt.legend(custom_lines, [f"{x}:{int(ns[x]*10)/10} $" for x in ns])
    plt.title("Heatmap of total price paid by each borough")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.tight_layout()
    plt.savefig("boroughs_total_ticket_prices.png")
    plt.show()

    surface = {"QUEENS":280., "BRONX":110.,"STATEN ISLAND": 152., "MANHATTAN": 59.1, "BROOKLYN": 180.}
    ns = {x:boroughs[x]/surface[x] for x in boroughs if x != "nan"}
    mx = max([ns[x] for x in ns])
    ns_normalised = {x: ns[x]/mx for x in ns}
    print(ns, ns_normalised, boroughs)
    for borough in polygons:
        for shape in polygons[borough]:
            for s in shape:
                plt.fill(s[:, 0], s[:, 1], c=cm.plasma(ns_normalised[borough.upper()]))
    from matplotlib.lines import Line2D
    colors = [cm.plasma(ns_normalised[borough]) for borough in list(ns_normalised)]
    custom_lines = [Line2D([0], [0], color=colors[x], lw=4) for x in range(5)]


    plt.legend(custom_lines, [f"{x}:{int(ns[x]*10)/10} $/km²" for x in ns])
    plt.title("Heatmap of normalised price by surface area, paid by each borough")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.tight_layout()
    plt.savefig("boroughs_total_ticket_prices_normalised.png")
    plt.show()

    population = {"QUEENS":2270976, "BRONX":1427056,"STATEN ISLAND": 475596, "MANHATTAN":1629153, "BROOKLYN": 2576771}
    ns = {x:boroughs[x]/population[x] for x in boroughs if x != "nan"}
    mx = max([ns[x] for x in ns])
    ns_normalised = {x: ns[x]/mx for x in ns}
    print(ns, ns_normalised, boroughs)
    for borough in polygons:
        for shape in polygons[borough]:
            for s in shape:
                plt.fill(s[:, 0], s[:, 1], c=cm.plasma(ns_normalised[borough.upper()]))
    from matplotlib.lines import Line2D
    colors = [cm.plasma(ns_normalised[borough]) for borough in list(ns_normalised)]
    custom_lines = [Line2D([0], [0], color=colors[x], lw=4) for x in range(5)]


    plt.legend(custom_lines, [f"{x}:{int(ns[x]*10)/10} $/person" for x in ns])
    plt.title("Heatmap of normalised price-per-person, paid by each borough")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.tight_layout()
    plt.savefig("boroughs_total_ticket_prices_per_people.png")
    plt.show()









aaa = False
if aaa:
    def cleanup(x):
        if str(x).lower() == "nan":
            return 0
        else:
            return int(str(x).split(".")[0].split("-")[0])


    centerlines = pd.read_csv("./augmented_data/processed_centerlines.csv")
    centerlines["L_LOW_HN"] = centerlines["L_LOW_HN"].apply(cleanup)
    centerlines["L_HIGH_HN"] = centerlines["L_HIGH_HN"].apply(cleanup)
    centerlines["R_LOW_HN"] = centerlines["R_LOW_HN"].apply(cleanup)
    centerlines["R_HIGH_HN"] = centerlines["R_HIGH_HN"].apply(cleanup)
    linedict = {}
    print("building dict")
    for i in range(len(centerlines)):
        if centerlines["ST_LABEL"].iloc[i] in linedict:
            linedict[centerlines["ST_LABEL"].iloc[i]].append(centerlines.iloc[i])
        else:
            linedict[centerlines["ST_LABEL"].iloc[i]] = [centerlines.iloc[i]]

    print("reading data")
    data = pd.read_parquet("./augmented_data/full3.parquet")
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
                if min(int(line["L_LOW_HN"]), int(line["R_LOW_HN"])) <= hn <= max(int(line["L_HIGH_HN"]),
                                                                                  int(line["R_HIGH_HN"])):
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


# Task 6?=


#data = pd.read_parquet("./augmented_data/full_coords.parquet")
#data = data.groupby("Issuer Precinct")

save_files = False
if save_files:
    if not os.path.exists("./parking.parquet"):
        df = pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', dtype=str, header=0)
        df.to_parquet('parking.parquet')
    if not os.path.exists("./parking.hdf5"):
        with pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', dtype=str, header=0,
                         chunksize=100000) as reader:
            for i, chunk in enumerate(reader):
                chunk.to_hdf("parking.hdf5", key="chunk_" + str(i), mode="a")
chunked = False
if chunked:
    df_list = pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', dtype=str, header=0, chunksize=100000)
else:
    #df_list = pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', header=0)
    ...
    #print(len(df_list))
    #df_list = df_list.dropna()
    #print(len(df_list))


fix_order = False
if fix_order:
    if chunked:
        df_list = next(df_list)
    df_list["Issue Date"] = pd.to_datetime(df_list["Issue Date"])
    df_list = df_list.sort_values("Issue Date")
    # df_list.to_parquet('parking.parquet')
    df_list.to_csv("Parking_Violations_Issued_-_Fiscal_Year_2023.csv", index=False)
    quit()
clustering = False
if clustering:
    ...
    # k-means
    km = KMeans()
    preds = km.fit_predict(df_list[0].to_numpy())

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

def county_to_borough(x):
    if not x is None:
        if (x == 'K') | (x.upper() == 'KINGS') | (x == 'BK'):
            return 'BROOKLYN'
        elif (x == 'BX') | (x.upper() == 'BRONX') | (x.upper() == "ABX"):
            return 'BRONX'
        elif (x == 'R') | (x.upper() == 'RICH') | (x == 'ST') | (x.upper() == 'RICHM'):
            return 'STATEN ISLAND'
        elif (x == 'NY') | (x == 'MN') | (x == "MS"):
            return 'MANHATTAN'
        elif (x == 'Q') | (x == 'QN') | (x.upper() == 'QNS') | (x.upper() == "QUEEN"):
            return 'QUEENS'
        else:
            return x
    else:
        return x
#df_list = [pd.read_parquet("parking.parquet")]
df_list = [pd.read_csv("Parking_Violations_Issued_-_Fiscal_Year_2023.csv", dtype=str)]
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
                            hover_data="counts",
                            title="Distribution of violators' countries of registry"
                            )
        # fig.colo
        fig.show()
    plates = i["Vehicle Make"].value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.xlabel("Vehicle Maker")
    plt.ylabel("Number of such vehicles ticketed")
    plt.title("Number of vehicles of each brand ticketed")
    plt.tight_layout()
    plt.savefig("vehicle_make.png")
    plt.clf()
    plates = i["Vehicle Body Type"].value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.xlabel("Vehicle Body Type")
    plt.ylabel("Number of such vehicles ticketed")
    plt.title("Number of vehicles of each body type ticketed")
    plt.tight_layout()
    plt.savefig("bodytype.png")
    plt.clf()
    plates = i["Vehicle Color"].value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.xlabel("Vehicle Colour")
    plt.ylabel("Number of such vehicles ticketed")
    plt.title("Number of vehicles of each colour ticketed")
    plt.tight_layout()
    plt.savefig("color.png")
    plt.clf()
    plates = i["Vehicle Year"].value_counts()[1:21]
    print(plates.head())
    idx = np.argsort(plates.index)
    plt.bar(plates.index[idx], plates[idx])
    plt.xticks(rotation=90)
    plt.xlabel("Vehicle Year of Manufacture")
    plt.ylabel("Number of such vehicles ticketed")
    plt.title("Number of vehicles ticketed, per year of manufacture")
    plt.tight_layout()
    plt.savefig("year.png")
    plt.clf()
    plates = i["Violation County"].astype(str).apply(county_to_borough).value_counts()[:20]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.xlabel("County of Violation")
    plt.ylabel("Number of vehicles ticketed there")
    plt.title("Number of vehicles ticketed in each county")
    plt.tight_layout()
    plt.savefig("county.png")
    plt.clf()
    plates = i["Issue Date"].astype(str).dropna().value_counts()
    print(plates.head())

    idx = np.argsort(
        [int(x.split("/")[2]) * 10000000 + int(x.split("/")[0]) * 1000 + int(x.split("/")[1]) for x in plates.index])
    plt.bar(plates.index[idx], plates[idx])
    plt.yscale("log")
    plt.xticks(plates.index[idx], rotation=90)
    plt.xlabel("Date")
    plt.ylabel("Number of tickets that day")
    plt.title("Number of vehicles ticketed per day")
    # plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=10)
    plt.tight_layout()
    plt.savefig("date.png")
    plt.clf()
    plates = i["Plate ID"].value_counts()[1:21]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.xlabel("Vehicle ID")
    plt.ylabel("Total number of tickets for that Register plate")
    plt.title("Number tickets for each registered vehicle")
    plt.tight_layout()
    plt.savefig("plates.png")

    plt.clf()
    plates = i["Issuer Code"].value_counts()[1:21]
    print(plates.head())
    plt.bar(plates.index, plates)
    plt.xticks(rotation=90)
    plt.xlabel("Issuer Code")
    plt.ylabel("Total number of vehicles ticketed")
    plt.title("Number of tickets issued by each issuer")
    plt.tight_layout()
    plt.savefig("issuers.png")
    plt.show()
    plt.show()
    # print(i["Vehicle Make"].value_counts())
    print("done")
    quit()
