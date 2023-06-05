import pandas as pd
import json


import numpy as np
from dask import dataframe as dd
from distributed import Client
from dask_jobqueue import SLURMCluster


cluster = SLURMCluster(

            processes=8,
            cores=8,
            memory='16GB',
            scheduler_options={'dashboard_address': ':8087'},
            walltime="01:00:00"
          )

cluster.scale(jobs=2)
client = Client(cluster, timeout="120s")


#client = Client(cluster, timeout="120s")
cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time']
path = "/d/hpc/projects/FRI/bigdata/data/NYTickets/2023_april.csv"
df = dd.read_csv(path, assume_missing = True, usecols=cols,
dtype={
    'Issuer Precinct': np.float64,

})

df = df.repartition(npartitions=8)
# df2 = df2.repartition(npartitions=1)

print("read data")



def augment_weather(df: pd.DataFrame, year, parquet = False):
    tmp = df
    
    # transpose if reading parquet?
    if parquet:
        tmp = pd.DataFrame(df).transpose()
        tmp.columns = df.columns
    
    weather = pd.read_csv("./data/weather/ny_weather_{0}.csv".format(year))
    weather = weather[['datetime', 'temp', 'humidity', 'snowdepth', 'windspeed', 'conditions', 'description']]
    weather['snowdepth'].fillna(0, inplace=True)
    
    weather['datetime'] = pd.to_datetime(weather['datetime']).dt.date
    
    tmp['datetime'] = tmp['datetime_issue'].dt.date
    #print(tmp.columns)
    #print(weather.columns)
    res = tmp.merge(weather, on='datetime', how='left')
    #cols = [i for i in range(res.shape[1])]
    cols = ['Issue Date', 'Issuer Precinct', 'Violation Time', 'Street Name', 'datetime_issue', 
    'datetime', 'temp', 'humidity', 'snowdepth', 'windspeed', 'conditions', 'description']
    out = pd.DataFrame(res, columns = cols)
    return out

def get_datetime(date, time):
    time = str(time)
    if len(time) < 3:
        return 'nan'
    if len(time) < 5:
        #print(time)
        time += 'A'
    if len(time) == 4:
        try:
            time_h = int(float(time[:1]))
        except Exception as e:
            time_h = 0
        try:
            time_m = int(float(time[1:3]))
        except Exception as e:
            time_m = 0
        if time[3] == 'P':
            time_h += 12 
    else:
        try:
            time_h = int(float(time[:2]))
        except Exception as e:
            time_h = 0
        try:
            time_m = int(float(time[2:4]))
        except Exception as e:
            time_m = 0 
        if time[4] == 'P':
            time_h += 12 


    time_h = time_h%24
    d_dt = pd.to_datetime(date, format="%m/%d/%Y")
    d_dt = d_dt.replace(hour=time_h, minute = time_m)
#     d_dt.hour = time_h
#     d_dt.minute = time_m
    return d_dt
    #return date + " " + time[:2] + ":"+time[2:4]+time[4]+'M'



#data['datetime_issue'] = data['datetime_issue'].map_partitions(pd.to_datetime, format="%Y-%m-%d %H:%M:%S", meta=('datetime_issue', 'datetime64[ns]'))
#data['datetime_issue'] =  dd.to_datetime(data['datetime_issue'])
df['datetime_issue'] = df.apply(lambda x: get_datetime(x['Issue Date'], x['Violation Time']), axis=1, meta=('datetime_issue', 'datetime64[ns]'))

#print(len(df[df['datetime_issue'].isna()]))
df = df[(df['datetime_issue'] != 'nan')]


# read only last 1 million events for 2022, 2023
year = 2022 
current = 2023
#17mil rows total 
# data from 2008
header = pd.read_csv("data/nyc_permitted_events.csv", nrows=1)
skip_r = 17000000 - (current-year)*1500000
event_cols = ['Police Precinct', 'Start Date/Time', 'End Date/Time']
events = dd.read_csv("data/nyc_permitted_events.csv", assume_missing=True)

events.columns = header.columns
#events.columns = event_cols

events['parsed police'] = events.apply(lambda x: str(x['Police Precinct']).split(",")[0], axis=1, result_type='expand', meta=('parse', 'str'))

events['Start Date/Time'] = dd.to_datetime(events['Start Date/Time'], format='%m/%d/%Y %I:%M:%S %p',)
events['End Date/Time'] = dd.to_datetime(events['End Date/Time'], format='%m/%d/%Y %I:%M:%S %p',)

events['Start Date'] = events['Start Date/Time'].dt.date


events = events.rename(columns={"parsed police": "Issuer Precinct"})

events['event'] = 1
events = events[['Issuer Precinct', 'event', 'Start Date']]


df = df.rename(columns={"datetime_issue": "Start Date"})
df['Start Date'] = df['Start Date'].dt.date
print(df.columns)
events = events[events['Issuer Precinct'] != 'nan']

#events = events.dropna(subset=['Issuer Precinct'])
events['Issuer Precinct'] = events['Issuer Precinct'].astype(int)

events_g = events.groupby(['Issuer Precinct', 'Start Date']).count()
events_g = events_g.reset_index()

df = df.merge(events_g, how = "left", on=["Issuer Precinct", "Start Date"])

df.astype(str).to_parquet("with_events.parquet")