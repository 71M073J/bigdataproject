import pandas as pd
import json


import numpy as np
from dask import dataframe as dd
from distributed import Client
from dask_jobqueue import SLURMCluster




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



cluster = SLURMCluster(

            processes=8,
            cores=8,
            memory='8GB',
            scheduler_options={'dashboard_address': ':8087'},
            walltime="01:00:00"
          )

cluster.scale(jobs=2)
client = Client(cluster, timeout="120s")


#client = Client(cluster, timeout="120s")
cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time']


path = "/d/hpc/projects/FRI/bigdata/data/NYTickets/"

names = ["2023_april", "2022_full", "2021", "2020", "2019", "2018"]
year = 2023

for f_name in names:
    
    full_path = path+f_name+".csv"
    df = dd.read_csv(full_path, assume_missing = True, usecols=cols, dtype={
    'Issuer Precinct': np.float64, })
  
    df = df.repartition(npartitions=16)
    
    
    df['datetime_issue'] = df.apply(lambda x: get_datetime(x['Issue Date'], x['Violation Time']), axis=1, meta=('datetime_issue', 'datetime64[ns]'))

    #print(len(df[df['datetime_issue'].isna()]))
    df = df[(df['datetime_issue'] != 'nan')]
    df = augment_weather(df, year)
    df.columns = df.columns.astype(str)
    year -= 1
    
    new_name = "with_weather_dask_" + f_name + ".parquet"
    df.to_parquet(new_name)