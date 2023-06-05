import polars as pl
import pandas as pd
import json

from augment_weather import augment_weather

import numpy as np
from dask import dataframe as dd
from distributed import Client
from dask_jobqueue import SLURMCluster


cluster = SLURMCluster(

            processes=2,
            cores=8,
            memory='16GB',
            scheduler_options={'dashboard_address': ':8087'},
            walltime="01:00:00"
          )

cluster.scale(jobs=2)
client = Client(cluster, timeout="120s")


#client = Client(cluster, timeout="120s")
cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time']
df = dd.read_csv("data/2023_parking_violations.csv", assume_missing = True, usecols=cols,
dtype={
    'Issuer Precinct': np.float64,

})

df = df.repartition(npartitions=32)
# df2 = df2.repartition(npartitions=1)

print("read data")

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
df = augment_weather(df, 2023)

df[['datetime_issue',  'temp', 'humidity', 'snowdepth', 'windspeed', 'conditions', 'description']].to_pickle("with_weather_dask")


df['Street Name'] = df['Street Name'].astype(str)

f = open('data/street-suffix-abbreviations.json')
street_suffix = json.load(f)

def fix_street_abbreviation(x):
    abb = str(x).split()[-1]

    if abb in street_suffix:
        x = x.replace(abb, street_suffix[abb][0])
    return x

df["Street Name"] = df.apply(lambda row: fix_street_abbreviation(row['Street Name']), axis = 1, meta=('Street Name', 'str'))

schools = pd.read_csv('./data/school_locations.csv')
def street_from_address(x):
    # make sure the addresses are written correctly
    a = x.split(" ")[1:]
    return ' '.join(a)

schools['Street Name'] = schools.apply(lambda row: street_from_address(row['primary_address_line_1']), axis = 1)

# extract relevant categories
schools = schools[["Street Name", "Location_Category_Description", "Police_precinct"]]

schools = schools[["Street Name", "Location_Category_Description", "Police_precinct"]]

# note we are choosing to rename to issuer precinct. 
# we could also join via Violation Precinct - it would give us a different picture
schools = schools.rename(columns={"Police_precinct": "Issuer Precinct"}, errors="raise")
schools = schools.rename(columns={"Location_Category_Description": "Schools"}, errors="raise")

schools_g = schools.groupby(['Street Name', 'Issuer Precinct']).count()
schools_g = schools_g.reset_index()



df = dd.merge(df, schools_g, how = "left", on=["Street Name", "Issuer Precinct"])

df[['Street Name', 'Issuer Precinct', 'Schools']].to_pickle("with_schools_dask")



business = pd.read_csv('./data/legally_operating_businesses.csv')

# filter data - active businesses only, ONLY in new york
business = business[(business["License Type"] == "Business") & (business["License Status"] == "Active") & (business["Address State"] == "NY") & (business["Address Borough"] != "Outside NYC")]

business["Address Borough"] = business["Address Borough"].str.upper()

business["Address Street Name"] = business["Address Street Name"].str.upper()
business["Address Street Name"] = business["Address Street Name"].astype('str')

business["Address Street Name"] = business.apply(lambda row: fix_street_abbreviation(row['Address Street Name']), axis = 1)

business = business.rename(columns={"Address Street Name": "Street Name"}, errors="raise")

business['Business'] = 1

def county_to_borough(x):
    if (x == 'K') | (x.upper() == 'KINGS') | (x == 'BK'):
        return 'BROOKLYN'
    elif (x == 'BX') | (x.upper() == 'BRONX'):
        return 'BRONX'
    elif (x == 'R') | (x.upper() == 'RICH') | (x == 'ST') | (x.upper() == 'RICHM'):
        return 'STATEN ISLAND'
    elif (x == 'NY') | (x == 'MN'):
        return 'MANHATTAN'
    elif (x == 'Q') | (x == 'QN') | (x.upper() == 'QNS'):
        return 'QUEENS'
    else:
        return x    

#df["Violation County"] = df["Violation County"].astype('str')
df["Address Borough"] = df.apply(lambda row: county_to_borough(row['Violation County']), axis = 1)

business = business[["Street Name", "Address Borough", "Business"]]

business_g = business.groupby(['Street Name', 'Address Borough']).count()
business_g = business_g.reset_index()

df = pd.merge(df, business_g, how = "left", on=["Street Name", "Address Borough"])

df['Business'] = df['Business'].fillna(0)

df[['Street Name', 'Address Borough', 'Business']].to_pickle("with_business_dask")




